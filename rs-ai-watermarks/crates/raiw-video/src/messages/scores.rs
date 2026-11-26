//! Bounded known-message correlation over every frame and complete windows
//! (`_internal/video_message_scores.py`).
//!
//! The reference must be known independently of decoding the tested file; these criteria do
//! not infer arbitrary watermark presence or AI origin.

use std::collections::VecDeque;

use indexmap::IndexMap;
use serde::Serialize;

use crate::error::{Result, VideoError};

pub const AGGREGATIONS: [&str; 4] = ["avg", "squared_avg", "l1norm_avg", "l2norm_avg"];
pub const WINDOW_FRAMES: usize = 12;
pub const THRESHOLD: f64 = 0.9;
pub const SUPPORTED_MESSAGE_BITS: [usize; 2] = [256, 1024];

/// Scores at native, half-open frame indices.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MessageWindow {
    pub start_frame: usize,
    pub end_frame: usize,
    pub scores: IndexMap<String, f64>,
}

/// Known-target correlation under a named experimental receiver profile.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VideoMessageReading {
    pub frames: usize,
    pub whole_clip: IndexMap<String, f64>,
    pub windows: Vec<MessageWindow>,
    pub per_frame_max: f64,
    pub max_bit_accuracy: f64,
}

impl VideoMessageReading {
    pub fn target_matched(&self) -> bool {
        self.max_bit_accuracy >= THRESHOLD
    }
}

/// `reference_bits`: independently known message bits from their exact hexadecimal form.
pub fn reference_bits(reference_hex: &str, num_bits: usize) -> Result<Vec<bool>> {
    if !SUPPORTED_MESSAGE_BITS.contains(&num_bits) {
        return Err(VideoError::value(format!(
            "Unsupported video message bit width: {num_bits}"
        )));
    }
    let digits = num_bits / 4;
    if reference_hex.chars().count() != digits || !reference_hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(VideoError::value(format!(
            "The independent reference must contain exactly {digits} hexadecimal digits"
        )));
    }
    let mut bits = Vec::with_capacity(num_bits);
    for c in reference_hex.chars() {
        let nibble = c.to_digit(16).unwrap_or_default();
        bits.extend((0..4).rev().map(|shift| (nibble >> shift) & 1 == 1));
    }
    Ok(bits)
}

/// Four weighted aggregations of one frame: `x`, `x|x|`, `x * sum|x|`, `x * ||x||`.
type Terms = [Vec<f64>; 4];

fn weighted_terms(values: &[f64]) -> Result<Terms> {
    let l1: f64 = values.iter().map(|v| v.abs()).sum();
    let l2 = values.iter().map(|v| v * v).sum::<f64>().sqrt();
    let terms = [
        values.to_vec(),
        values.iter().map(|v| v * v.abs()).collect(),
        values.iter().map(|v| v * l1).collect(),
        values.iter().map(|v| v * l2).collect(),
    ];
    if !l1.is_finite() || !l2.is_finite() || terms.iter().flatten().any(|v| !v.is_finite()) {
        return Err(VideoError::runtime("Video message aggregation overflowed"));
    }
    Ok(terms)
}

fn add_terms(total: &mut Terms, terms: &Terms) {
    for (sum, part) in total.iter_mut().zip(terms) {
        for (s, v) in sum.iter_mut().zip(part) {
            *s += v;
        }
    }
}

fn sum_terms<'a>(rows: impl Iterator<Item = &'a Terms>, bits: usize) -> Result<Terms> {
    let mut total: Terms = std::array::from_fn(|_| vec![0.0; bits]);
    for terms in rows {
        add_terms(&mut total, terms);
    }
    if total.iter().flatten().any(|v| !v.is_finite()) {
        return Err(VideoError::runtime("Video message aggregation overflowed"));
    }
    Ok(total)
}

/// `MessageAccumulator`: streams float64 reductions, retaining at most a window and one batch.
#[derive(Debug)]
pub struct MessageAccumulator {
    expected: Vec<bool>,
    sums: Terms,
    pending: VecDeque<Terms>,
    count: usize,
    next_start: usize,
    frame_max: f64,
    windows: Vec<MessageWindow>,
    closed: bool,
}

impl MessageAccumulator {
    pub fn new(expected: Vec<bool>) -> Result<Self> {
        let bits = expected.len();
        if !SUPPORTED_MESSAGE_BITS.contains(&bits) {
            return Err(VideoError::value(
                "An independently known 256-bit or 1024-bit reference is required",
            ));
        }
        Ok(Self {
            expected,
            sums: std::array::from_fn(|_| vec![0.0; bits]),
            pending: VecDeque::new(),
            count: 0,
            next_start: 0,
            frame_max: 0.0,
            windows: Vec::new(),
            closed: false,
        })
    }

    fn bits(&self) -> usize {
        self.expected.len()
    }

    fn scores(&self, sums: &Terms) -> IndexMap<String, f64> {
        let bits = self.bits() as f64;
        AGGREGATIONS
            .iter()
            .zip(sums)
            .map(|(name, sum)| {
                let count = sum
                    .iter()
                    .zip(&self.expected)
                    .filter(|(s, e)| (**s > 0.0) == **e)
                    .count();
                ((*name).to_owned(), count as f64 / bits)
            })
            .collect()
    }

    /// Check a nonempty, sequential batch; an invalid batch poisons the stream.
    pub fn update(&mut self, logits: &[Vec<f32>]) -> Result<()> {
        if self.closed {
            return Err(VideoError::runtime(
                "Video message accumulator is already finalized",
            ));
        }
        self.closed = true;
        let bits = self.bits();
        if logits.is_empty()
            || logits
                .iter()
                .any(|row| row.len() != bits || row.iter().any(|v| !v.is_finite()))
        {
            return Err(VideoError::runtime(
                "Video message decoder returned invalid raw logits",
            ));
        }
        let rows: Vec<Terms> = logits
            .iter()
            .map(|row| weighted_terms(&row.iter().map(|v| f64::from(*v)).collect::<Vec<_>>()))
            .collect::<Result<_>>()?;
        let batch = sum_terms(rows.iter(), bits)?;
        add_terms(&mut self.sums, &batch);
        if self.sums.iter().flatten().any(|v| !v.is_finite()) {
            return Err(VideoError::runtime(
                "Video message accumulation returned nonfinite values",
            ));
        }
        for row in logits {
            let matches = row
                .iter()
                .zip(&self.expected)
                .filter(|(v, e)| (**v > 0.0) == **e)
                .count();
            self.frame_max = self.frame_max.max(matches as f64 / bits as f64);
        }
        self.count += logits.len();
        self.pending.extend(rows);
        while self.pending.len() >= WINDOW_FRAMES {
            let window = sum_terms(self.pending.iter().take(WINDOW_FRAMES), bits)?;
            let scores = self.scores(&window);
            self.windows.push(MessageWindow {
                start_frame: self.next_start,
                end_frame: self.next_start + WINDOW_FRAMES,
                scores,
            });
            self.next_start += 1;
            self.pending.pop_front();
        }
        self.closed = false;
        Ok(())
    }

    /// Finalize only after the complete video stream has been decoded.
    pub fn finish(mut self) -> Result<VideoMessageReading> {
        if self.closed || self.count == 0 {
            return Err(VideoError::runtime(
                "Video message decoding has no complete, unfinalized stream",
            ));
        }
        self.closed = true;
        if self.windows.is_empty() {
            let window = sum_terms(self.pending.iter(), self.bits())?;
            let scores = self.scores(&window);
            self.windows.push(MessageWindow {
                start_frame: 0,
                end_frame: self.count,
                scores,
            });
        }
        let whole = self.scores(&self.sums);
        let maximum = self
            .windows
            .iter()
            .flat_map(|w| w.scores.values())
            .chain(whole.values())
            .fold(self.frame_max, |a, b| a.max(*b));
        Ok(VideoMessageReading {
            frames: self.count,
            whole_clip: whole,
            windows: self.windows,
            per_frame_max: self.frame_max,
            max_bit_accuracy: maximum,
        })
    }
}
