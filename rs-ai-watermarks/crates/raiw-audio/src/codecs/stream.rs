//! `_stream_neural_track`: single-pass resynthesis up to 30 s, raised-cosine overlap windows beyond.

use std::f64::consts::PI;

use crate::dsp::numpy::cos_f32;
use crate::error::{AudioError, Result};
use crate::frames::Frames;
use crate::scan::{MAX_WHOLE_SECONDS, MAX_WHOLE_VALUES};

enum Phase {
    Start,
    Windowed {
        prev_tail: Option<Frames>,
        in_overlap: Option<Frames>,
    },
    Done,
}

/// Streams a whole-track transform with bounded memory and crossfaded window seams.
pub struct NeuralStream<I, F> {
    source: I,
    transform: F,
    label: &'static str,
    allow_surround: bool,
    sample_rate: usize,
    pending: Frames,
    channels: Option<usize>,
    eof: bool,
    window: usize,
    overlap: usize,
    fade_in: Vec<f32>,
    phase: Phase,
}

impl<I, F> std::fmt::Debug for NeuralStream<I, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NeuralStream")
            .field("label", &self.label)
            .field("window", &self.window)
            .finish_non_exhaustive()
    }
}

/// `_stream_neural_track(chunks, sample_rate, label, transform, allow_surround=...)`.
pub fn stream_neural_track<I, F>(
    source: I,
    sample_rate: u32,
    label: &'static str,
    transform: F,
    allow_surround: bool,
) -> NeuralStream<I, F>
where
    I: Iterator<Item = Result<Frames>>,
    F: FnMut(&Frames) -> Result<Frames>,
{
    NeuralStream {
        source,
        transform,
        label,
        allow_surround,
        sample_rate: sample_rate as usize,
        pending: Frames::empty(1),
        channels: None,
        eof: false,
        window: 0,
        overlap: 0,
        fade_in: Vec::new(),
        phase: Phase::Start,
    }
}

/// The raised-cosine fade `0.5 - 0.5 cos(pi k / (n + 1))`, `k = 1..=n`, in NumPy's float32 arithmetic.
pub fn fade_in(overlap: usize) -> Vec<f32> {
    let (pi, denom) = (PI as f32, (overlap + 1) as f32);
    (1..=overlap)
        .map(|k| 0.5 - 0.5 * cos_f32(pi * k as f32 / denom))
        .collect()
}

impl<I, F> NeuralStream<I, F>
where
    I: Iterator<Item = Result<Frames>>,
    F: FnMut(&Frames) -> Result<Frames>,
{
    fn pull(&mut self) -> Result<bool> {
        if self.eof {
            return Ok(false);
        }
        let Some(chunk) = self.source.next().transpose()? else {
            self.eof = true;
            return Ok(false);
        };
        let allowed = if self.allow_surround {
            matches!(chunk.channels(), 1 | 2 | 6 | 8)
        } else {
            matches!(chunk.channels(), 1 | 2)
        };
        if !allowed {
            let expected = if self.allow_surround {
                "mono, stereo, 5.1, or 7.1"
            } else {
                "mono or stereo"
            };
            return Err(AudioError::invalid(format!(
                "{} processing requires {expected}",
                self.label
            )));
        }
        match self.channels {
            None => {
                self.channels = Some(chunk.channels());
                self.pending = Frames::empty(chunk.channels());
            }
            Some(c) if c != chunk.channels() => {
                return Err(AudioError::invalid(format!(
                    "{} processing requires a constant channel count",
                    self.label
                )));
            }
            Some(_) => {}
        }
        self.pending.append(&chunk);
        Ok(true)
    }

    fn pop(&mut self, frames: usize) -> Frames {
        let head = self.pending.slice(0..frames);
        self.pending = self.pending.slice(frames..self.pending.frames());
        head
    }

    fn start(&mut self) -> Result<Option<Frames>> {
        if !self.pull()? || self.pending.is_empty() {
            return Err(AudioError::invalid(format!(
                "{} requires nonempty audio",
                self.label
            )));
        }
        let channels = self.channels.unwrap_or(1);
        let whole_limit = (MAX_WHOLE_SECONDS * self.sample_rate).min((MAX_WHOLE_VALUES / channels).max(1));
        while self.pending.frames() <= whole_limit && self.pull()? {}
        if self.eof && self.pending.frames() <= whole_limit {
            self.phase = Phase::Done;
            let all = self.pop(self.pending.frames());
            return (self.transform)(&all).map(Some);
        }
        self.window = (20 * self.sample_rate).min((whole_limit * 2 / 3).max(4));
        self.overlap = (2 * self.sample_rate).min(self.window / 4).max(1);
        self.fade_in = fade_in(self.overlap);
        self.phase = Phase::Windowed {
            prev_tail: None,
            in_overlap: None,
        };
        self.windowed()
    }

    fn blend(&self, prev_tail: &Frames, win_out: &Frames) -> Frames {
        let channels = win_out.channels();
        let mut head = win_out.clone();
        let out = head.as_mut_slice();
        for (k, &fi) in self.fade_in.iter().enumerate() {
            let fo = 1.0 - fi;
            for c in 0..channels {
                let i = k * channels + c;
                out[i] = prev_tail.as_slice()[i] * fo + win_out.as_slice()[i] * fi;
            }
        }
        head
    }

    fn windowed(&mut self) -> Result<Option<Frames>> {
        let Phase::Windowed {
            prev_tail,
            in_overlap,
        } = std::mem::replace(&mut self.phase, Phase::Done)
        else {
            return Ok(None);
        };
        let need = if in_overlap.is_none() {
            self.window + self.overlap
        } else {
            self.window
        };
        while self.pending.frames() < need && self.pull()? {}
        if self.pending.is_empty() {
            return Ok(prev_tail);
        }
        let step_in = self.pop(need.min(self.pending.frames()));
        let win_in = match in_overlap {
            Some(mut overlap) => {
                overlap.append(&step_in);
                overlap
            }
            None => step_in,
        };
        let win_out = (self.transform)(&win_in)?;
        if win_out.frames() != win_in.frames() || win_out.channels() != win_in.channels() {
            return Err(AudioError::failed(format!(
                "{} returned invalid window shape ({}, {})",
                self.label,
                win_out.frames(),
                win_out.channels()
            )));
        }
        let head = match &prev_tail {
            None => win_out.clone(),
            Some(tail) => self.blend(tail, &win_out),
        };
        if self.eof && self.pending.is_empty() {
            return Ok(Some(head));
        }
        let keep = head.frames() - self.overlap;
        self.phase = Phase::Windowed {
            prev_tail: Some(win_out.tail(self.overlap)),
            in_overlap: Some(win_in.tail(self.overlap)),
        };
        Ok(Some(head.slice(0..keep)))
    }
}

impl<I, F> Iterator for NeuralStream<I, F>
where
    I: Iterator<Item = Result<Frames>>,
    F: FnMut(&Frames) -> Result<Frames>,
{
    type Item = Result<Frames>;

    fn next(&mut self) -> Option<Self::Item> {
        let step = match self.phase {
            Phase::Start => self.start(),
            Phase::Windowed { .. } => self.windowed(),
            Phase::Done => return None,
        };
        match step {
            Ok(Some(frames)) => Some(Ok(frames)),
            Ok(None) => {
                self.phase = Phase::Done;
                None
            }
            Err(err) => {
                self.phase = Phase::Done;
                Some(Err(err))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(frames: usize) -> Frames {
        Frames::mono((0..frames).map(|v| (v % 97) as f32 / 97.0).collect())
    }

    #[test]
    fn identity_transform_is_lossless_across_windows() {
        let rate = 100;
        let x = ramp(rate * 75 + 13);
        let chunks: Vec<Result<Frames>> = (0..x.frames())
            .step_by(800)
            .map(|s| Ok(x.slice(s..s + 800)))
            .collect();
        let mut calls = 0;
        let out: Vec<Frames> = stream_neural_track(
            chunks.into_iter(),
            rate as u32,
            "TEST",
            |w: &Frames| {
                calls += 1;
                Ok(w.clone())
            },
            false,
        )
        .collect::<Result<_>>()
        .unwrap();
        let joined = Frames::concat(&out, 1);
        assert_eq!(joined.frames(), x.frames());
        assert!(
            joined
                .as_slice()
                .iter()
                .zip(x.as_slice())
                .all(|(a, b)| (a - b).abs() < 1e-6)
        );
        assert!(calls > 1);
    }

    #[test]
    fn short_tracks_run_once() {
        let x = ramp(250);
        let out: Vec<Frames> = stream_neural_track(
            std::iter::once(Ok(x.clone())),
            100,
            "TEST",
            |w: &Frames| Ok(w.scale(0.5)),
            false,
        )
        .collect::<Result<_>>()
        .unwrap();
        assert_eq!(out, vec![x.scale(0.5)]);
    }

    #[test]
    fn empty_input_is_rejected() {
        let mut stream =
            stream_neural_track(std::iter::empty(), 100, "DAC", |w: &Frames| Ok(w.clone()), false);
        assert_eq!(
            stream.next().unwrap().unwrap_err().to_string(),
            "DAC requires nonempty audio"
        );
    }
}
