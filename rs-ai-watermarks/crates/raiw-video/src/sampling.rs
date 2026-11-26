//! The uniform frame-sampling rule shared by regeneration and the fidelity probe
//! (`select_sampled_indices` / `_iter_sampled_frames` in `video_invisible.py`).
//!
//! A frame is taken when `timestamp + 1e-9 >= next_sample_time`; the comparison, phase and period
//! are part of the contract.

use std::collections::HashSet;
use std::path::Path;

use raiw_cv::resize::Resizable;
use raiw_cv::{Interpolation, Mat, resize};

use crate::decode::{FramePixel, VideoReader};
use crate::error::Result;

/// `select_sampled_indices`: frame indices chosen by the relative-timestamp schedule.
pub fn select_sampled_indices(timestamps: &[f64], effective_fps: f64, duration: Option<f64>) -> Vec<usize> {
    let Some(&base) = timestamps.first() else {
        return Vec::new();
    };
    let period = 1.0 / effective_fps;
    let mut next = 0.0;
    let mut selected = Vec::new();
    for (index, &ts) in timestamps.iter().enumerate() {
        let relative = ts - base;
        if duration.is_some_and(|d| relative + 1e-9 >= d) {
            break;
        }
        if relative + 1e-9 >= next {
            selected.push(index);
            next += period;
        }
    }
    selected
}

/// Options for [`SampledFrames`].
#[derive(Clone, Debug)]
pub struct SamplingOptions {
    pub source_fps: f64,
    pub duration: Option<f64>,
    pub effective_fps: f64,
    /// Output `(width, height)`; frames of another size are Lanczos-resized.
    pub size: (usize, usize),
    pub timestamps: Option<Vec<f64>>,
}

/// The per-frame keep/stop decision of `_iter_sampled_frames`, shared by decoded and pushed streams.
#[derive(Clone, Debug)]
pub struct FrameSampler {
    options: SamplingOptions,
    selected: Option<HashSet<usize>>,
    max_selected: Option<usize>,
    next_sample_time: f64,
    base_ts: f64,
    frame_index: usize,
}

impl FrameSampler {
    pub fn new(options: SamplingOptions) -> Self {
        let selected = options.timestamps.as_ref().map(|ts| {
            select_sampled_indices(ts, options.effective_fps, options.duration)
                .into_iter()
                .collect::<HashSet<_>>()
        });
        let max_selected = selected.as_ref().and_then(|set| set.iter().copied().max());
        let base_ts = options
            .timestamps
            .as_ref()
            .and_then(|ts| ts.first().copied())
            .unwrap_or(0.0);
        Self {
            options,
            selected,
            max_selected,
            next_sample_time: 0.0,
            base_ts,
            frame_index: 0,
        }
    }

    pub fn size(&self) -> (usize, usize) {
        self.options.size
    }

    /// Whether the next frame is kept; `None` once the schedule has ended.
    pub fn next_wanted(&mut self) -> Option<bool> {
        let index = self.frame_index;
        let want = if let Some(selected) = &self.selected {
            if self.max_selected.is_none_or(|max| index > max) {
                return None;
            }
            selected.contains(&index)
        } else {
            let timestamp = match &self.options.timestamps {
                Some(ts) if index < ts.len() => ts[index] - self.base_ts,
                _ => index as f64 / self.options.source_fps,
            };
            if self.options.duration.is_some_and(|d| timestamp + 1e-9 >= d) {
                return None;
            }
            let want = timestamp + 1e-9 >= self.next_sample_time;
            if want {
                self.next_sample_time += 1.0 / self.options.effective_fps;
            }
            want
        };
        self.frame_index += 1;
        Some(want)
    }

    /// Lanczos-resize a kept frame to the sampling geometry.
    pub fn fit<T: Resizable>(&self, frame: Mat<T>) -> Result<Mat<T>> {
        let (w, h) = self.options.size;
        Ok(if frame.size() == (w, h) {
            frame
        } else {
            resize(&frame, w, h, Interpolation::Lanczos4)?
        })
    }
}

/// `_iter_sampled_frames` over an 8-bit or 16-bit decode, without retaining dropped frames.
#[derive(Debug)]
pub struct SampledFrames<T: FramePixel = u8> {
    reader: VideoReader<T>,
    sampler: FrameSampler,
    done: bool,
}

impl SampledFrames<u8> {
    pub fn open(source: &Path, options: SamplingOptions) -> Result<Self> {
        Self::open_depth(source, options)
    }
}

impl<T: FramePixel + Resizable> SampledFrames<T> {
    pub fn open_depth(source: &Path, options: SamplingOptions) -> Result<Self> {
        Ok(Self {
            reader: VideoReader::open(source)?,
            sampler: FrameSampler::new(options),
            done: false,
        })
    }

    fn step(&mut self) -> Result<Option<Mat<T>>> {
        while !self.done {
            let Some(want) = self.sampler.next_wanted() else {
                self.done = true;
                break;
            };
            if !want {
                if !self.reader.grab()? {
                    self.done = true;
                }
                continue;
            }
            let Some(frame) = self.reader.read()? else {
                self.done = true;
                break;
            };
            return self.sampler.fit(frame).map(Some);
        }
        Ok(None)
    }
}

impl<T: FramePixel + Resizable> Iterator for SampledFrames<T> {
    type Item = Result<Mat<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.step().transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_keeps_the_boundary_frame() {
        let ts: Vec<f64> = (0..10).map(|i| f64::from(i) / 30.0).collect();
        assert_eq!(select_sampled_indices(&ts, 15.0, None), vec![0, 2, 4, 6, 8]);
        assert_eq!(
            select_sampled_indices(&ts, 30.0, Some(0.2)),
            vec![0, 1, 2, 3, 4, 5]
        );
        assert_eq!(select_sampled_indices(&ts, 10.0, None), vec![0, 3, 6, 9]);
    }
}
