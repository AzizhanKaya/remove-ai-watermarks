//! Interleaved float32 sample frames, the Rust form of NumPy `(frames, channels)` arrays.

use std::ops::Range;

use crate::dsp::numpy::row_mean_f32;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FramesError {
    #[error("Audio must contain at least one channel")]
    NoChannels,
    #[error("Interleaved samples ({samples}) are not a whole number of {channels}-channel frames")]
    Partial { samples: usize, channels: usize },
}

/// C-ordered `(frames, channels)` float32 samples.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Frames {
    data: Vec<f32>,
    channels: usize,
}

impl Frames {
    pub fn new(data: Vec<f32>, channels: usize) -> Result<Self, FramesError> {
        if channels == 0 {
            return Err(FramesError::NoChannels);
        }
        if !data.len().is_multiple_of(channels) {
            return Err(FramesError::Partial {
                samples: data.len(),
                channels,
            });
        }
        Ok(Self { data, channels })
    }

    pub fn zeros(frames: usize, channels: usize) -> Self {
        Self {
            data: vec![0.0; frames * channels.max(1)],
            channels: channels.max(1),
        }
    }

    pub fn empty(channels: usize) -> Self {
        Self::zeros(0, channels)
    }

    /// `np.column_stack` of equally long channels.
    pub fn from_channels(channels: &[Vec<f32>]) -> Self {
        let count = channels.len().max(1);
        let frames = channels.first().map_or(0, Vec::len);
        let mut data = vec![0.0; frames * count];
        for (c, channel) in channels.iter().enumerate() {
            for (i, &v) in channel.iter().take(frames).enumerate() {
                data[i * count + c] = v;
            }
        }
        Self {
            data,
            channels: count,
        }
    }

    pub fn mono(samples: Vec<f32>) -> Self {
        Self {
            data: samples,
            channels: 1,
        }
    }

    pub const fn channels(&self) -> usize {
        self.channels
    }

    pub fn frames(&self) -> usize {
        self.data.len() / self.channels
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn into_vec(self) -> Vec<f32> {
        self.data
    }

    pub fn row(&self, frame: usize) -> &[f32] {
        &self.data[frame * self.channels..(frame + 1) * self.channels]
    }

    pub fn rows(&self) -> std::slice::ChunksExact<'_, f32> {
        self.data.chunks_exact(self.channels)
    }

    pub fn channel(&self, index: usize) -> Vec<f32> {
        self.data
            .iter()
            .skip(index)
            .step_by(self.channels)
            .copied()
            .collect()
    }

    pub fn to_channels(&self) -> Vec<Vec<f32>> {
        (0..self.channels).map(|c| self.channel(c)).collect()
    }

    /// The `x.mean(axis=1)` downmix.
    pub fn downmix(&self) -> Vec<f32> {
        if self.channels == 1 {
            return self.data.clone();
        }
        self.rows().map(row_mean_f32).collect()
    }

    /// Frames `range`, clamped to the available frames.
    #[must_use]
    pub fn slice(&self, range: Range<usize>) -> Self {
        let end = range.end.min(self.frames());
        let start = range.start.min(end);
        Self {
            data: self.data[start * self.channels..end * self.channels].to_vec(),
            channels: self.channels,
        }
    }

    /// The last `count` frames (all of them when fewer exist).
    #[must_use]
    pub fn tail(&self, count: usize) -> Self {
        let frames = self.frames();
        self.slice(frames.saturating_sub(count)..frames)
    }

    /// Channels `range` of every frame.
    #[must_use]
    pub fn select_channels(&self, range: Range<usize>) -> Self {
        let width = range.len();
        let mut data = Vec::with_capacity(self.frames() * width);
        for row in self.rows() {
            data.extend_from_slice(&row[range.clone()]);
        }
        Self {
            data,
            channels: width.max(1),
        }
    }

    pub fn append(&mut self, other: &Self) {
        debug_assert_eq!(self.channels, other.channels);
        self.data.extend_from_slice(&other.data);
    }

    pub fn concat(parts: &[Self], channels: usize) -> Self {
        let mut data = Vec::with_capacity(parts.iter().map(|p| p.data.len()).sum());
        for part in parts {
            data.extend_from_slice(&part.data);
        }
        Self {
            data,
            channels: channels.max(1),
        }
    }

    /// `np.concatenate(..., axis=1)` of equally long blocks.
    pub fn hstack(blocks: &[Self]) -> Self {
        let frames = blocks.first().map_or(0, Self::frames);
        let channels: usize = blocks.iter().map(Self::channels).sum();
        let mut data = Vec::with_capacity(frames * channels);
        for i in 0..frames {
            for block in blocks {
                data.extend_from_slice(block.row(i));
            }
        }
        Self {
            data,
            channels: channels.max(1),
        }
    }

    pub fn is_finite(&self) -> bool {
        self.data.iter().all(|v| v.is_finite())
    }

    /// `float(np.max(np.abs(x)))`, NaN-propagating like NumPy.
    pub fn peak(&self) -> f32 {
        let mut peak = 0.0f32;
        for &v in &self.data {
            if v.is_nan() {
                return f32::NAN;
            }
            peak = peak.max(v.abs());
        }
        peak
    }

    #[must_use]
    pub fn map(&self, f: impl Fn(f32) -> f32) -> Self {
        Self {
            data: self.data.iter().map(|&v| f(v)).collect(),
            channels: self.channels,
        }
    }

    #[must_use]
    pub fn scale(&self, gain: f32) -> Self {
        self.map(|v| v * gain)
    }

    /// Little-endian bytes of the samples (`ndarray.astype("<f4").tobytes()`).
    pub fn to_le_bytes(&self) -> Vec<u8> {
        self.data.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channels_roundtrip() {
        let frames = Frames::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2).unwrap();
        assert_eq!(frames.channel(1), vec![2.0, 4.0, 6.0]);
        assert_eq!(Frames::from_channels(&frames.to_channels()), frames);
        assert_eq!(frames.downmix(), vec![1.5, 3.5, 5.5]);
        assert_eq!(frames.tail(2).as_slice(), &[3.0, 4.0, 5.0, 6.0]);
        assert_eq!(frames.select_channels(1..2).as_slice(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn rejects_partial_frames() {
        assert!(Frames::new(vec![1.0; 3], 2).is_err());
        assert!(Frames::new(vec![], 0).is_err());
    }
}
