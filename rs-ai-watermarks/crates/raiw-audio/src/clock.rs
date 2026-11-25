//! Rational, half-open audio clocks for full-track resynthesis (`_internal/audio_resampling.py`).
//!
//! The model grid covers the source duration; the inverse keeps exactly the native instants
//! `k / native_rate < T`. Zero-phase filtering does not preserve edge amplitudes exactly.

use serde::Serialize;

use crate::dsp::resample::{gcd, resample_poly};
use crate::frames::Frames;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ClockError {
    #[error("{0} must be a positive integer")]
    NotPositive(&'static str),
    #[error("Expected a floating audio array with {0} frames")]
    Frames(usize),
    #[error("Audio samples must be finite in float32")]
    NonFinite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RationalClock {
    pub native_frames: usize,
    pub native_rate: u32,
    pub model_rate: u32,
    pub hop: usize,
    pub minimum_model_frames: usize,
}

/// `RationalClock.record()`, every boundary decision without rounding the source clock.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ClockRecord {
    pub native_frames: usize,
    pub native_rate: u32,
    pub duration: [u64; 2],
    pub model_rate: u32,
    pub forward_ratio: [usize; 2],
    pub model_frames: usize,
    pub model_hop: usize,
    pub model_right_reflection: usize,
    pub inverse_frames: usize,
    pub excluded_inverse_indices: [usize; 2],
    pub boundary_policy: &'static str,
}

impl RationalClock {
    pub const DAC_HOP: usize = 512;

    pub fn new(
        native_frames: usize,
        native_rate: u32,
        model_rate: u32,
        hop: usize,
        minimum_model_frames: usize,
    ) -> Result<Self, ClockError> {
        let checks = [
            ("native_frames", native_frames),
            ("native_rate", native_rate as usize),
            ("model_rate", model_rate as usize),
            ("hop", hop),
            ("minimum_model_frames", minimum_model_frames),
        ];
        if let Some((name, _)) = checks.iter().find(|(_, v)| *v == 0) {
            return Err(ClockError::NotPositive(name));
        }
        Ok(Self {
            native_frames,
            native_rate,
            model_rate,
            hop,
            minimum_model_frames,
        })
    }

    /// The Python defaults: 44.1 kHz model grid, hop 512, at least 1024 model frames.
    pub fn standard(native_frames: usize, native_rate: u32) -> Result<Self, ClockError> {
        Self::new(native_frames, native_rate, 44_100, 512, 1024)
    }

    /// `(numerator, denominator)` of the source duration in lowest terms.
    pub fn duration(&self) -> (u64, u64) {
        let g = gcd(self.native_frames, self.native_rate as usize) as u64;
        (self.native_frames as u64 / g, u64::from(self.native_rate) / g)
    }

    pub fn ratio(&self) -> (usize, usize) {
        let (native, model) = (self.native_rate as usize, self.model_rate as usize);
        let g = gcd(native, model);
        (model / g, native / g)
    }

    pub fn model_frames(&self) -> usize {
        let (up, down) = self.ratio();
        (self.native_frames * up).div_ceil(down)
    }

    pub fn inverse_frames(&self) -> usize {
        let (up, down) = self.ratio();
        (self.model_frames() * down).div_ceil(up)
    }

    pub fn padded_frames(&self) -> usize {
        self.model_frames()
            .max(self.minimum_model_frames)
            .div_ceil(self.hop)
            * self.hop
    }

    pub fn record(&self) -> ClockRecord {
        let (num, den) = self.duration();
        let (up, down) = self.ratio();
        ClockRecord {
            native_frames: self.native_frames,
            native_rate: self.native_rate,
            duration: [num, den],
            model_rate: self.model_rate,
            forward_ratio: [up, down],
            model_frames: self.model_frames(),
            model_hop: self.hop,
            model_right_reflection: self.padded_frames() - self.model_frames(),
            inverse_frames: self.inverse_frames(),
            excluded_inverse_indices: [self.native_frames, self.inverse_frames()],
            boundary_policy: "constant-zero FIR exterior; right model reflection; inverse native half-open grid",
        }
    }

    /// Resample all native input samples, retaining `ceil(T * model_rate)`.
    pub fn forward(&self, samples: &Frames) -> Result<Frames, ClockError> {
        check(samples, self.native_frames)?;
        let (up, down) = self.ratio();
        let out = map_channels(samples, |c| resample_poly(c, up, down));
        check(&out, self.model_frames())?;
        Ok(out)
    }

    /// Right-reflect to a hop boundary (`np.pad(..., mode="reflect")`).
    pub fn pad_model(&self, samples: &Frames) -> Result<Frames, ClockError> {
        check(samples, self.model_frames())?;
        let extra = self.padded_frames() - self.model_frames();
        Ok(map_channels(samples, |c| reflect_pad_right(c, extra)))
    }

    /// Reject wrong model geometry before removing declared right context.
    pub fn unpad_model(&self, samples: &Frames) -> Result<Frames, ClockError> {
        check(samples, self.padded_frames())?;
        Ok(samples.slice(0..self.model_frames()))
    }

    /// Retain exactly the native sample instants `k / native_rate < T`.
    pub fn inverse(&self, samples: &Frames) -> Result<Frames, ClockError> {
        check(samples, self.model_frames())?;
        let (up, down) = self.ratio();
        let out = map_channels(samples, |c| resample_poly(c, down, up));
        check(&out, self.inverse_frames())?;
        let kept = out.slice(0..self.native_frames);
        check(&kept, self.native_frames)?;
        Ok(kept)
    }
}

fn check(samples: &Frames, frames: usize) -> Result<(), ClockError> {
    if samples.frames() != frames {
        return Err(ClockError::Frames(frames));
    }
    if !samples.is_finite() {
        return Err(ClockError::NonFinite);
    }
    Ok(())
}

fn map_channels(samples: &Frames, f: impl Fn(&[f32]) -> Vec<f32> + Sync) -> Frames {
    use rayon::prelude::*;
    let channels: Vec<Vec<f32>> = samples.to_channels().par_iter().map(|c| f(c)).collect();
    Frames::from_channels(&channels)
}

/// NumPy's `reflect` padding on the right, repeating for pads longer than the signal.
pub fn reflect_pad_right(x: &[f32], extra: usize) -> Vec<f32> {
    let n = x.len();
    let mut out = Vec::with_capacity(n + extra);
    out.extend_from_slice(x);
    if n == 1 {
        out.resize(n + extra, x[0]);
        return out;
    }
    let period = 2 * (n - 1);
    for k in 1..=extra {
        let pos = (n - 1 + k) % period;
        let idx = if pos < n { pos } else { period - pos };
        out.push(x[idx]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_matches_python_clock() {
        let clock = RationalClock::standard(480_240, 48_000).unwrap();
        assert_eq!((clock.model_frames(), clock.inverse_frames()), (441_221, 480_241));
        assert_eq!(clock.duration(), (2001, 200));
        assert_eq!(clock.record().excluded_inverse_indices, [480_240, 480_241]);
    }

    #[test]
    fn reflection_matches_numpy() {
        assert_eq!(
            reflect_pad_right(&[1.0, 2.0, 3.0], 5),
            vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0]
        );
        assert_eq!(reflect_pad_right(&[7.0], 3), vec![7.0; 4]);
    }

    #[test]
    fn invalid_clocks() {
        assert_eq!(
            RationalClock::standard(0, 48_000),
            Err(ClockError::NotPositive("native_frames"))
        );
    }
}
