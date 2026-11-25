//! `scipy.signal.resample_poly` with `padtype="constant"`, reproducing `upfirdn`'s accumulation order.

use rayon::prelude::*;

use super::window::firwin_kaiser;

const KAISER_BETA: f64 = 5.0;
const PARALLEL_BLOCK: usize = 1 << 13;

pub const fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// A polyphase rational resampler, designed once per reduced `(up, down)` ratio.
#[derive(Debug, Clone)]
pub struct PolyphaseResampler {
    up: usize,
    down: usize,
    /// Transposed, flipped taps (`upfirdn._pad_h`): phase `p` owns `[p * hpp, (p + 1) * hpp)`.
    phases: Vec<f32>,
    taps_per_phase: usize,
    pre_remove: usize,
}

impl PolyphaseResampler {
    /// Design the filter for `up / down`; `None` for an identity ratio.
    pub fn new(up: usize, down: usize) -> Option<Self> {
        assert!(up >= 1 && down >= 1, "resampling factors must be positive");
        let g = gcd(up, down);
        let (up, down) = (up / g, down / g);
        if up == 1 && down == 1 {
            return None;
        }
        let max_rate = up.max(down);
        let half_len = 10 * max_rate;
        let pre_pad = down - half_len % down;
        let mut taps = vec![0.0f32; pre_pad];
        taps.extend(
            firwin_kaiser(2 * half_len + 1, 1.0 / max_rate as f64, KAISER_BETA)
                .into_iter()
                .map(|v| v as f32 * up as f32),
        );
        let taps_per_phase = taps.len().div_ceil(up);
        taps.resize(taps_per_phase * up, 0.0);
        let mut phases = vec![0.0f32; taps.len()];
        for phase in 0..up {
            for i in 0..taps_per_phase {
                phases[phase * taps_per_phase + i] = taps[(taps_per_phase - 1 - i) * up + phase];
            }
        }
        Some(Self {
            up,
            down,
            phases,
            taps_per_phase,
            pre_remove: (half_len + pre_pad) / down,
        })
    }

    pub const fn ratio(&self) -> (usize, usize) {
        (self.up, self.down)
    }

    /// `ceil(n_in * up / down)`.
    pub const fn output_len(&self, n_in: usize) -> usize {
        (n_in * self.up).div_ceil(self.down)
    }

    /// Resample one channel; the result has exactly [`Self::output_len`] samples.
    pub fn process(&self, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.output_len(x.len())];
        if out.len() > PARALLEL_BLOCK {
            out.par_chunks_mut(PARALLEL_BLOCK)
                .enumerate()
                .for_each(|(block, slots)| self.fill(x, self.pre_remove + block * PARALLEL_BLOCK, slots));
        } else {
            self.fill(x, self.pre_remove, &mut out);
        }
        out
    }

    /// Output `k` accumulates `x[j] * h` over increasing `j` from zero, as `upfirdn._apply_impl` does.
    fn fill(&self, x: &[f32], first: usize, slots: &mut [f32]) {
        let hpp = self.taps_per_phase;
        let Some(last_x) = x.len().checked_sub(1) else {
            slots.fill(0.0);
            return;
        };
        for (offset, slot) in slots.iter_mut().enumerate() {
            let total = (first + offset) * self.down;
            let x_idx = total / self.up;
            let phase = total % self.up;
            let lo = (x_idx + 1).saturating_sub(hpp);
            let hi = x_idx.min(last_x);
            *slot = if lo <= hi {
                let base = phase * hpp + (lo + hpp - 1 - x_idx);
                let coeffs = &self.phases[base..base + (hi - lo + 1)];
                x[lo..=hi]
                    .iter()
                    .zip(coeffs)
                    .fold(0.0f32, |acc, (&s, &h)| acc + s * h)
            } else {
                0.0
            };
        }
    }
}

/// `resample_poly(x, up, down, axis=0, padtype="constant")` on one float32 channel.
pub fn resample_poly(x: &[f32], up: usize, down: usize) -> Vec<f32> {
    match PolyphaseResampler::new(up, down) {
        Some(resampler) => resampler.process(x),
        None => x.to_vec(),
    }
}

/// Resample between two integer rates, as `resample_poly(x, to // g, from // g)`.
pub fn resample_rates(x: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    let g = gcd(from_rate as usize, to_rate as usize);
    resample_poly(x, to_rate as usize / g, from_rate as usize / g)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_ratio_copies() {
        assert_eq!(resample_poly(&[1.0, 2.0], 3, 3), vec![1.0, 2.0]);
    }

    #[test]
    fn output_length_is_ceiling() {
        let r = PolyphaseResampler::new(147, 160).unwrap();
        assert_eq!(r.output_len(160), 147);
        assert_eq!(r.output_len(161), 148);
    }

    #[test]
    fn dc_is_preserved_in_the_interior() {
        let x = vec![0.5f32; 4000];
        let y = resample_poly(&x, 1, 3);
        assert!((y[600] - 0.5).abs() < 1e-4);
    }
}
