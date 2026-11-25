//! Bit-compatible `torch.Generator(device="cpu").manual_seed(s)` streams.
//!
//! SNAC's noise blocks draw `torch.randn` from a freshly seeded CPU generator on every
//! call, so reproducing its output needs torch's MT19937 plus the 16-lane Box-Muller
//! `normal_fill` used for contiguous float32 tensors of at least 16 elements.

use std::f64::consts::PI;

const STATE: usize = 624;
const SHIFT: usize = 397;

/// The MT19937 engine behind torch's CPU generator.
#[derive(Debug, Clone)]
pub struct Mt19937 {
    state: [u32; STATE],
    index: usize,
}

impl Mt19937 {
    pub fn new(seed: u64) -> Self {
        let mut state = [0u32; STATE];
        state[0] = seed as u32;
        for i in 1..STATE {
            let prev = state[i - 1];
            state[i] = 1_812_433_253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        Self { state, index: STATE }
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.index >= STATE {
            self.twist();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    fn twist(&mut self) {
        for i in 0..STATE {
            let y = (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % STATE] & 0x7fff_ffff);
            let mag = if y & 1 == 1 { 0x9908_b0df } else { 0 };
            self.state[i] = self.state[(i + SHIFT) % STATE] ^ (y >> 1) ^ mag;
        }
        self.index = 0;
    }

    /// `at::uniform_real_distribution<float>(0, 1)`: 24 random bits scaled to `[0, 1)`.
    pub fn uniform_f32(&mut self) -> f32 {
        (self.next_u32() & 0x00ff_ffff) as f32 * (1.0 / 16_777_216.0)
    }
}

/// `torch.randn(n, generator=torch.Generator().manual_seed(seed))` for `n >= 16`.
///
/// Returns `None` for shorter draws, which torch serves from a different scalar path.
pub fn torch_randn(n: usize, seed: u64) -> Option<Vec<f32>> {
    if n < 16 {
        return None;
    }
    let mut engine = Mt19937::new(seed);
    let mut data: Vec<f32> = (0..n).map(|_| engine.uniform_f32()).collect();
    for start in (0..n - 15).step_by(16) {
        normal_fill_16(&mut data[start..start + 16]);
    }
    if n % 16 != 0 {
        let tail = &mut data[n - 16..];
        for value in tail.iter_mut() {
            *value = engine.uniform_f32();
        }
        normal_fill_16(tail);
    }
    Some(data)
}

fn normal_fill_16(data: &mut [f32]) {
    let two_pi = (2.0 * PI) as f32;
    for j in 0..8 {
        let u1 = 1.0 - data[j];
        let u2 = data[j + 8];
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = two_pi * u2;
        data[j] = radius * theta.cos();
        data[j + 8] = radius * theta.sin();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_reference_stream() {
        let mut engine = Mt19937::new(5489);
        assert_eq!(engine.next_u32(), 3_499_211_612);
        let mut zero = Mt19937::new(0);
        assert_eq!(zero.next_u32(), 2_357_136_044);
    }

    #[test]
    fn short_draws_are_unsupported() {
        assert!(torch_randn(15, 0).is_none());
        assert_eq!(torch_randn(33, 0).map(|v| v.len()), Some(33));
    }
}
