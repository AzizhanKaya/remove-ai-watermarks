//! `WanRotaryPosEmbed`: 3-D rotary embedding over (frame, row, column) patch positions.
//!
//! The head dimension splits into a time band (`d - 4 * (d / 6)` channels) and two spatial
//! bands (`2 * (d / 6)` each); every band is `get_1d_rotary_pos_embed(theta=10000)` computed
//! in float64 and cast to float32, and pairs `(2i, 2i + 1)` rotate together.

use burn::tensor::{Device, Tensor, TensorData, s};

const THETA: f64 = 10_000.0;

/// Per-token cos/sin of every rotary pair, uploaded once per patch grid.
#[derive(Debug, Clone)]
pub struct Rope {
    cos: Tensor<4>,
    sin: Tensor<4>,
}

impl Rope {
    /// Tables for a `(frames, rows, cols)` patch grid in token order.
    pub fn new(head_dim: usize, (frames, rows, cols): (usize, usize, usize), device: &Device) -> Self {
        let (cos, sin) = tables(head_dim, (frames, rows, cols));
        let shape = [frames * rows * cols, 1, head_dim / 2, 1];
        Self {
            cos: Tensor::from_data(TensorData::new(cos, shape), device),
            sin: Tensor::from_data(TensorData::new(sin, shape), device),
        }
    }

    /// Rotate `[tokens, heads, head_dim]` queries or keys.
    pub fn apply(&self, x: Tensor<3>) -> Tensor<3> {
        let [tokens, heads, dim] = x.dims();
        let pairs = x.reshape([tokens, heads, dim / 2, 2]);
        let x1 = pairs.clone().slice(s![.., .., .., 0..1]);
        let x2 = pairs.slice(s![.., .., .., 1..2]);
        let even = x1.clone() * self.cos.clone() - x2.clone() * self.sin.clone();
        let odd = x1 * self.sin.clone() + x2 * self.cos.clone();
        Tensor::cat(vec![even, odd], 3).reshape([tokens, heads, dim])
    }
}

/// Host `[tokens, head_dim / 2]` cos and sin tables.
pub fn tables(head_dim: usize, (frames, rows, cols): (usize, usize, usize)) -> (Vec<f32>, Vec<f32>) {
    let spatial = 2 * (head_dim / 6);
    let bands = [head_dim - 2 * spatial, spatial, spatial];
    let freqs: Vec<Vec<f64>> = bands
        .iter()
        .map(|&dim| {
            (0..dim / 2)
                .map(|i| 1.0 / THETA.powf((2 * i) as f64 / dim as f64))
                .collect()
        })
        .collect();
    let pairs = head_dim / 2;
    let tokens = frames * rows * cols;
    let (mut cos, mut sin) = (
        Vec::with_capacity(tokens * pairs),
        Vec::with_capacity(tokens * pairs),
    );
    for f in 0..frames {
        for r in 0..rows {
            for c in 0..cols {
                for (band, position) in freqs.iter().zip([f, r, c]) {
                    for &freq in band {
                        let angle = position as f64 * freq;
                        cos.push(angle.cos() as f32);
                        sin.push(angle.sin() as f32);
                    }
                }
            }
        }
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bands_split_like_diffusers() {
        let (cos, sin) = tables(128, (2, 1, 3));
        assert_eq!(cos.len(), 6 * 64);
        // token (f=1, r=0, c=2): time band angle 1, column band angle 2 at pair 43
        assert_eq!(cos[3 * 64], 1f64.cos() as f32);
        assert_eq!(sin[5 * 64 + 43], 2f64.sin() as f32);
        assert_eq!(sin[5 * 64 + 22], 0.0);
    }
}
