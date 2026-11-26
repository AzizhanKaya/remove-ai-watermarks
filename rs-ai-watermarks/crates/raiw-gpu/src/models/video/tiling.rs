//! diffusers-compatible spatial tiling: overlapping tiles, linear blends, cropped assembly.

use crate::error::Result;

/// A host CHW plane.
#[derive(Debug, Clone, PartialEq)]
pub struct Plane {
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub data: Vec<f32>,
}

impl Plane {
    pub fn new(channels: usize, height: usize, width: usize, data: Vec<f32>) -> Self {
        Self {
            channels,
            height,
            width,
            data,
        }
    }

    fn at(&self, c: usize, y: usize, x: usize) -> f32 {
        self.data[(c * self.height + y) * self.width + x]
    }

    fn crop(&self, y0: usize, y1: usize, x0: usize, x1: usize) -> Self {
        let (h, w) = (y1 - y0, x1 - x0);
        let mut data = Vec::with_capacity(self.channels * h * w);
        for c in 0..self.channels {
            for y in y0..y1 {
                let row = (c * self.height + y) * self.width;
                data.extend_from_slice(&self.data[row + x0..row + x1]);
            }
        }
        Self::new(self.channels, h, w, data)
    }
}

/// Tile geometry in input units, plus how outputs blend and crop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tiling {
    pub tile: usize,
    pub stride: usize,
    pub blend: usize,
    pub limit: usize,
}

/// `AutoencoderKL.tiled_encode/_decode`: run `process` on overlapping tiles, blend each tile into
/// its (already blended) upper and left neighbours in place, crop and concatenate.
pub fn run_tiled(
    input: &Plane,
    tiling: Tiling,
    mut process: impl FnMut(&Plane) -> Result<Plane>,
) -> Result<Plane> {
    let mut rows: Vec<Vec<Plane>> = Vec::new();
    for i in (0..input.height).step_by(tiling.stride) {
        let mut row = Vec::new();
        for j in (0..input.width).step_by(tiling.stride) {
            let tile = input.crop(
                i,
                (i + tiling.tile).min(input.height),
                j,
                (j + tiling.tile).min(input.width),
            );
            row.push(process(&tile)?);
        }
        rows.push(row);
    }
    for i in 0..rows.len() {
        for j in 0..rows[i].len() {
            let mut tile = rows[i][j].clone();
            if i > 0 {
                blend_v(&rows[i - 1][j], &mut tile, tiling.blend);
            }
            if j > 0 {
                blend_h(&rows[i][j - 1], &mut tile, tiling.blend);
            }
            rows[i][j] = tile;
        }
    }
    let cropped: Vec<Vec<Plane>> = rows
        .iter()
        .map(|row| {
            row.iter()
                .map(|t| t.crop(0, tiling.limit.min(t.height), 0, tiling.limit.min(t.width)))
                .collect()
        })
        .collect();
    let channels = cropped[0][0].channels;
    let height: usize = cropped.iter().map(|row| row[0].height).sum();
    let width: usize = cropped[0].iter().map(|t| t.width).sum();
    let mut data = vec![0.0f32; channels * height * width];
    let mut y0 = 0;
    for row in &cropped {
        let mut x0 = 0;
        for tile in row {
            for c in 0..channels {
                for y in 0..tile.height {
                    let dst = (c * height + y0 + y) * width + x0;
                    let src = (c * tile.height + y) * tile.width;
                    data[dst..dst + tile.width].copy_from_slice(&tile.data[src..src + tile.width]);
                }
            }
            x0 += tile.width;
        }
        y0 += row[0].height;
    }
    Ok(Plane::new(channels, height, width, data))
}

fn blend_v(a: &Plane, b: &mut Plane, extent: usize) {
    let extent = extent.min(a.height).min(b.height);
    for y in 0..extent {
        let t = (y as f64 / extent as f64) as f32;
        let keep = (1.0 - y as f64 / extent as f64) as f32;
        for c in 0..b.channels {
            for x in 0..b.width {
                let idx = (c * b.height + y) * b.width + x;
                b.data[idx] = a.at(c, a.height - extent + y, x) * keep + b.data[idx] * t;
            }
        }
    }
}

fn blend_h(a: &Plane, b: &mut Plane, extent: usize) {
    let extent = extent.min(a.width).min(b.width);
    for x in 0..extent {
        let t = (x as f64 / extent as f64) as f32;
        let keep = (1.0 - x as f64 / extent as f64) as f32;
        for c in 0..b.channels {
            for y in 0..b.height {
                let idx = (c * b.height + y) * b.width + x;
                b.data[idx] = a.at(c, y, a.width - extent + x) * keep + b.data[idx] * t;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_tiling_reassembles_the_input() {
        let input = Plane::new(1, 6, 6, (0..36).map(|v| v as f32).collect());
        let tiling = Tiling {
            tile: 4,
            stride: 3,
            blend: 1,
            limit: 3,
        };
        let out = run_tiled(&input, tiling, |tile| Ok(tile.clone())).unwrap();
        assert_eq!((out.height, out.width), (6, 6));
        assert_eq!(out.data, input.data);
    }
}
