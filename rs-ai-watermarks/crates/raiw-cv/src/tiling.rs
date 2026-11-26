//! Port of `_internal/tiling.py`: tile geometry, feather weights and the blend loop.

use crate::error::{Result, invalid};
use crate::mat::Mat;
use crate::resize::resize;
use crate::types::Interpolation;

/// Strictly positive floor of accumulated blend weights.
const WEIGHT_EPS: f32 = 1e-3;

/// A tile crop box: top-left corner plus size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tile {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

/// `_axis_positions`: tile starts along one axis, the last one flush with the far edge.
pub fn axis_positions(length: usize, tile: usize, overlap: isize) -> Result<Vec<usize>> {
    if tile == 0 {
        return Err(invalid(
            "axis_positions",
            format!("tile must be positive, got {tile}"),
        ));
    }
    if length <= tile {
        return Ok(vec![0]);
    }
    let overlap = overlap.max(0).min(tile as isize - 1) as usize;
    let step = tile - overlap;
    let last = length - tile;
    let mut positions: Vec<usize> = (0..=last).step_by(step).collect();
    if positions.last() != Some(&last) {
        positions.push(last);
    }
    Ok(positions)
}

/// `plan_tiles`: uniform tiles in row-major order.
pub fn plan_tiles(width: usize, height: usize, tile_size: usize, overlap: isize) -> Result<Vec<Tile>> {
    let xs = axis_positions(width, tile_size, overlap)?;
    let ys = axis_positions(height, tile_size, overlap)?;
    let (tw, th) = (tile_size.min(width), tile_size.min(height));
    Ok(ys
        .iter()
        .flat_map(|&y| {
            xs.iter().map(move |&x| Tile {
                x,
                y,
                width: tw,
                height: th,
            })
        })
        .collect())
}

fn ramp(n: usize, overlap: isize) -> Vec<f32> {
    let mut w = vec![1.0f32; n];
    if overlap > 0 && n > 1 {
        let len = (overlap as usize).min((n / 2).max(1));
        let taper: Vec<f32> = (0..len).map(|i| (i as f32 + 1.0) / (len as f32 + 1.0)).collect();
        w[..len].copy_from_slice(&taper);
        for (i, t) in taper.iter().rev().enumerate() {
            w[n - len + i] = *t;
        }
    }
    w
}

/// `feather_weights`: the separable linear taper, floored at `1e-3`.
pub fn feather_weights(width: usize, height: usize, overlap: isize) -> Mat<f32> {
    let (rx, ry) = (ramp(width, overlap), ramp(height, overlap));
    Mat::from_fn(width, height, 1, |x, y, _| (ry[y] * rx[x]).max(WEIGHT_EPS))
}

/// `run_tiled`: regenerate every tile of an RGB image and feather-blend the results.
///
/// A tile returned at another size is resized back with Lanczos-4 (Python uses PIL's Lanczos).
pub fn run_tiled(
    mut generate_tile: impl FnMut(&Mat<u8>) -> Result<Mat<u8>>,
    image: &Mat<u8>,
    tile_size: usize,
    overlap: isize,
    mut progress: Option<&mut dyn FnMut(&str)>,
) -> Result<Mat<u8>> {
    if image.channels() != 3 {
        return Err(invalid("run_tiled", "RGB image required"));
    }
    let (w, h) = image.size();
    let tiles = plan_tiles(w, h, tile_size, overlap)?;
    let mut accum = vec![0.0f32; w * h * 3];
    let mut weight_sum = vec![0.0f32; w * h];
    let weights = feather_weights(tiles[0].width, tiles[0].height, overlap);
    let total = tiles.len();
    for (index, t) in tiles.iter().enumerate() {
        if let Some(report) = progress.as_deref_mut() {
            report(&format!(
                "Tiled diffusion: tile {}/{total} at ({},{}) {}x{}...",
                index + 1,
                t.x,
                t.y,
                t.width,
                t.height
            ));
        }
        let crop = image.crop(crate::types::Rect::new(t.x, t.y, t.width, t.height))?;
        let mut result = generate_tile(&crop)?;
        if result.size() != (t.width, t.height) {
            result = resize(&result, t.width, t.height, Interpolation::Lanczos4)?;
        }
        if result.channels() != 3 {
            return Err(invalid("run_tiled", "tile generator must return RGB"));
        }
        for y in 0..t.height {
            for x in 0..t.width {
                let wv = weights.at(x, y, 0);
                let (gx, gy) = (t.x + x, t.y + y);
                for c in 0..3 {
                    accum[(gy * w + gx) * 3 + c] += f32::from(result.at(x, y, c)) * wv;
                }
                weight_sum[gy * w + gx] += wv;
            }
        }
    }
    let data = accum
        .iter()
        .enumerate()
        .map(|(i, &a)| (a / weight_sum[i / 3].max(WEIGHT_EPS)).clamp(0.0, 255.0) as u8)
        .collect();
    Mat::from_vec(w, h, 3, data)
}
