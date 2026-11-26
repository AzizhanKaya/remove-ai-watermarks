//! Context-ring motion fitting shared by fill stabilization and background recovery.

use raiw_cv::{Interpolation, Mat};

use super::flow::{Maps, backward_warp, displacement_to_maps};
use super::pixel::BoolPlane;
use crate::error::Result;
use crate::py::median_f32;

/// `np.linalg.norm(flow_to + backward_warp(flow_from, maps(flow_to)), axis=2)`.
pub fn forward_backward_error(flow_to: &Mat<f32>, flow_from: &Mat<f32>) -> Result<Vec<f32>> {
    let maps = displacement_to_maps(flow_to);
    let back = backward_warp(flow_from, &maps, Interpolation::Linear)?;
    Ok(flow_to
        .data()
        .chunks_exact(2)
        .zip(back.data().chunks_exact(2))
        .map(|(a, b)| {
            let (u, v) = (a[0] + b[0], a[1] + b[1]);
            (u * u + v * v).sqrt()
        })
        .collect())
}

/// Least-squares `u = c0*x + c1*y + c2` over the given samples (`np.linalg.lstsq`).
fn fit_plane(xs: &[f32], ys: &[f32], values: &[f32]) -> [f32; 3] {
    let mut ata = [[0.0f64; 3]; 3];
    let mut atb = [0.0f64; 3];
    for ((x, y), v) in xs.iter().zip(ys).zip(values) {
        let row = [f64::from(*x), f64::from(*y), 1.0];
        for i in 0..3 {
            for j in 0..3 {
                ata[i][j] += row[i] * row[j];
            }
            atb[i] += row[i] * f64::from(*v);
        }
    }
    solve3(ata, atb).map_or([0.0, 0.0, 0.0], |c| [c[0] as f32, c[1] as f32, c[2] as f32])
}

fn solve3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let pivot = (col..3).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..3 {
            let factor = a[row][col] / a[col][col];
            let pivot_row = a[col];
            for (value, pivot) in a[row].iter_mut().zip(pivot_row).skip(col) {
                *value -= factor * pivot;
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = [0.0; 3];
    for row in (0..3).rev() {
        let mut sum = b[row];
        for k in row + 1..3 {
            sum -= a[row][k] * x[k];
        }
        x[row] = sum / a[row][row];
    }
    Some(x)
}

/// Fit a robust affine motion on `context` pixels and propagate it over the crop, keeping the
/// direct flow where `keep_direct` holds. `None` when fewer than `min_inliers` samples survive.
pub fn propagate_context_motion(
    flow: &Mat<f32>,
    context: &BoolPlane,
    keep_direct: &BoolPlane,
    min_inliers: usize,
) -> Option<Mat<f32>> {
    let width = flow.width();
    let data = flow.data();
    let (mut cx, mut cy, mut u, mut v) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (index, set) in context.data.iter().enumerate() {
        if *set {
            cx.push((index % width) as f32);
            cy.push((index / width) as f32);
            u.push(data[index * 2]);
            v.push(data[index * 2 + 1]);
        }
    }
    let med_u = median_f32(&mut u.clone());
    let med_v = median_f32(&mut v.clone());
    let (mut ix, mut iy, mut iu, mut iv) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..u.len() {
        if (u[i] - med_u).abs() <= 3.0 && (v[i] - med_v).abs() <= 3.0 {
            ix.push(cx[i]);
            iy.push(cy[i]);
            iu.push(u[i]);
            iv.push(v[i]);
        }
    }
    if ix.len() < min_inliers {
        return None;
    }
    let cu = fit_plane(&ix, &iy, &iu);
    let cv = fit_plane(&ix, &iy, &iv);
    let mut out = flow.clone();
    let out_data = out.data_mut();
    for (index, keep) in keep_direct.data.iter().enumerate() {
        if *keep {
            continue;
        }
        let (x, y) = ((index % width) as f32, (index / width) as f32);
        out_data[index * 2] = cu[0] * x + cu[1] * y + cu[2];
        out_data[index * 2 + 1] = cv[0] * x + cv[1] * y + cv[2];
    }
    Some(out)
}

/// `maps` and the in-bounds plane of a backward map over a `width` x `height` crop.
pub fn maps_in_bounds(maps: &Maps) -> BoolPlane {
    let (w, h) = maps.0.size();
    let data = maps
        .0
        .data()
        .iter()
        .zip(maps.1.data())
        .map(|(x, y)| *x >= 0.0 && *x <= (w - 1) as f32 && *y >= 0.0 && *y <= (h - 1) as f32)
        .collect();
    BoolPlane {
        width: w,
        height: h,
        data,
    }
}
