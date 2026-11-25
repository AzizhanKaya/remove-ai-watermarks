//! `cv::connectedComponents` / `connectedComponentsWithStats` with OpenCV's label order.
//!
//! OpenCV's default 8-way labeling (Spaghetti/BBDT) scans 2x2 blocks, so labels follow the first
//! 2x2 block of each component in raster order; 4-way SAUF follows the first pixel.

use crate::element::Element;
use crate::error::{Result, invalid};
use crate::mat::{Mat, MatView};
use crate::types::Point2d;

/// Pixel connectivity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Connectivity {
    Four,
    Eight,
}

/// Per-label statistics (`CC_STAT_LEFT .. CC_STAT_AREA`, centroids).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ComponentStats {
    pub left: i32,
    pub top: i32,
    pub width: i32,
    pub height: i32,
    pub area: i32,
    pub centroid: Point2d,
}

/// Labels plus per-label stats; label 0 is the background.
#[derive(Clone, Debug)]
pub struct Components {
    pub count: usize,
    pub labels: Mat<i32>,
    pub stats: Vec<ComponentStats>,
}

fn find(parent: &mut [u32], mut i: u32) -> u32 {
    while parent[i as usize] != i {
        let p = parent[i as usize];
        parent[i as usize] = parent[p as usize];
        i = p;
    }
    i
}

fn union(parent: &mut [u32], a: u32, b: u32) {
    let (ra, rb) = (find(parent, a), find(parent, b));
    if ra != rb {
        let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
        parent[hi as usize] = lo;
    }
}

/// `cv::connectedComponents(src, connectivity)`: returns `(count, labels)`.
pub fn connected_components<'a, T: Element>(src: impl Into<MatView<'a, T>>, connectivity: Connectivity) -> Result<(usize, Mat<i32>)> {
    let src = src.into();
    let (count, labels) = label(&src, connectivity)?;
    Ok((count, labels))
}

/// `cv::connectedComponentsWithStats(src, connectivity)`.
pub fn connected_components_with_stats<'a, T: Element>(
    src: impl Into<MatView<'a, T>>,
    connectivity: Connectivity,
) -> Result<Components> {
    let src = src.into();
    let (count, labels) = label(&src, connectivity)?;
    let mut acc = vec![(i32::MAX, i32::MAX, i32::MIN, i32::MIN, 0i32, 0f64, 0f64); count];
    for y in 0..labels.height() {
        for (x, &l) in labels.row(y).iter().enumerate() {
            let a = &mut acc[l as usize];
            let (xi, yi) = (x as i32, y as i32);
            a.0 = a.0.min(xi);
            a.1 = a.1.min(yi);
            a.2 = a.2.max(xi);
            a.3 = a.3.max(yi);
            a.4 += 1;
            a.5 += x as f64;
            a.6 += y as f64;
        }
    }
    let stats = acc
        .into_iter()
        .map(|(l, t, r, b, area, sx, sy)| {
            if area == 0 {
                ComponentStats { left: -1, top: -1, width: 0, height: 0, area: 0, centroid: Point2d::new(f64::NAN, f64::NAN) }
            } else {
                ComponentStats {
                    left: l,
                    top: t,
                    width: r - l + 1,
                    height: b - t + 1,
                    area,
                    centroid: Point2d::new(sx / f64::from(area), sy / f64::from(area)),
                }
            }
        })
        .collect();
    Ok(Components { count, labels, stats })
}

fn label<T: Element>(src: &MatView<'_, T>, connectivity: Connectivity) -> Result<(usize, Mat<i32>)> {
    if src.channels() != 1 {
        return Err(invalid("connected_components", "single-channel input required"));
    }
    let (w, h) = (src.width(), src.height());
    let n = w * h;
    let fg: Vec<bool> = (0..h).flat_map(|y| src.row(y).iter().map(|v| v.to_f64() != 0.0)).collect();
    let mut parent: Vec<u32> = (0..n as u32).collect();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if !fg[i] {
                continue;
            }
            if x > 0 && fg[i - 1] {
                union(&mut parent, i as u32, (i - 1) as u32);
            }
            if y > 0 {
                let up = i - w;
                if fg[up] {
                    union(&mut parent, i as u32, up as u32);
                }
                if connectivity == Connectivity::Eight {
                    if x > 0 && fg[up - 1] {
                        union(&mut parent, i as u32, (up - 1) as u32);
                    }
                    if x + 1 < w && fg[up + 1] {
                        union(&mut parent, i as u32, (up + 1) as u32);
                    }
                }
            }
        }
    }
    let scan: Vec<usize> = match connectivity {
        Connectivity::Four => (0..n).collect(),
        Connectivity::Eight => {
            let mut order = Vec::with_capacity(n);
            for by in (0..h).step_by(2) {
                for bx in (0..w).step_by(2) {
                    for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                        let (y, x) = (by + dy, bx + dx);
                        if y < h && x < w {
                            order.push(y * w + x);
                        }
                    }
                }
            }
            order
        }
    };
    let mut final_label = vec![0i32; n];
    let mut next = 1i32;
    for &i in &scan {
        if fg[i] {
            let r = find(&mut parent, i as u32) as usize;
            if final_label[r] == 0 {
                final_label[r] = next;
                next += 1;
            }
        }
    }
    let data: Vec<i32> =
        (0..n).map(|i| if fg[i] { final_label[find(&mut parent, i as u32) as usize] } else { 0 }).collect();
    Ok((next as usize, Mat::from_vec(w, h, 1, data)?))
}
