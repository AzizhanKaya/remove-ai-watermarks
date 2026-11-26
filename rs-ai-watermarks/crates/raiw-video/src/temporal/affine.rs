//! `cv::estimateAffinePartial2D(..., RANSAC)` (private adapter; raiw-cv has no estimator).
//!
//! RANSAC follows OpenCV 4.10: two-point similarity kernel, `cv::RNG` seeded with `(uint64)-1`,
//! squared-error inlier test, adaptive iteration bound at 0.99 confidence and 2000 iterations.
//! The Levenberg-Marquardt refinement is replaced by the exact linear least-squares fit it
//! converges to on this linear 4-parameter problem.

use raiw_cv::Point2f;

/// OpenCV's multiply-with-carry `cv::RNG`.
#[derive(Clone, Copy, Debug)]
struct CvRng(u64);

impl CvRng {
    const COEFF: u64 = 4_164_903_690;

    fn next(&mut self) -> u32 {
        self.0 = u64::from(self.0 as u32)
            .wrapping_mul(Self::COEFF)
            .wrapping_add(self.0 >> 32);
        self.0 as u32
    }

    fn uniform(&mut self, a: i32, b: i32) -> i32 {
        if a == b {
            a
        } else {
            (self.next() % (b - a) as u32) as i32 + a
        }
    }
}

type Model = [f64; 6];

fn similarity_kernel(from: [Point2f; 2], to: [Point2f; 2]) -> Model {
    let (x1, y1, x2, y2) = (
        f64::from(from[0].x),
        f64::from(from[0].y),
        f64::from(from[1].x),
        f64::from(from[1].y),
    );
    let (bx1, by1, bx2, by2) = (
        f64::from(to[0].x),
        f64::from(to[0].y),
        f64::from(to[1].x),
        f64::from(to[1].y),
    );
    let d = 1.0 / ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
    let s0 = d * ((bx1 - bx2) * (x1 - x2) + (by1 - by2) * (y1 - y2));
    let s1 = d * ((by1 - by2) * (x1 - x2) - (bx1 - bx2) * (y1 - y2));
    let s2 = d
        * ((by1 - by2) * (x1 * y2 - x2 * y1)
            - (bx1 * y2 - bx2 * y1) * (y1 - y2)
            - (bx1 * x2 - bx2 * x1) * (x1 - x2));
    let s3 = d
        * (-(bx1 - bx2) * (x1 * y2 - x2 * y1)
            - (by1 * x2 - by2 * x1) * (x1 - x2)
            - (by1 * y2 - by2 * y1) * (y1 - y2));
    [s0, -s1, s2, s1, s0, s3]
}

fn inliers(from: &[Point2f], to: &[Point2f], model: &Model, threshold: f64) -> Vec<bool> {
    let t = (threshold * threshold) as f32;
    from.iter()
        .zip(to)
        .map(|(f, p)| {
            let a = model[0] * f64::from(f.x) + model[1] * f64::from(f.y) + model[2] - f64::from(p.x);
            let b = model[3] * f64::from(f.x) + model[4] * f64::from(f.y) + model[5] - f64::from(p.y);
            ((a * a + b * b) as f32) <= t
        })
        .collect()
}

fn update_iterations(confidence: f64, outlier_ratio: f64, model_points: i32, max_iters: usize) -> usize {
    let p = confidence.clamp(0.0, 1.0);
    let ep = outlier_ratio.clamp(0.0, 1.0);
    let num = (1.0 - p).max(f64::MIN_POSITIVE);
    let denom = 1.0 - (1.0 - ep).powi(model_points);
    if denom < f64::MIN_POSITIVE {
        return 0;
    }
    let (num, denom) = (num.ln(), denom.ln());
    if denom >= 0.0 || -num >= max_iters as f64 * -denom {
        max_iters
    } else {
        (num / denom).round_ties_even() as usize
    }
}

fn least_squares(from: &[Point2f], to: &[Point2f], mask: &[bool]) -> Option<Model> {
    // Unknowns (a, b, tx, ty): X = a x - b y + tx, Y = b x + a y + ty.
    let mut ata = [[0.0f64; 4]; 4];
    let mut atb = [0.0f64; 4];
    for ((f, t), keep) in from.iter().zip(to).zip(mask) {
        if !keep {
            continue;
        }
        let (x, y) = (f64::from(f.x), f64::from(f.y));
        for (row, target) in [
            ([x, -y, 1.0, 0.0], f64::from(t.x)),
            ([y, x, 0.0, 1.0], f64::from(t.y)),
        ] {
            for i in 0..4 {
                for j in 0..4 {
                    ata[i][j] += row[i] * row[j];
                }
                atb[i] += row[i] * target;
            }
        }
    }
    let solution = solve4(ata, atb)?;
    let [a, b, tx, ty] = solution;
    Some([a, -b, tx, b, a, ty])
}

fn solve4(mut a: [[f64; 4]; 4], mut b: [f64; 4]) -> Option<[f64; 4]> {
    for col in 0..4 {
        let pivot = (col..4).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..4 {
            let factor = a[row][col] / a[col][col];
            let pivot_row = a[col];
            for (value, pivot) in a[row].iter_mut().zip(pivot_row).skip(col) {
                *value -= factor * pivot;
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = [0.0; 4];
    for row in (0..4).rev() {
        let mut sum = b[row];
        for k in row + 1..4 {
            sum -= a[row][k] * x[k];
        }
        x[row] = sum / a[row][row];
    }
    Some(x)
}

/// The estimated `[[a, -b, tx], [b, a, ty]]` model and the RANSAC inlier mask.
pub fn estimate_affine_partial_2d(
    from: &[Point2f],
    to: &[Point2f],
    threshold: f64,
) -> Option<([[f64; 3]; 2], Vec<bool>)> {
    const MODEL_POINTS: usize = 2;
    let count = from.len();
    if count != to.len() || count < MODEL_POINTS {
        return None;
    }
    let mut rng = CvRng(u64::MAX);
    let mut best: Option<(Model, Vec<bool>)> = None;
    let mut max_good = 0usize;
    let mut niters = 2000usize;
    if count == MODEL_POINTS {
        let model = similarity_kernel([from[0], from[1]], [to[0], to[1]]);
        best = Some((model, vec![true; count]));
    } else {
        let mut iter = 0;
        while iter < niters {
            let mut idx = [0usize; MODEL_POINTS];
            for i in 0..MODEL_POINTS {
                loop {
                    let candidate = rng.uniform(0, count as i32) as usize;
                    if !idx[..i].contains(&candidate) {
                        idx[i] = candidate;
                        break;
                    }
                }
            }
            let model = similarity_kernel([from[idx[0]], from[idx[1]]], [to[idx[0]], to[idx[1]]]);
            let mask = inliers(from, to, &model, threshold);
            let good = mask.iter().filter(|m| **m).count();
            if good > max_good.max(MODEL_POINTS - 1) {
                max_good = good;
                niters = update_iterations(
                    0.99,
                    (count - good) as f64 / count as f64,
                    MODEL_POINTS as i32,
                    niters,
                );
                best = Some((model, mask));
            }
            iter += 1;
        }
    }
    let (model, mask) = best?;
    let refined = least_squares(from, to, &mask).unwrap_or(model);
    Some((
        [
            [refined[0], refined[1], refined[2]],
            [refined[3], refined[4], refined[5]],
        ],
        mask,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_matches_opencv_sequence() {
        let mut rng = CvRng(u64::MAX);
        assert_eq!(rng.next(), 130_063_605);
    }

    #[test]
    fn recovers_a_pure_translation() {
        let from: Vec<Point2f> = (0..12)
            .map(|i| Point2f::new(i as f32 * 3.0, (i * i % 7) as f32 * 5.0))
            .collect();
        let to: Vec<Point2f> = from.iter().map(|p| Point2f::new(p.x + 6.5, p.y - 2.0)).collect();
        let (m, mask) = estimate_affine_partial_2d(&from, &to, 1.5).expect("model");
        assert!(mask.iter().all(|m| *m));
        assert!((m[0][2] - 6.5).abs() < 1e-9 && (m[1][2] + 2.0).abs() < 1e-9 && (m[0][0] - 1.0).abs() < 1e-9);
    }
}
