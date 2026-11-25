//! Rasterization from `drawing.cpp` (`fillConvexPoly`, filled `ellipse`, `ellipse2Poly`) and the
//! geometry helpers `boxPoints` / `boundingRect`.

use std::sync::OnceLock;

use crate::element::{Element, cv_round};
use crate::error::{Result, invalid};
use crate::mat::{Mat, MatView};
use crate::types::{Point, Point2f, Rect, RotatedRect, Scalar};

const XY_SHIFT: i64 = 16;
const XY_ONE: i64 = 1 << XY_SHIFT;

/// `cv::LineTypes` for non-antialiased drawing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum LineType {
    Four,
    #[default]
    Eight,
}

/// OpenCV's `SinTable`: `sin(i deg)` rounded to 7 decimals, `i in 0..=450`.
fn sin_table() -> &'static [f32; 451] {
    static TABLE: OnceLock<[f32; 451]> = OnceLock::new();
    TABLE.get_or_init(|| {
        std::array::from_fn(|i| format!("{:.7}", (i as f64).to_radians().sin()).parse::<f32>().unwrap_or(0.0))
    })
}

/// `cv::ellipse2Poly` (double precision): arc vertices every `delta` degrees.
pub fn ellipse2poly_f64(center: (f64, f64), axes: (f64, f64), angle: i32, arc_start: i32, arc_end: i32, delta: i32) -> Vec<(f64, f64)> {
    let tab = sin_table();
    let mut angle = angle;
    while angle < 0 {
        angle += 360;
    }
    while angle > 360 {
        angle -= 360;
    }
    let (mut s, mut e) = if arc_start > arc_end { (arc_end, arc_start) } else { (arc_start, arc_end) };
    while s < 0 {
        s += 360;
        e += 360;
    }
    while e > 360 {
        e -= 360;
        s -= 360;
    }
    if e - s > 360 {
        s = 0;
        e = 360;
    }
    let a = if angle < 0 { angle + 360 } else { angle } as usize;
    let (alpha, beta) = (f64::from(tab[450 - a]), f64::from(tab[a]));
    let mut pts = Vec::new();
    let mut i = s;
    while i < e + delta {
        let mut ang = i.min(e);
        if ang < 0 {
            ang += 360;
        }
        let x = axes.0 * f64::from(tab[(450 - ang) as usize]);
        let y = axes.1 * f64::from(tab[ang as usize]);
        pts.push((center.0 + x * alpha - y * beta, center.1 + x * beta + y * alpha));
        i += delta;
    }
    if pts.len() == 1 {
        pts = vec![center; 2];
    }
    pts
}

/// `cv::ellipse2Poly` with integer output (consecutive duplicates removed).
pub fn ellipse2poly(center: Point, axes: (i32, i32), angle: i32, arc_start: i32, arc_end: i32, delta: i32) -> Vec<Point> {
    let raw = ellipse2poly_f64(
        (f64::from(center.x), f64::from(center.y)),
        (f64::from(axes.0), f64::from(axes.1)),
        angle,
        arc_start,
        arc_end,
        delta,
    );
    let mut pts: Vec<Point> = Vec::new();
    for (x, y) in raw {
        let p = Point::new(cv_round(x), cv_round(y));
        if pts.last() != Some(&p) {
            pts.push(p);
        }
    }
    if pts.len() == 1 {
        pts = vec![center; 2];
    }
    pts
}

/// `clipLine` on 64-bit coordinates; returns whether any part is visible.
fn clip_line(w: i64, h: i64, p1: &mut (i64, i64), p2: &mut (i64, i64)) -> bool {
    if w <= 0 || h <= 0 {
        return false;
    }
    let (right, bottom) = (w - 1, h - 1);
    let code = |p: (i64, i64)| i32::from(p.0 < 0) + i32::from(p.0 > right) * 2 + i32::from(p.1 < 0) * 4 + i32::from(p.1 > bottom) * 8;
    let (mut c1, mut c2) = (code(*p1), code(*p2));
    if (c1 & c2) == 0 && (c1 | c2) != 0 {
        if c1 & 12 != 0 {
            let a = if c1 < 8 { 0 } else { bottom };
            p1.0 += ((a - p1.1) as f64 * (p2.0 - p1.0) as f64 / (p2.1 - p1.1) as f64) as i64;
            p1.1 = a;
            c1 = i32::from(p1.0 < 0) + i32::from(p1.0 > right) * 2;
        }
        if c2 & 12 != 0 {
            let a = if c2 < 8 { 0 } else { bottom };
            p2.0 += ((a - p2.1) as f64 * (p2.0 - p1.0) as f64 / (p2.1 - p1.1) as f64) as i64;
            p2.1 = a;
            c2 = i32::from(p2.0 < 0) + i32::from(p2.0 > right) * 2;
        }
        if (c1 & c2) == 0 && (c1 | c2) != 0 {
            if c1 != 0 {
                let a = if c1 == 1 { 0 } else { right };
                p1.1 += ((a - p1.0) as f64 * (p2.1 - p1.1) as f64 / (p2.0 - p1.0) as f64) as i64;
                p1.0 = a;
                c1 = 0;
            }
            if c2 != 0 {
                let a = if c2 == 1 { 0 } else { right };
                p2.1 += ((a - p2.0) as f64 * (p2.1 - p1.1) as f64 / (p2.0 - p1.0) as f64) as i64;
                p2.0 = a;
                c2 = 0;
            }
        }
    }
    (c1 | c2) == 0
}

struct Canvas<'m, T> {
    img: &'m mut Mat<T>,
    color: Vec<T>,
}

impl<T: Element> Canvas<'_, T> {
    fn put(&mut self, x: i64, y: i64) {
        if (0..self.img.width() as i64).contains(&x) && (0..self.img.height() as i64).contains(&y) {
            self.img.pixel_mut(x as usize, y as usize).copy_from_slice(&self.color);
        }
    }

    fn hline(&mut self, y: usize, x1: usize, x2: usize) {
        for x in x1..=x2 {
            self.img.pixel_mut(x, y).copy_from_slice(&self.color);
        }
    }

    /// `Line` via `LineIterator` (integer endpoints, left to right).
    fn line(&mut self, a: Point, b: Point, line_type: LineType) {
        let (w, h) = (self.img.width() as i64, self.img.height() as i64);
        let (mut p1, mut p2) = ((i64::from(a.x), i64::from(a.y)), (i64::from(b.x), i64::from(b.y)));
        let outside = |p: (i64, i64)| p.0 < 0 || p.0 >= w || p.1 < 0 || p.1 >= h;
        if (outside(p1) || outside(p2)) && !clip_line(w, h, &mut p1, &mut p2) {
            return;
        }
        let (mut dx, mut dy) = (p2.0 - p1.0, p2.1 - p1.1);
        if dx < 0 {
            dx = -dx;
            dy = -dy;
            p1 = p2;
        }
        let (mut sx, mut sy) = (1i64, 1i64);
        if dy < 0 {
            dy = -dy;
            sy = -1;
        }
        let vert = dy > dx;
        if vert {
            std::mem::swap(&mut dx, &mut dy);
            std::mem::swap(&mut sx, &mut sy);
        }
        let (mut err, plus_delta, minus_delta, count, mut minus_shift, mut plus_shift, mut minus_step, mut plus_step);
        match line_type {
            LineType::Eight => {
                err = dx - 2 * dy;
                plus_delta = 2 * dx;
                minus_delta = -2 * dy;
                minus_shift = sx;
                plus_shift = 0;
                minus_step = 0;
                plus_step = sy;
                count = dx + 1;
            }
            LineType::Four => {
                err = 0;
                plus_delta = 2 * dx + 2 * dy;
                minus_delta = -2 * dy;
                minus_shift = sx;
                plus_shift = -sx;
                minus_step = 0;
                plus_step = sy;
                count = dx + dy + 1;
            }
        }
        if vert {
            std::mem::swap(&mut plus_step, &mut plus_shift);
            std::mem::swap(&mut minus_step, &mut minus_shift);
        }
        let (mut x, mut y) = p1;
        for _ in 0..count {
            self.put(x, y);
            let mask = if err < 0 { -1 } else { 0 };
            err += minus_delta + (plus_delta & mask);
            x += minus_shift + (plus_shift & mask);
            y += minus_step + (plus_step & mask);
        }
    }

    /// `Line2`: a line between `XY_SHIFT` fixed-point endpoints.
    fn line2(&mut self, a: (i64, i64), b: (i64, i64)) {
        let (w, h) = (self.img.width() as i64, self.img.height() as i64);
        let (mut p1, mut p2) = (a, b);
        if !clip_line(w << XY_SHIFT, h << XY_SHIFT, &mut p1, &mut p2) {
            return;
        }
        let (mut dx, mut dy) = (p2.0 - p1.0, p2.1 - p1.1);
        let j = if dx < 0 { -1 } else { 0 };
        let ax = (dx ^ j) - j;
        let i = if dy < 0 { -1 } else { 0 };
        let ay = (dy ^ i) - i;
        let (x_step, y_step, ecount);
        if ax > ay {
            dy = (dy ^ j) - j;
            if j != 0 {
                std::mem::swap(&mut p1, &mut p2);
            }
            x_step = XY_ONE;
            y_step = dy * XY_ONE / (ax | 1);
            ecount = (p2.0 - p1.0) >> XY_SHIFT;
        } else {
            dx = (dx ^ i) - i;
            if i != 0 {
                std::mem::swap(&mut p1, &mut p2);
            }
            x_step = dx * XY_ONE / (ay | 1);
            y_step = XY_ONE;
            ecount = (p2.1 - p1.1) >> XY_SHIFT;
        }
        p1.0 += XY_ONE >> 1;
        p1.1 += XY_ONE >> 1;
        self.put((p2.0 + (XY_ONE >> 1)) >> XY_SHIFT, (p2.1 + (XY_ONE >> 1)) >> XY_SHIFT);
        let mut n = ecount;
        if ax > ay {
            p1.0 >>= XY_SHIFT;
            while n >= 0 {
                self.put(p1.0, p1.1 >> XY_SHIFT);
                p1.0 += 1;
                p1.1 += y_step;
                n -= 1;
            }
        } else {
            p1.1 >>= XY_SHIFT;
            while n >= 0 {
                self.put(p1.0 >> XY_SHIFT, p1.1);
                p1.0 += x_step;
                p1.1 += 1;
                n -= 1;
            }
        }
    }

    /// `FillConvexPoly` for 4/8-connected (non-antialiased) edges.
    fn fill_convex(&mut self, v: &[(i64, i64)], line_type: LineType, shift: i64) {
        let npts = v.len();
        if npts == 0 {
            return;
        }
        let delta = (1i64 << shift) >> 1;
        let (w, h) = (self.img.width() as i64, self.img.height() as i64);
        let mut p0 = v[npts - 1];
        p0 = (p0.0 << (XY_SHIFT - shift), p0.1 << (XY_SHIFT - shift));
        let (mut xmin, mut xmax, mut ymin, mut ymax) = (v[0].0, v[0].0, v[0].1, v[0].1);
        let mut imin = 0;
        for (i, &q) in v.iter().enumerate() {
            if q.1 < ymin {
                ymin = q.1;
                imin = i;
            }
            ymax = ymax.max(q.1);
            xmax = xmax.max(q.0);
            xmin = xmin.min(q.0);
            let p = (q.0 << (XY_SHIFT - shift), q.1 << (XY_SHIFT - shift));
            if shift == 0 {
                let a = Point::new((p0.0 >> XY_SHIFT) as i32, (p0.1 >> XY_SHIFT) as i32);
                let b = Point::new((p.0 >> XY_SHIFT) as i32, (p.1 >> XY_SHIFT) as i32);
                self.line(a, b, line_type);
            } else {
                self.line2(p0, p);
            }
            p0 = p;
        }
        let (xmin, xmax, ymin, ymax) = ((xmin + delta) >> shift, (xmax + delta) >> shift, (ymin + delta) >> shift, (ymax + delta) >> shift);
        if npts < 3 || xmax < 0 || ymax < 0 || xmin >= w || ymin >= h {
            return;
        }
        let ymax = ymax.min(h - 1);
        #[derive(Clone, Copy)]
        struct Edge {
            idx: usize,
            di: usize,
            x: i64,
            dx: i64,
            ye: i64,
        }
        let mut edge = [Edge { idx: imin, di: 1, x: -XY_ONE, dx: 0, ye: ymin }, Edge { idx: imin, di: npts - 1, x: -XY_ONE, dx: 0, ye: ymin }];
        let mut edges = npts as i64;
        let half = XY_ONE >> 1;
        let mut y = ymin;
        loop {
            for e in &mut edge {
                if y >= e.ye {
                    let mut idx0 = e.idx;
                    let mut idx = (idx0 + e.di) % npts;
                    loop {
                        if edges <= 0 {
                            edges -= 1;
                            break;
                        }
                        edges -= 1;
                        let ty = (v[idx].1 + delta) >> shift;
                        if ty > y {
                            let (xs, xe) = (v[idx0].0 << (XY_SHIFT - shift), v[idx].0 << (XY_SHIFT - shift));
                            e.ye = ty;
                            e.dx = ((xe - xs) * 2 + (ty - y)) / (2 * (ty - y));
                            e.x = xs;
                            e.idx = idx;
                            break;
                        }
                        idx0 = idx;
                        idx = (idx + e.di) % npts;
                    }
                }
            }
            if edges < 0 {
                break;
            }
            if y >= 0 {
                let (l, r) = if edge[0].x > edge[1].x { (1, 0) } else { (0, 1) };
                let x1 = (edge[l].x + half) >> XY_SHIFT;
                let x2 = (edge[r].x + half) >> XY_SHIFT;
                if x2 >= 0 && x1 < w {
                    self.hline(y as usize, x1.max(0) as usize, x2.min(w - 1) as usize);
                }
            }
            edge[0].x += edge[0].dx;
            edge[1].x += edge[1].dx;
            y += 1;
            if y > ymax {
                break;
            }
        }
    }
}

fn color_of<T: Element>(color: Scalar, cn: usize) -> Vec<T> {
    (0..cn).map(|c| T::from_f64(color.get(c))).collect()
}

/// `cv::fillConvexPoly(img, pts, color, lineType)` with integer vertices.
pub fn fill_convex_poly<T: Element>(img: &mut Mat<T>, pts: &[Point], color: impl Into<Scalar>, line_type: LineType) {
    let cn = img.channels();
    let v: Vec<(i64, i64)> = pts.iter().map(|p| (i64::from(p.x), i64::from(p.y))).collect();
    let color = color_of(color.into(), cn);
    Canvas { img, color }.fill_convex(&v, line_type, 0);
}

/// `cv::ellipse(img, center, axes, angle, 0, 360, color, thickness=-1, lineType)`: a filled ellipse.
pub fn fill_ellipse<T: Element>(
    img: &mut Mat<T>,
    center: Point,
    axes: (i32, i32),
    angle: f64,
    color: impl Into<Scalar>,
    line_type: LineType,
) -> Result<()> {
    if axes.0 < 0 || axes.1 < 0 {
        return Err(invalid("fill_ellipse", "negative axes"));
    }
    let (cx, cy) = (i64::from(center.x) << XY_SHIFT, i64::from(center.y) << XY_SHIFT);
    let (ax, ay) = (i64::from(axes.0) << XY_SHIFT, i64::from(axes.1) << XY_SHIFT);
    let delta = ((ax.max(ay) + (XY_ONE >> 1)) >> XY_SHIFT) as i32;
    let delta = if delta < 3 { 90 } else if delta < 10 { 30 } else if delta < 15 { 18 } else { 5 };
    let raw = ellipse2poly_f64((cx as f64, cy as f64), (ax as f64, ay as f64), cv_round(angle), 0, 360, delta);
    let mut v: Vec<(i64, i64)> = Vec::new();
    for (x, y) in raw {
        let mut p = (i64::from(cv_round(x / XY_ONE as f64)) << XY_SHIFT, i64::from(cv_round(y / XY_ONE as f64)) << XY_SHIFT);
        p.0 += i64::from(cv_round(x - p.0 as f64));
        p.1 += i64::from(cv_round(y - p.1 as f64));
        if v.last() != Some(&p) {
            v.push(p);
        }
    }
    if v.len() <= 1 {
        v = vec![(cx, cy); 2];
    }
    let cn = img.channels();
    let color = color_of(color.into(), cn);
    Canvas { img, color }.fill_convex(&v, line_type, XY_SHIFT);
    Ok(())
}

/// `cv::boxPoints(rect)`: corners in OpenCV's order (bottom-left first for angle 0).
pub fn box_points(rect: RotatedRect) -> [Point2f; 4] {
    let angle = f64::from(rect.angle) * std::f64::consts::PI / 180.0;
    let b = (angle.cos() * 0.5) as f32;
    let a = (angle.sin() * 0.5) as f32;
    let (c, w, h) = (rect.center, rect.width, rect.height);
    let p0 = Point2f::new(c.x - a * h - b * w, c.y + b * h - a * w);
    let p1 = Point2f::new(c.x + a * h - b * w, c.y - b * h - a * w);
    let p2 = Point2f::new(2.0 * c.x - p0.x, 2.0 * c.y - p0.y);
    let p3 = Point2f::new(2.0 * c.x - p1.x, 2.0 * c.y - p1.y);
    [p0, p1, p2, p3]
}

/// `cv::boundingRect(mask)`: the tight box around nonzero pixels (empty when none).
pub fn bounding_rect<'a, T: Element>(mask: impl Into<MatView<'a, T>>) -> Rect {
    let mask = mask.into();
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
    for y in 0..mask.height() {
        for (x, px) in mask.row(y).chunks_exact(mask.channels()).enumerate() {
            if px.iter().any(|v| v.to_f64() != 0.0) {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
    }
    if x0 == usize::MAX { Rect::default() } else { Rect::new(x0, y0, x1 - x0 + 1, y1 - y0 + 1) }
}

/// `cv::boundingRect(points)` for integer points.
pub fn bounding_rect_points(points: &[Point]) -> Option<(Point, i32, i32)> {
    let first = points.first()?;
    let (mut x0, mut y0, mut x1, mut y1) = (first.x, first.y, first.x, first.y);
    for p in points {
        x0 = x0.min(p.x);
        y0 = y0.min(p.y);
        x1 = x1.max(p.x);
        y1 = y1.max(p.y);
    }
    Some((Point::new(x0, y0), x1 - x0 + 1, y1 - y0 + 1))
}
