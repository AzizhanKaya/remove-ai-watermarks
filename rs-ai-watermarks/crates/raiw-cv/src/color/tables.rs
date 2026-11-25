//! Integer lookup tables of `RGB2HSV_b` (hue range 180).

use std::sync::OnceLock;

use crate::element::cv_round;

const HSV_SHIFT: i32 = 12;

struct HsvTables {
    sdiv: [i32; 256],
    hdiv180: [i32; 256],
}

fn hsv_tables() -> &'static HsvTables {
    static TABLES: OnceLock<HsvTables> = OnceLock::new();
    TABLES.get_or_init(|| {
        let mut sdiv = [0; 256];
        let mut hdiv180 = [0; 256];
        for i in 1..256 {
            sdiv[i] = cv_round(f64::from(255 << HSV_SHIFT) / i as f64);
            hdiv180[i] = cv_round(f64::from(180 << HSV_SHIFT) / (6.0 * i as f64));
        }
        HsvTables { sdiv, hdiv180 }
    })
}

#[inline]
pub(super) fn hsv_u8(b: u8, g: u8, r: u8) -> [u8; 3] {
    let t = hsv_tables();
    let (b, g, r) = (i32::from(b), i32::from(g), i32::from(r));
    let v = b.max(g).max(r);
    let vmin = b.min(g).min(r);
    let diff = v - vmin;
    let vr = if v == r { -1 } else { 0 };
    let vg = if v == g { -1 } else { 0 };
    let s = (diff * t.sdiv[v as usize] + (1 << (HSV_SHIFT - 1))) >> HSV_SHIFT;
    let h = (vr & (g - b)) + (!vr & ((vg & (b - r + 2 * diff)) + (!vg & (r - g + 4 * diff))));
    let mut h = (h * t.hdiv180[diff as usize] + (1 << (HSV_SHIFT - 1))) >> HSV_SHIFT;
    if h < 0 {
        h += 180;
    }
    [h.clamp(0, 255) as u8, s as u8, v as u8]
}
