//! Samsung Galaxy AI Italian or Korean disclosure, bottom-left (`samsung_engine.py`).

use super::{FamilyMask, TextMarkFamily};
use crate::text_mark::{Corner, Frontend, TextMarkConfig, TextMarkEngine};

pub const DETECT_NCC_THRESHOLD: f64 = 0.50;
pub const PROVENANCE_NCC_FACTOR: f64 = 0.40 / DETECT_NCC_THRESHOLD;

pub const CONFIG: TextMarkConfig = TextMarkConfig {
    name: "Samsung Galaxy AI",
    asset: "samsung_alpha.png",
    corner: Corner::Bl,
    margin_floor: 2,
    width_frac: 0.40,
    height_frac: 0.060,
    margin_x_frac: 0.004,
    margin_bottom_frac: 0.002,
    max_saturation: 55.0,
    logo_min_luma: 110.0,
    tophat_delta: 8.0,
    morph_open_size: 3,
    detect_min_coverage: 0.01,
    detect_ncc_threshold: DETECT_NCC_THRESHOLD,
    frontend: Frontend::Tophat,
    ladder: &[1.0],
    provenance_ncc_factor: PROVENANCE_NCC_FACTOR,
    alpha_width_frac: 0.3195,
    alpha_height_frac: 0.0378,
    min_gw: 16,
    ..TextMarkConfig::DEFAULTS
};

pub const KOREAN_CONFIG: TextMarkConfig = TextMarkConfig {
    asset: "samsung_ko_alpha.png",
    width_frac: 0.28,
    height_frac: 0.065,
    alpha_width_frac: 0.20,
    alpha_height_frac: 0.20 * 49.0 / 344.0,
    ladder: &[0.95, 1.0, 1.05],
    max_saturation: 255.0,
    frontend: Frontend::Contrast,
    detect_ncc_threshold: 0.46,
    provenance_ncc_factor: 1.0,
    ..CONFIG
};

pub fn engine() -> TextMarkFamily {
    TextMarkFamily::new(
        TextMarkEngine::new(CONFIG),
        vec![TextMarkEngine::glyph(KOREAN_CONFIG)],
        FamilyMask::Samsung,
    )
}
