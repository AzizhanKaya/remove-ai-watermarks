//! Microsoft top-right white AI badge or beige text pill (`microsoft_engine.py`).

use super::{FamilyMask, TextMarkFamily};
use crate::text_mark::{Corner, Frontend, ScaleBasis, TextMarkConfig, TextMarkEngine};

pub const DETECT_NCC_THRESHOLD: f64 = 0.38;

pub const CONFIG: TextMarkConfig = TextMarkConfig {
    name: "Microsoft top-right AI badge",
    asset: "microsoft_alpha.png",
    corner: Corner::Tr,
    margin_floor: 2,
    width_frac: 0.170,
    height_frac: 0.055,
    margin_x_frac: 0.004,
    margin_bottom_frac: 0.003,
    max_saturation: 60.0,
    logo_min_luma: 170.0,
    tophat_delta: 10.0,
    morph_open_size: 5,
    detect_min_coverage: 0.30,
    detect_ncc_threshold: DETECT_NCC_THRESHOLD,
    alpha_width_frac: 0.152,
    alpha_height_frac: 0.040,
    min_gw: 24,
    frontend: Frontend::Binary,
    scale_basis: ScaleBasis::Long,
    provenance_ncc_factor: 0.7,
    ..TextMarkConfig::DEFAULTS
};

pub const TEXT_CONFIG: TextMarkConfig = TextMarkConfig {
    asset: "microsoft_text_alpha.png",
    width_frac: 0.20,
    height_frac: 0.065,
    margin_x_frac: 0.002,
    margin_bottom_frac: 0.002,
    scale_basis: ScaleBasis::Width,
    alpha_width_frac: 0.14,
    alpha_height_frac: 0.14 * 49.0 / 339.0,
    frontend: Frontend::Contrast,
    ladder: &[0.95, 1.0, 1.05],
    detect_ncc_threshold: 0.60,
    provenance_ncc_factor: 1.0,
    max_saturation: 255.0,
    ..CONFIG
};

pub fn engine() -> TextMarkFamily {
    TextMarkFamily::new(
        TextMarkEngine::new(CONFIG),
        vec![TextMarkEngine::glyph(TEXT_CONFIG)],
        FamilyMask::Microsoft,
    )
}
