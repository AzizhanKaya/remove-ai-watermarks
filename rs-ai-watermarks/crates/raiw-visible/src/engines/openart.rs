//! OpenArt bowtie icon plus "OpenArt" wordmark, frame-center (`openart_engine.py`).
//!
//! Uncalibrated: one confirmed carrier and no captured corpus; do not lower the gate.

use super::TextMark;
use crate::text_mark::{Corner, Frontend, ScaleBasis, TextMarkConfig, TextMarkEngine};

pub const DETECT_NCC_THRESHOLD: f64 = 0.62;

pub const CONFIG: TextMarkConfig = TextMarkConfig {
    name: "OpenArt",
    asset: "openart_alpha.png",
    corner: Corner::Cc,
    margin_floor: 4,
    width_frac: 0.55,
    height_frac: 0.18,
    margin_x_frac: 0.0,
    margin_bottom_frac: 0.0,
    max_saturation: 55.0,
    logo_min_luma: 150.0,
    tophat_delta: 12.0,
    morph_open_size: 5,
    detect_min_coverage: 0.02,
    detect_ncc_threshold: DETECT_NCC_THRESHOLD,
    provenance_ncc_factor: 1.0,
    frontend: Frontend::Tophat,
    scale_basis: ScaleBasis::Short,
    alpha_width_frac: 360.0 / 1024.0,
    alpha_height_frac: 90.0 / 1024.0,
    min_gw: 12,
    ..TextMarkConfig::DEFAULTS
};

pub fn engine() -> TextMark {
    TextMark::new(TextMarkEngine::new(CONFIG))
}
