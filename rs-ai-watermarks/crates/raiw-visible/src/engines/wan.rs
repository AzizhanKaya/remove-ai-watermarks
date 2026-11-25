//! Alibaba Wan logo plus "Wan" wordmark, bottom-right (`wan_engine.py`). Provisional calibration.

use super::TextMark;
use crate::text_mark::{Corner, Frontend, ScaleBasis, TextMarkConfig, TextMarkEngine};

pub const DETECT_NCC_THRESHOLD: f64 = 0.55;

pub const CONFIG: TextMarkConfig = TextMarkConfig {
    name: "Wan",
    asset: "wan_alpha.png",
    corner: Corner::Br,
    margin_floor: 4,
    width_frac: 0.17,
    height_frac: 0.075,
    margin_x_frac: 0.002,
    margin_bottom_frac: 0.002,
    max_saturation: 55.0,
    logo_min_luma: 150.0,
    tophat_delta: 12.0,
    morph_open_size: 5,
    detect_min_coverage: 0.04,
    detect_ncc_threshold: DETECT_NCC_THRESHOLD,
    frontend: Frontend::Tophat,
    scale_basis: ScaleBasis::Short,
    alpha_width_frac: 268.0 / 2048.0,
    alpha_height_frac: 103.0 / 2048.0,
    min_gw: 8,
    provenance_ncc_factor: 1.0,
    ..TextMarkConfig::DEFAULTS
};

/// The match must end within 0.015 of the short side from both edges.
pub fn engine() -> TextMark {
    TextMark::new(TextMarkEngine::anchored(CONFIG, 0.015, 0.015))
}
