//! ByteDance Jimeng / Dreamina "★ 即梦AI" wordmark, bottom-right (`jimeng_engine.py`).

use super::TextMark;
use crate::text_mark::{Corner, TextMarkConfig, TextMarkEngine};

pub const DETECT_NCC_THRESHOLD: f64 = 0.45;
pub const PROVENANCE_NCC_FACTOR: f64 = 0.7;

pub const CONFIG: TextMarkConfig = TextMarkConfig {
    name: "Jimeng",
    asset: "jimeng_alpha.png",
    corner: Corner::Br,
    margin_floor: 4,
    width_frac: 0.27,
    height_frac: 0.092,
    margin_x_frac: 0.008,
    margin_bottom_frac: 0.010,
    max_saturation: 55.0,
    logo_min_luma: 150.0,
    tophat_delta: 12.0,
    morph_open_size: 5,
    detect_min_coverage: 0.02,
    detect_ncc_threshold: DETECT_NCC_THRESHOLD,
    provenance_ncc_factor: PROVENANCE_NCC_FACTOR,
    rivals: &["doubao_alpha.png"],
    alpha_width_frac: 0.2021,
    alpha_height_frac: 0.0576,
    min_gw: 8,
    ..TextMarkConfig::DEFAULTS
};

pub fn engine() -> TextMark {
    TextMark::new(TextMarkEngine::new(CONFIG))
}
