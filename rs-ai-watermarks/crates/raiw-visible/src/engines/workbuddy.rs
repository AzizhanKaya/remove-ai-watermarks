//! Tencent WorkBuddy "AI生成 / WORKBUDDY" disclosure, bottom-right (`workbuddy_engine.py`).

use super::TextMark;
use crate::text_mark::{Corner, Frontend, ScaleBasis, TextMarkConfig, TextMarkEngine};

pub const CONFIG: TextMarkConfig = TextMarkConfig {
    name: "WorkBuddy AI disclosure",
    asset: "workbuddy_alpha.png",
    corner: Corner::Br,
    margin_floor: 4,
    width_frac: 0.22,
    height_frac: 0.13,
    margin_x_frac: 0.002,
    margin_bottom_frac: 0.002,
    max_saturation: 255.0,
    logo_min_luma: 110.0,
    tophat_delta: 8.0,
    morph_open_size: 3,
    detect_min_coverage: 0.01,
    detect_ncc_threshold: 0.60,
    alpha_width_frac: 0.12,
    alpha_height_frac: 0.12 * 177.0 / 351.0,
    frontend: Frontend::Tophat,
    scale_basis: ScaleBasis::Width,
    ladder: &[0.95, 1.0, 1.05],
    provenance_ncc_factor: 0.90,
    min_gw: 24,
    ..TextMarkConfig::DEFAULTS
};

pub fn engine() -> TextMark {
    TextMark::new(TextMarkEngine::bottom_right_box(CONFIG, 0.0, 0.0))
}
