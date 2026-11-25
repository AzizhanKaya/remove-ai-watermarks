//! Baidu "百度" text run plus white "AI生成" tag, bottom-right (`baidu_engine.py`).

use super::TextMark;
use crate::text_mark::{Corner, Frontend, RectPolicy, ScaleBasis, TextMarkConfig, TextMarkEngine};

pub const DETECT_NCC_THRESHOLD: f64 = 0.48;

pub const CONFIG: TextMarkConfig = TextMarkConfig {
    name: "Baidu",
    asset: "baidu_alpha.png",
    corner: Corner::Br,
    margin_floor: 4,
    width_frac: 0.25,
    height_frac: 0.07,
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
    ladder: &[0.95, 1.0, 1.05],
    alpha_width_frac: 0.090,
    alpha_height_frac: 0.046,
    min_gw: 8,
    rivals: &["doubao_alpha.png", "qwen_alpha.png"],
    provenance_ncc_factor: 1.0,
    ..TextMarkConfig::DEFAULTS
};

pub fn engine() -> TextMark {
    TextMark::new(TextMarkEngine::new(CONFIG).with_rect(RectPolicy::BaiduTag))
}
