//! Brand-less bare "AI生成" TC260 label, bottom-right (`generic_ai_label_engine.py`).
//!
//! Calibration is synthetic only; treat it as weaker evidence than the brand-tuned engines.

use super::TextMark;
use crate::text_mark::{Corner, Frontend, MaskPolicy, ScaleBasis, TextMarkConfig, TextMarkEngine};

pub const DETECT_NCC_THRESHOLD: f64 = 0.50;

pub const CONFIG: TextMarkConfig = TextMarkConfig {
    name: "Generic AI-generated label",
    asset: "generic_ai_label_alpha.png",
    corner: Corner::Br,
    margin_floor: 4,
    width_frac: 0.28,
    height_frac: 0.10,
    margin_x_frac: 0.010,
    margin_bottom_frac: 0.010,
    max_saturation: 55.0,
    logo_min_luma: 150.0,
    tophat_delta: 12.0,
    morph_open_size: 5,
    detect_min_coverage: 0.04,
    detect_ncc_threshold: DETECT_NCC_THRESHOLD,
    frontend: Frontend::Tophat,
    scale_basis: ScaleBasis::Width,
    rivals: &["doubao_alpha.png", "qwen_alpha.png", "baidu_alpha.png", "kling_alpha.png", "yuanbao_alpha.png"],
    alpha_width_frac: 0.1724,
    alpha_height_frac: 0.0625,
    min_gw: 8,
    provenance_ncc_factor: 1.0,
    ..TextMarkConfig::DEFAULTS
};

pub fn engine() -> TextMark {
    TextMark::new(TextMarkEngine::new(CONFIG).with_mask(MaskPolicy::Aligned { alpha_floor: 0.05, dilate: 1 }))
}
