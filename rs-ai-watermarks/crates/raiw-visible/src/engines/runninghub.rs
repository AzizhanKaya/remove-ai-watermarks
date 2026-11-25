//! RunningHub "RunningHub AI生成" faint gray text, top-left (`runninghub_engine.py`).

use super::TextMark;
use crate::text_mark::{Corner, Frontend, PostGate, ScaleBasis, TextMarkConfig, TextMarkEngine};

pub const DETECT_NCC_THRESHOLD: f64 = 0.34;

pub const CONFIG: TextMarkConfig = TextMarkConfig {
    name: "RunningHub",
    asset: "runninghub_alpha.png",
    corner: Corner::Tl,
    margin_floor: 4,
    width_frac: 0.45,
    height_frac: 0.10,
    margin_x_frac: 0.002,
    margin_bottom_frac: 0.002,
    max_saturation: 55.0,
    logo_min_luma: 150.0,
    tophat_delta: 12.0,
    morph_open_size: 5,
    detect_min_coverage: 0.04,
    detect_ncc_threshold: DETECT_NCC_THRESHOLD,
    frontend: Frontend::Gray,
    scale_basis: ScaleBasis::Width,
    ladder: &[0.95, 1.0, 1.05],
    alpha_width_frac: 0.32,
    alpha_height_frac: 0.04,
    min_gw: 8,
    provenance_ncc_factor: 1.0,
    ..TextMarkConfig::DEFAULTS
};

/// The match must start this close to the top-left corner (fractions of the frame).
pub const GATE: PostGate = PostGate::TopLeft { max_x: 0.025, max_y: 0.015 };

pub fn engine() -> TextMark {
    TextMark::new(TextMarkEngine::new(CONFIG).with_gate(GATE))
}
