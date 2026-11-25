//! Tencent Yuanbao "元宝 / AI生成" two-line mark, bottom-right (`yuanbao_engine.py`).

use super::{FamilyMask, TextMarkFamily};
use crate::text_mark::{Corner, Frontend, PostGate, ScaleBasis, TextMarkConfig, TextMarkEngine};

pub const DETECT_NCC_THRESHOLD: f64 = 0.395;

pub const CONFIG: TextMarkConfig = TextMarkConfig {
    name: "Tencent Yuanbao",
    asset: "yuanbao_alpha.png",
    corner: Corner::Br,
    margin_floor: 4,
    width_frac: 0.20,
    height_frac: 0.15,
    margin_x_frac: 0.002,
    margin_bottom_frac: 0.002,
    max_saturation: 55.0,
    logo_min_luma: 150.0,
    tophat_delta: 12.0,
    morph_open_size: 5,
    detect_min_coverage: 0.04,
    detect_ncc_threshold: DETECT_NCC_THRESHOLD,
    frontend: Frontend::Contrast,
    scale_basis: ScaleBasis::Short,
    ladder: &[0.95, 1.0, 1.05],
    alpha_width_frac: 0.08,
    alpha_height_frac: 0.0446,
    min_gw: 32,
    provenance_ncc_factor: 1.0,
    ..TextMarkConfig::DEFAULTS
};

pub const COMPACT_CONFIG: TextMarkConfig = TextMarkConfig {
    asset: "yuanbao_compact_alpha.png",
    width_frac: 0.28,
    height_frac: 0.13,
    scale_basis: ScaleBasis::Width,
    alpha_width_frac: 0.09,
    alpha_height_frac: 0.09 * 238.0 / 387.0 * 0.75,
    ladder: &[0.95, 1.0, 1.05],
    detect_ncc_threshold: 0.50,
    provenance_ncc_factor: 0.62,
    max_saturation: 255.0,
    ..CONFIG
};

pub const COMPACT_GRAY_CONFIG: TextMarkConfig = TextMarkConfig {
    asset: "yuanbao_compact_gray_alpha.png",
    width_frac: 0.28,
    height_frac: 0.13,
    scale_basis: ScaleBasis::Width,
    alpha_width_frac: 0.06,
    alpha_height_frac: 0.06 * 238.0 / 387.0 * 0.90,
    ladder: &[0.95, 1.0, 1.05],
    frontend: Frontend::Gray,
    detect_ncc_threshold: 0.50,
    provenance_ncc_factor: 0.50,
    max_saturation: 255.0,
    ..CONFIG
};

/// The bottom-right anchor plus both stacked lines and 2D glyph detail.
pub const GATE: PostGate = PostGate::TwoLine { max_right: 0.04, max_bottom: 0.04, min_line: 0.36, min_detail: 0.31 };

pub fn engine() -> TextMarkFamily {
    TextMarkFamily::new(
        TextMarkEngine::new(CONFIG).with_gate(GATE),
        vec![
            TextMarkEngine::bottom_right_box(COMPACT_CONFIG, 0.0, 0.0),
            TextMarkEngine::bottom_right_box(COMPACT_GRAY_CONFIG, 0.0, 0.0),
        ],
        FamilyMask::Layout,
    )
}
