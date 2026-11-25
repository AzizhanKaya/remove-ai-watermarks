//! Kuaishou Kling AI "可灵AI 3.0/4.0" and "KlingAI 3.0/4.0" strips, bottom-right (`kling_engine.py`).

use super::{FamilyMask, TextMarkFamily};
use crate::text_mark::{Corner, Frontend, Ladder, ScaleBasis, TextMarkConfig, TextMarkEngine};

pub const DETECT_NCC_THRESHOLD: f64 = 0.35;

pub const CONFIG: TextMarkConfig = TextMarkConfig {
    name: "Kling AI",
    asset: "kling_alpha.png",
    corner: Corner::Br,
    margin_floor: 4,
    width_frac: 0.19,
    height_frac: 0.05,
    margin_x_frac: 0.03,
    margin_bottom_frac: 0.023,
    max_saturation: 55.0,
    logo_min_luma: 150.0,
    tophat_delta: 12.0,
    morph_open_size: 5,
    detect_min_coverage: 0.04,
    detect_ncc_threshold: DETECT_NCC_THRESHOLD,
    frontend: Frontend::Tophat,
    scale_basis: ScaleBasis::Short,
    alpha_width_frac: 0.12,
    alpha_height_frac: 0.0287,
    min_gw: 8,
    provenance_ncc_factor: 1.0,
    ..TextMarkConfig::DEFAULTS
};

pub const CJK_4_CONFIG: TextMarkConfig = TextMarkConfig { asset: "kling_4_alpha.png", ..CONFIG };

pub const LATIN_CONFIG: TextMarkConfig = TextMarkConfig {
    asset: "kling_latin_alpha.png",
    width_frac: 0.22,
    height_frac: 0.07,
    margin_x_frac: 0.002,
    margin_bottom_frac: 0.002,
    alpha_width_frac: 0.14,
    alpha_height_frac: 0.032,
    ladder: &[0.9, 1.0, 1.1],
    detect_ncc_threshold: 0.40,
    ..CONFIG
};

pub const LATIN_4_CONFIG: TextMarkConfig = TextMarkConfig { asset: "kling_latin_4_alpha.png", ..LATIN_CONFIG };

fn text(config: TextMarkConfig) -> TextMarkEngine {
    TextMarkEngine::new(config).with_ladder(Ladder::GlyphDetail)
}

pub fn engine() -> TextMarkFamily {
    TextMarkFamily::new(
        text(CONFIG),
        vec![text(CJK_4_CONFIG), text(LATIN_CONFIG), text(LATIN_4_CONFIG)],
        FamilyMask::Kling,
    )
}
