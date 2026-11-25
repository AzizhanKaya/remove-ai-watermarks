//! ByteDance Doubao "豆包AI生成" strip, bottom-right (`doubao_engine.py`).

use super::{FamilyMask, TextMarkFamily};
use crate::text_mark::{Corner, Frontend, ScaleBasis, ScanHook, TextMarkConfig, TextMarkEngine};

/// Front-end specific strict gate of the continuous top-hat response.
pub const DETECT_NCC_THRESHOLD: f64 = 0.50;

pub const CONFIG: TextMarkConfig = TextMarkConfig {
    name: "Doubao",
    asset: "doubao_alpha.png",
    corner: Corner::Br,
    margin_floor: 4,
    width_frac: 0.22,
    height_frac: 0.075,
    margin_x_frac: 0.004,
    margin_bottom_frac: 0.004,
    max_saturation: 55.0,
    logo_min_luma: 150.0,
    tophat_delta: 12.0,
    morph_open_size: 5,
    detect_min_coverage: 0.04,
    detect_ncc_threshold: DETECT_NCC_THRESHOLD,
    frontend: Frontend::Tophat,
    scale_basis: ScaleBasis::Short,
    alpha_width_frac: 0.1636,
    alpha_height_frac: 0.0405,
    min_gw: 8,
    ..TextMarkConfig::DEFAULTS
};

/// The outlined single-line layout.
pub const OUTLINE_CONFIG: TextMarkConfig = TextMarkConfig {
    asset: "doubao_outline_alpha.png",
    width_frac: 0.28,
    height_frac: 0.13,
    scale_basis: ScaleBasis::Width,
    alpha_width_frac: 0.14,
    alpha_height_frac: 0.14 * 0.75 * 77.0 / 346.0,
    max_saturation: 255.0,
    frontend: Frontend::Contrast,
    ladder: &[0.95, 1.0, 1.05],
    detect_ncc_threshold: 0.50,
    provenance_ncc_factor: 0.25,
    ..CONFIG
};

/// Tinted rescan for translucent white glyphs over strongly colored corners.
pub const TINTED: ScanHook = ScanHook::Tinted { max_saturation: 100.0, min_colored_spread: 100, min_colored_fraction: 0.10 };

/// Bright neutral strokes and edge detail the outlined layout must show.
pub const OUTLINE: ScanHook = ScanHook::Outline { min_bright_neutral: 0.29, min_mean_laplacian: 10.0 };

pub fn engine() -> TextMarkFamily {
    let base = TextMarkEngine::new(CONFIG).with_hook(TINTED);
    let outlined = TextMarkEngine::bottom_right_box(OUTLINE_CONFIG, 1.2, 0.7).with_hook(OUTLINE);
    TextMarkFamily::new(base, vec![outlined], FamilyMask::Doubao)
}
