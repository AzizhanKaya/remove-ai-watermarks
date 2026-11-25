//! LiblibAI triangle logo plus "LiblibAI" wordmark, bottom-center (`liblib_engine.py`).

use super::TextMark;
use crate::text_mark::{Corner, Frontend, RectPolicy, ScaleBasis, ScanHook, TextMarkConfig, TextMarkEngine};

pub const DETECT_NCC_THRESHOLD: f64 = 0.42;

/// Per-mark size floor against small generic icons.
pub const MIN_SHORT_SIDE: usize = 480;

pub const CONFIG: TextMarkConfig = TextMarkConfig {
    name: "LiblibAI",
    asset: "liblib_alpha.png",
    corner: Corner::Bc,
    margin_floor: 4,
    width_frac: 0.20,
    height_frac: 0.09,
    margin_x_frac: 0.0,
    margin_bottom_frac: 0.02,
    max_saturation: 55.0,
    logo_min_luma: 150.0,
    tophat_delta: 12.0,
    morph_open_size: 5,
    detect_min_coverage: 0.04,
    detect_ncc_threshold: DETECT_NCC_THRESHOLD,
    frontend: Frontend::Tophat,
    scale_basis: ScaleBasis::Width,
    ladder: &[0.9, 1.0, 1.1],
    alpha_width_frac: 0.10,
    alpha_height_frac: 0.026,
    min_gw: 8,
    provenance_ncc_factor: 1.0,
    ..TextMarkConfig::DEFAULTS
};

pub fn engine() -> TextMark {
    TextMark::new(
        TextMarkEngine::new(CONFIG)
            .with_hook(ScanHook::MinShortSide(MIN_SHORT_SIDE))
            .with_rect(RectPolicy::LiblibLogo),
    )
}
