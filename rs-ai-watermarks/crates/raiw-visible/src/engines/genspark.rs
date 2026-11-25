//! Genspark sparkle badge plus "Genspark" wordmark, frame-center (`genspark_engine.py`).

use super::TextMark;
use crate::text_mark::{Corner, Frontend, ScaleBasis, ScanHook, TextMarkConfig, TextMarkEngine};

pub const DETECT_NCC_THRESHOLD: f64 = 0.65;

/// Scans above this short side run on a downscaled copy.
pub const MAX_SCAN_SHORT_SIDE: usize = 960;

const ALPHA_WIDTH_FRAC: f64 = 466.0 / 960.0;

pub const CONFIG: TextMarkConfig = TextMarkConfig {
    name: "Genspark",
    asset: "genspark_alpha.png",
    corner: Corner::Cc,
    margin_floor: 4,
    width_frac: 0.72,
    height_frac: 0.24,
    margin_x_frac: 0.0,
    margin_bottom_frac: 0.0,
    max_saturation: 55.0,
    logo_min_luma: 150.0,
    tophat_delta: 12.0,
    morph_open_size: 5,
    detect_min_coverage: 0.02,
    detect_ncc_threshold: DETECT_NCC_THRESHOLD,
    provenance_ncc_factor: 1.0,
    frontend: Frontend::Tophat,
    scale_basis: ScaleBasis::Short,
    alpha_width_frac: ALPHA_WIDTH_FRAC,
    alpha_height_frac: (82.0 / 360.0) * ALPHA_WIDTH_FRAC,
    min_gw: 16,
    ..TextMarkConfig::DEFAULTS
};

pub fn engine() -> TextMark {
    TextMark::new(TextMarkEngine::new(CONFIG).with_hook(ScanHook::Downscale(MAX_SCAN_SHORT_SIDE)))
}
