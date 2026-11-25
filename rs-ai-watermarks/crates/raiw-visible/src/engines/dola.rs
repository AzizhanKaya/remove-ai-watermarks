//! ByteDance Dola "Dola AI" image disclosure, bottom-right (`dola_engine.py`).

use super::TextMark;
use crate::text_mark::{Corner, Frontend, ScaleBasis, TextMarkConfig, TextMarkEngine};

pub const CONFIG: TextMarkConfig = TextMarkConfig {
    name: "Dola AI image disclosure",
    asset: "dola_alpha.png",
    corner: Corner::Br,
    margin_floor: 4,
    width_frac: 0.28,
    height_frac: 0.09,
    margin_x_frac: 0.002,
    margin_bottom_frac: 0.002,
    max_saturation: 255.0,
    logo_min_luma: 110.0,
    tophat_delta: 8.0,
    morph_open_size: 3,
    detect_min_coverage: 0.01,
    detect_ncc_threshold: 0.50,
    alpha_width_frac: 0.10,
    alpha_height_frac: 0.10 * 86.0 / 347.0,
    frontend: Frontend::Tophat,
    scale_basis: ScaleBasis::Width,
    ladder: &[0.95, 1.0, 1.05, 1.20, 1.26],
    provenance_ncc_factor: 1.0,
    min_gw: 24,
    ..TextMarkConfig::DEFAULTS
};

pub fn engine() -> TextMark {
    TextMark::new(TextMarkEngine::boxed(CONFIG, 0.0, 0.0))
}
