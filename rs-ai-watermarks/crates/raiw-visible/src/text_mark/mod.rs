//! Shared visible text-mark detector and localizer (`_text_mark_engine.py`, `_text_mark_variants.py`).
//!
//! A mark supplies a [`TextMarkConfig`] plus the few policies Python expressed as subclasses:
//! the ladder sweep, the post-gate demotion, a scan hook, and the removal footprint. Detection is
//! a trust-level-blind [`TextMarkEngine::scan`] followed by [`TextMarkEngine::verdict`].

mod footprint;
mod scan;

pub use footprint::{FootprintArgs, MaskPolicy, RectPolicy};
pub use scan::{Fields, Ladder, PostGate, ScanHook, glyph_detail_ncc};

use raiw_cv::Mat;
use serde::Serialize;

use crate::geometry::{MatchBox, Region};

/// Minimum short side for text-mark detection (small icons correlate spuriously).
pub const MIN_DETECT_SHORT_SIDE: usize = 200;

/// Shared provenance relaxation factor of the NCC gate.
pub const DEFAULT_PROVENANCE_NCC_FACTOR: f64 = 0.7;

/// The shipped three-rung scale ladder.
pub const DEFAULT_LADDER: &[f64] = &[0.8, 1.0, 1.25];

/// Where the locate box is anchored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Corner {
    Br,
    Bl,
    Tl,
    Tr,
    Bc,
    Cc,
}

/// The image detection correlates the silhouette against.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Frontend {
    Binary,
    Tophat,
    Gray,
    Contrast,
}

/// The image dimension a mark's geometry scales with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ScaleBasis {
    Short,
    Width,
    Long,
}

/// All per-mark tuning for a text-mark detector/localizer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextMarkConfig {
    pub name: &'static str,
    pub asset: &'static str,
    pub corner: Corner,
    pub margin_floor: i32,
    pub width_frac: f64,
    pub height_frac: f64,
    pub margin_x_frac: f64,
    pub margin_bottom_frac: f64,
    pub max_saturation: f32,
    pub logo_min_luma: f32,
    pub tophat_delta: f32,
    pub morph_open_size: usize,
    pub detect_min_coverage: f64,
    pub detect_ncc_threshold: f64,
    pub alpha_width_frac: f64,
    pub alpha_height_frac: f64,
    pub min_gw: i32,
    pub frontend: Frontend,
    pub template_blur: f64,
    pub scale_basis: ScaleBasis,
    pub ladder: &'static [f64],
    pub rivals: &'static [&'static str],
    pub rival_margin: f64,
    pub provenance_ncc_factor: f64,
}

impl TextMarkConfig {
    /// The dataclass defaults for every optional field.
    pub const DEFAULTS: Self = Self {
        name: "",
        asset: "",
        corner: Corner::Br,
        margin_floor: 4,
        width_frac: 0.0,
        height_frac: 0.0,
        margin_x_frac: 0.0,
        margin_bottom_frac: 0.0,
        max_saturation: 55.0,
        logo_min_luma: 150.0,
        tophat_delta: 12.0,
        morph_open_size: 5,
        detect_min_coverage: 0.0,
        detect_ncc_threshold: 0.0,
        alpha_width_frac: 0.0,
        alpha_height_frac: 0.0,
        min_gw: 8,
        frontend: Frontend::Binary,
        template_blur: 0.0,
        scale_basis: ScaleBasis::Width,
        ladder: DEFAULT_LADDER,
        rivals: &[],
        rival_margin: 0.10,
        provenance_ncc_factor: DEFAULT_PROVENANCE_NCC_FACTOR,
    };

    /// The acceptance threshold at a trust level.
    pub fn threshold(&self, provenance: bool) -> f64 {
        self.detect_ncc_threshold * if provenance { self.provenance_ncc_factor } else { 1.0 }
    }
}

/// A located watermark box in absolute pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Location {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

impl Location {
    pub const fn bbox(&self) -> Region {
        Region::new(self.x, self.y, self.w, self.h)
    }
}

/// The verdict of one text-mark detection.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TextMarkDetection {
    pub detected: bool,
    pub confidence: f64,
    pub region: Region,
    /// Fraction of the locate box covered by binary glyph pixels.
    pub coverage: f64,
    /// Inclusive locate-box-local template match box, when any template matched.
    pub match_box: Option<MatchBox>,
    /// The trust level this verdict was taken at.
    pub provenance: bool,
    /// The asset of the winning layout (`None` for symbol and empty detections).
    pub template_asset: Option<&'static str>,
    /// Qwen's three-lobe symbol variant (`QwenSymbolDetection`).
    pub symbol: bool,
}

/// Prefer an accepted variant, then the strongest score; the first of equals wins.
pub fn best_detection<I: IntoIterator<Item = TextMarkDetection>>(detections: I) -> TextMarkDetection {
    let mut best: Option<TextMarkDetection> = None;
    for d in detections {
        let better = best
            .as_ref()
            .is_none_or(|b| (d.detected, d.confidence) > (b.detected, b.confidence));
        if better {
            best = Some(d);
        }
    }
    best.unwrap_or_default()
}

/// The located part of a scan.
#[derive(Clone, Debug)]
pub struct Located {
    pub loc: Location,
    /// Box-sized binary glyph mask (`extract_mask`).
    pub glyphs: Mat<u8>,
    pub coverage: f64,
    /// `None` when the binary front-end fell below its coverage gate.
    pub score: Option<f64>,
    pub match_box: Option<MatchBox>,
    pub match_patch: Option<Mat<u8>>,
}

/// The trust-level-blind half of text-mark detection.
#[derive(Clone, Debug, Default)]
pub struct TextMarkScan {
    /// `None` when detection stopped before any scan (empty or too-small image).
    pub located: Option<Located>,
    pub base: usize,
    /// `(height, width)` of the scanned image.
    pub frame: (usize, usize),
}

/// A configured text-mark engine: config plus the policies Python expressed as subclasses.
#[derive(Clone, Debug)]
pub struct TextMarkEngine {
    pub config: TextMarkConfig,
    pub ladder: Ladder,
    pub gate: PostGate,
    pub hook: ScanHook,
    pub rect: RectPolicy,
    pub mask: MaskPolicy,
}

impl TextMarkEngine {
    /// The base `TextMarkEngine`.
    pub const fn new(config: TextMarkConfig) -> Self {
        Self {
            config,
            ladder: Ladder::Standard,
            gate: PostGate::None,
            hook: ScanHook::None,
            rect: RectPolicy::Blob,
            mask: MaskPolicy::Rect,
        }
    }

    /// `BottomRightAnchoredEngine`.
    pub const fn anchored(config: TextMarkConfig, max_right: f64, max_bottom: f64) -> Self {
        Self::new(config).with_gate(PostGate::BottomRight { max_right, max_bottom })
    }

    /// `GlyphTextMarkEngine`.
    pub const fn glyph(config: TextMarkConfig) -> Self {
        Self::new(config).with_mask(MaskPolicy::Glyph)
    }

    /// `BoxTextMarkEngine`.
    pub const fn boxed(config: TextMarkConfig, pad_left: f64, pad_right: f64) -> Self {
        Self::new(config).with_mask(MaskPolicy::Box { pad_left, pad_right })
    }

    /// `BottomRightGlyphTextMarkEngine`: anchored sweep, anchored gate, aligned glyph mask.
    pub const fn bottom_right_glyph(config: TextMarkConfig) -> Self {
        Self {
            ladder: Ladder::Anchored { max_right: 0.04, max_bottom: 0.04 },
            gate: PostGate::BottomRight { max_right: 0.04, max_bottom: 0.04 },
            ..Self::glyph(config)
        }
    }

    /// `BottomRightBoxTextMarkEngine`.
    pub const fn bottom_right_box(config: TextMarkConfig, pad_left: f64, pad_right: f64) -> Self {
        Self::bottom_right_glyph(config).with_mask(MaskPolicy::Box { pad_left, pad_right })
    }

    pub const fn with_ladder(mut self, ladder: Ladder) -> Self {
        self.ladder = ladder;
        self
    }

    pub const fn with_gate(mut self, gate: PostGate) -> Self {
        self.gate = gate;
        self
    }

    pub const fn with_hook(mut self, hook: ScanHook) -> Self {
        self.hook = hook;
        self
    }

    pub const fn with_rect(mut self, rect: RectPolicy) -> Self {
        self.rect = rect;
        self
    }

    pub const fn with_mask(mut self, mask: MaskPolicy) -> Self {
        self.mask = mask;
        self
    }

    /// The image dimension this mark's geometry scales with.
    pub fn scale_base(&self, height: usize, width: usize) -> usize {
        match self.config.scale_basis {
            ScaleBasis::Short => height.min(width),
            ScaleBasis::Long => height.max(width),
            ScaleBasis::Width => width,
        }
    }

    /// Anchor the watermark box in the configured corner.
    pub fn locate(&self, height: usize, width: usize) -> Location {
        let c = &self.config;
        let (h, w) = (height as i32, width as i32);
        let base = self.scale_base(height, width) as f64;
        let frac = |f: f64| (base * f) as i32;
        let mut wm_w = 40.max(frac(c.width_frac));
        let mut wm_h = 16.max(frac(c.height_frac));
        let margin_x = c.margin_floor.max(frac(c.margin_x_frac));
        let margin_b = c.margin_floor.max(frac(c.margin_bottom_frac));
        let x = match c.corner {
            Corner::Br | Corner::Tr => 0.max(w - margin_x - wm_w),
            Corner::Bc | Corner::Cc => 0.max((w - wm_w).div_euclid(2)),
            Corner::Bl | Corner::Tl => margin_x.min(0.max(w - wm_w)),
        };
        let y = match c.corner {
            Corner::Tl | Corner::Tr => margin_b.min(0.max(h - wm_h)),
            Corner::Cc => 0.max((h - wm_h).div_euclid(2)),
            Corner::Br | Corner::Bl | Corner::Bc => 0.max(h - margin_b - wm_h),
        };
        wm_w = wm_w.min(w - x);
        wm_h = wm_h.min(h - y);
        Location { x, y, w: wm_w, h: wm_h }
    }

    /// Return the layout among `self` and `variants` whose asset produced `detection`.
    pub fn variant_of<'a>(&'a self, variants: &'a [Self], detection: &TextMarkDetection) -> &'a Self {
        variants
            .iter()
            .find(|v| Some(v.config.asset) == detection.template_asset)
            .unwrap_or(self)
    }
}

/// The rival marks' own configs, looked up by asset for the competitive margin.
pub fn rival_config(asset: &str) -> Option<&'static TextMarkConfig> {
    use crate::engines::{baidu, doubao, generic_ai_label, jimeng, kling, qwen, samsung, yuanbao};
    Some(match asset {
        "doubao_alpha.png" => &doubao::CONFIG,
        "jimeng_alpha.png" => &jimeng::CONFIG,
        "qwen_alpha.png" => &qwen::CONFIG,
        "samsung_alpha.png" => &samsung::CONFIG,
        "baidu_alpha.png" => &baidu::CONFIG,
        "kling_alpha.png" => &kling::CONFIG,
        "yuanbao_alpha.png" => &yuanbao::CONFIG,
        "generic_ai_label_alpha.png" => &generic_ai_label::CONFIG,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(detected: bool, confidence: f64, asset: &'static str) -> TextMarkDetection {
        TextMarkDetection { detected, confidence, template_asset: Some(asset), ..TextMarkDetection::default() }
    }

    #[test]
    fn best_prefers_accepted_then_score_then_first() {
        let pick = best_detection([det(false, 0.9, "a"), det(true, 0.4, "b"), det(true, 0.4, "c")]);
        assert_eq!(pick.template_asset, Some("b"));
        let pick = best_detection([det(false, 0.3, "a"), det(false, 0.5, "b")]);
        assert_eq!(pick.template_asset, Some("b"));
    }
}
