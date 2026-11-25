//! Per-vendor engines (`*_engine.py`) and the engine contract the registry drives.

pub mod baidu;
pub mod dola;
pub mod doubao;
pub mod family;
pub mod gemini;
pub mod generic_ai_label;
pub mod genspark;
pub mod jimeng;
pub mod kling;
pub mod liblib;
pub mod microsoft;
pub mod openart;
pub mod pill;
pub mod qwen;
pub mod runninghub;
pub mod samsung;
pub mod wan;
pub mod workbuddy;
pub mod yuanbao;

use std::collections::BTreeMap;
use std::fmt;

use crate::error::Result;
use crate::geometry::Region;
use crate::raster::{Image, Mask};
use crate::text_mark::{FootprintArgs, TextMarkDetection, TextMarkEngine, TextMarkScan};

pub use family::{FamilyMask, TextMarkFamily};
pub use gemini::{GeminiEngine, SparkleDetection, WatermarkSize};
pub use pill::{LibLibPillEngine, PillDetection, PillEngine};
pub use qwen::QwenEngine;

/// Physical measurements a mark's gate may need (`KnownMark._features`).
pub type Features = BTreeMap<&'static str, f64>;

/// The engine's own detection, threaded back into its mask builder.
#[derive(Clone, Debug, PartialEq)]
pub enum EngineDetection {
    Text(TextMarkDetection),
    Sparkle(SparkleDetection),
    Pill(PillDetection),
}

impl EngineDetection {
    pub fn detected(&self) -> bool {
        match self {
            Self::Text(d) => d.detected,
            Self::Sparkle(d) => d.detected,
            Self::Pill(d) => d.detected,
        }
    }

    pub fn confidence(&self) -> f64 {
        match self {
            Self::Text(d) => d.confidence,
            Self::Sparkle(d) => d.confidence,
            Self::Pill(d) => d.confidence,
        }
    }

    pub fn region(&self) -> Region {
        match self {
            Self::Text(d) => d.region,
            Self::Sparkle(d) => d.region,
            Self::Pill(d) => d.region,
        }
    }

    /// Whether the pill's solid label template verified the text.
    pub fn verified_label(&self) -> bool {
        matches!(self, Self::Pill(d) if d.verified_label)
    }
}

/// A detection type an engine produces.
pub trait EngineOutput: Clone + fmt::Debug + Send + Sync {
    fn into_engine(self) -> EngineDetection;
    fn from_engine(detection: &EngineDetection) -> Option<&Self>;
}

impl EngineOutput for TextMarkDetection {
    fn into_engine(self) -> EngineDetection {
        EngineDetection::Text(self)
    }

    fn from_engine(detection: &EngineDetection) -> Option<&Self> {
        match detection {
            EngineDetection::Text(d) => Some(d),
            _ => None,
        }
    }
}

/// One visible-mark detector: a trust-level-blind scan, then a verdict at a trust level.
///
/// `detect` and `detect_both` are derived from the two halves, so they reach the same numbers
/// by construction; per-mark demotions belong in `verdict` (Python's `_post_gate`).
pub trait MarkEngine: Send + Sync + fmt::Debug {
    type Scan;
    type Detection: EngineOutput;

    fn scan(&self, image: &Image) -> Result<Self::Scan>;
    fn verdict(&self, scan: &Self::Scan, provenance: bool) -> Result<Self::Detection>;

    /// Full-frame mask (255 = remove) of the mark's footprint, bounded by `detection` when given.
    fn footprint_mask(
        &self,
        image: &Image,
        force: bool,
        dilate: Option<i32>,
        detection: Option<&Self::Detection>,
    ) -> Result<Option<Mask>>;

    /// Gate features (computed only for detected marks).
    fn features(&self, _image: &Image) -> Result<Features> {
        Ok(Features::new())
    }

    fn detect(&self, image: &Image, provenance: bool) -> Result<Self::Detection> {
        let scan = self.scan(image)?;
        self.verdict(&scan, provenance)
    }

    /// `(strict, relaxed)` from one scan.
    fn detect_both(&self, image: &Image) -> Result<(Self::Detection, Self::Detection)> {
        let scan = self.scan(image)?;
        Ok((self.verdict(&scan, false)?, self.verdict(&scan, true)?))
    }
}

/// The object-safe engine view the registry holds.
pub trait VisibleEngine: Send + Sync + fmt::Debug {
    fn detect(&self, image: &Image, provenance: bool) -> Result<EngineDetection>;
    fn detect_both(&self, image: &Image) -> Result<(EngineDetection, EngineDetection)>;
    fn footprint_mask(&self, image: &Image, force: bool, detection: Option<&EngineDetection>) -> Result<Option<Mask>>;
    fn features(&self, image: &Image) -> Result<Features>;
}

impl<E: MarkEngine> VisibleEngine for E {
    fn detect(&self, image: &Image, provenance: bool) -> Result<EngineDetection> {
        Ok(MarkEngine::detect(self, image, provenance)?.into_engine())
    }

    fn detect_both(&self, image: &Image) -> Result<(EngineDetection, EngineDetection)> {
        let (strict, relaxed) = MarkEngine::detect_both(self, image)?;
        Ok((strict.into_engine(), relaxed.into_engine()))
    }

    fn footprint_mask(&self, image: &Image, force: bool, detection: Option<&EngineDetection>) -> Result<Option<Mask>> {
        let detection = detection.and_then(E::Detection::from_engine);
        MarkEngine::footprint_mask(self, image, force, None, detection)
    }

    fn features(&self, image: &Image) -> Result<Features> {
        MarkEngine::features(self, image)
    }
}

/// A single-layout text mark (the plain `TextMarkEngine` subclasses).
#[derive(Clone, Debug)]
pub struct TextMark {
    pub engine: TextMarkEngine,
}

impl TextMark {
    pub const fn new(engine: TextMarkEngine) -> Self {
        Self { engine }
    }

    pub fn locate(&self, height: usize, width: usize) -> crate::text_mark::Location {
        self.engine.locate(height, width)
    }
}

impl MarkEngine for TextMark {
    type Scan = TextMarkScan;
    type Detection = TextMarkDetection;

    fn scan(&self, image: &Image) -> Result<TextMarkScan> {
        self.engine.scan(image)
    }

    fn verdict(&self, scan: &TextMarkScan, provenance: bool) -> Result<TextMarkDetection> {
        self.engine.verdict(scan, provenance)
    }

    fn footprint_mask(
        &self,
        image: &Image,
        force: bool,
        dilate: Option<i32>,
        detection: Option<&TextMarkDetection>,
    ) -> Result<Option<Mask>> {
        let redetect = |img: &Image| self.engine.detect(img, false);
        self.engine.footprint_mask(image, FootprintArgs { force, dilate, detection }, &redetect)
    }
}
