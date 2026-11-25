//! The visible-mark registry (`watermark_registry.py`): perception, the pure arbiter, then
//! localize -> fill -> read-only validation for every decided mark.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::OnceLock;

use serde::Serialize;

use crate::cvx::{any_nonzero, bounding_rect};
use crate::engines::{
    self, EngineDetection, Features, GeminiEngine, LibLibPillEngine, PillEngine, QwenEngine, VisibleEngine,
    gemini::GeminiMark,
};
use crate::eraser::{Backend, FillBackends, FillKind};
use crate::error::{Result, VisibleError};
use crate::geometry::Region;
use crate::raster::{Image, Mask};

/// How hard to trust a borderline mark (`--sensitivity`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Sensitivity {
    /// Relax a gate only on same-product evidence.
    #[default]
    Auto,
    /// Never relax.
    Strict,
}

/// The trust level a mark's gate resolves to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Trust {
    Strict,
    Confirmed,
}

const ASSUME_AI_REMOVED: &str = "sensitivity='assume_ai' was removed in 0.16: it relaxed EVERY mark's detection \
gate on the bare assertion that an image is AI, which says nothing about which vendor made it or where the mark \
is. If you can see a mark the detector missed, act on what you see: erase(image, region=(x, y, w, h)), or the CLI \
`--mark <name> --no-detect` for a known text mark. Use sensitivity='auto' for the default evidence-driven behavior.";

/// Parse a sensitivity, rejecting the removed `assume_ai` loudly.
pub fn validate_sensitivity(value: &str) -> Result<Sensitivity> {
    match value {
        "auto" => Ok(Sensitivity::Auto),
        "strict" => Ok(Sensitivity::Strict),
        "assume_ai" => Err(VisibleError::Sensitivity(ASSUME_AI_REMOVED.to_owned())),
        other => Err(VisibleError::Sensitivity(format!("unknown sensitivity {other:?}; expected 'auto' or 'strict'"))),
    }
}

/// Uniform detection result for a known mark.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MarkDetection {
    pub key: &'static str,
    pub label: &'static str,
    pub location: &'static str,
    pub detected: bool,
    pub confidence: f64,
    pub region: Region,
    /// The engine's own detection, threaded back into this mark's mask builder.
    #[serde(skip)]
    pub engine: Option<EngineDetection>,
}

/// A located mark: its verdict plus the full-frame removal mask.
#[derive(Clone, Debug)]
pub struct Localization {
    pub detected: bool,
    pub confidence: f64,
    pub region: Region,
    pub mask: Option<Mask>,
}

/// The fill a removal runs: backend resolution plus the masked fill itself.
pub trait Fill: Send + Sync {
    fn resolve(&self, backend: Backend) -> FillKind;
    fn fill(&self, image: &Image, mask: &Mask, kind: FillKind) -> Result<Image>;
}

impl Fill for FillBackends {
    fn resolve(&self, backend: Backend) -> FillKind {
        FillBackends::resolve(self, backend)
    }

    fn fill(&self, image: &Image, mask: &Mask, kind: FillKind) -> Result<Image> {
        FillBackends::fill(self, image, mask, kind)
    }
}

/// One registry row: where a mark lives, how to find and mask it, and what confirms its vendor.
pub struct KnownMark {
    pub key: &'static str,
    pub label: &'static str,
    pub location: &'static str,
    pub in_auto: bool,
    /// Product family for same-product corroboration.
    pub product: &'static str,
    pub manufacturer: &'static str,
    pub label_regime: Option<&'static str>,
    /// The sentence `identify` reports when this mark is the strongest evidence.
    pub platform: Option<&'static str>,
    pub provenance_signals: &'static [&'static str],
    pub provenance_platform_tokens: &'static [&'static str],
    pub tc260_producer_codes: &'static [&'static str],
    /// False for a weak detector that must not grant a sibling the relaxed trust level.
    pub can_corroborate: bool,
    engine: Box<dyn VisibleEngine>,
}

impl fmt::Debug for KnownMark {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KnownMark").field("key", &self.key).field("product", &self.product).finish_non_exhaustive()
    }
}

impl KnownMark {
    /// A row with every optional field at its default.
    pub fn new(
        key: &'static str,
        label: &'static str,
        location: &'static str,
        product: &'static str,
        manufacturer: &'static str,
        engine: Box<dyn VisibleEngine>,
    ) -> Self {
        Self {
            key,
            label,
            location,
            in_auto: true,
            product,
            manufacturer,
            label_regime: None,
            platform: None,
            provenance_signals: &[],
            provenance_platform_tokens: &[],
            tc260_producer_codes: &[],
            can_corroborate: true,
            engine,
        }
    }

    fn wrap(&self, engine: EngineDetection) -> MarkDetection {
        MarkDetection {
            key: self.key,
            label: self.label,
            location: self.location,
            detected: engine.detected(),
            confidence: engine.confidence(),
            region: engine.region(),
            engine: Some(engine),
        }
    }

    /// Verdict plus bbox (no mask); `provenance` relaxes the gate when metadata confirms the vendor.
    pub fn detect(&self, image: &Image, provenance: bool) -> Result<MarkDetection> {
        Ok(self.wrap(self.engine.detect(image, provenance)?))
    }

    /// `(strict, relaxed)` from one pass over the image.
    pub fn detect_both(&self, image: &Image) -> Result<(MarkDetection, MarkDetection)> {
        let (strict, relaxed) = self.engine.detect_both(image)?;
        Ok((self.wrap(strict), self.wrap(relaxed)))
    }

    /// The mark's own gate features.
    pub fn features(&self, image: &Image) -> Result<Features> {
        self.engine.features(image)
    }

    /// Detect (or reuse `detection`) and build the removal mask.
    pub fn localize(
        &self,
        image: &Image,
        provenance: bool,
        force: bool,
        detection: Option<&MarkDetection>,
    ) -> Result<Localization> {
        let owned;
        let det = match detection {
            Some(d) => d,
            None => {
                owned = self.detect(image, provenance)?;
                &owned
            }
        };
        let mut loc = Localization { detected: det.detected, confidence: det.confidence, region: det.region, mask: None };
        if det.detected || force {
            loc.mask = self.engine.footprint_mask(image, force, det.engine.as_ref())?;
        }
        Ok(loc)
    }

    /// Localize then fill; returns the result and the removed mark's region (None if nothing was removed).
    pub fn remove(
        &self,
        image: &Image,
        backend: Backend,
        provenance: bool,
        force: bool,
        detection: Option<&MarkDetection>,
        fill: &dyn Fill,
    ) -> Result<(Image, Option<Region>)> {
        let loc = self.localize(image, provenance, force, detection)?;
        let Some(mask) = loc.mask.filter(any_nonzero) else { return Ok((image.clone(), None)) };
        let filled = fill.fill(image, &mask, fill.resolve(backend))?;
        Ok((filled, loc.detected.then_some(loc.region)))
    }
}

/// The evidence and policy the arbiter decides against.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Context {
    pub sensitivity: Sensitivity,
    /// Vendor (mark) keys local metadata confirms.
    pub provenance: BTreeSet<String>,
    /// Marks trusted provenance from another vendor vetoes.
    pub vetoed_marks: BTreeSet<String>,
}

impl Context {
    pub fn new<P, V>(sensitivity: Sensitivity, provenance: P, vetoed_marks: V) -> Self
    where
        P: IntoIterator,
        P::Item: Into<String>,
        V: IntoIterator,
        V::Item: Into<String>,
    {
        Self {
            sensitivity,
            provenance: provenance.into_iter().map(Into::into).collect(),
            vetoed_marks: vetoed_marks.into_iter().map(Into::into).collect(),
        }
    }
}

/// One mark's perception output at both trust levels, with no policy applied.
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub key: &'static str,
    pub label: &'static str,
    pub detected_strict: bool,
    pub detected_relaxed: bool,
    pub features: Features,
    pub strict_detection: Option<MarkDetection>,
    pub relaxed_detection: Option<MarkDetection>,
}

impl Candidate {
    /// A policy-only candidate (no retained detections).
    pub fn bare(key: &'static str, strict: bool, relaxed: bool, features: Features) -> Self {
        Self {
            key,
            label: key,
            detected_strict: strict,
            detected_relaxed: relaxed,
            features,
            strict_detection: None,
            relaxed_detection: None,
        }
    }
}

/// The arbiter's verdict for one fired mark.
#[derive(Clone, Debug, PartialEq)]
pub struct Decision {
    pub candidate: Candidate,
    pub relax: bool,
}

/// Per-mark validation of one fill.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum MarkValidationStatus {
    Cleaned,
    Partial,
    Unvalidated,
}

/// Aggregate status of one automatic pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VisibleRemovalStatus {
    NoWatermark,
    Cleaned,
    Partial,
    Unvalidated,
}

/// One visible-mark fill and its read-only post-removal validation.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MarkRemovalResult {
    pub key: &'static str,
    pub label: &'static str,
    pub location: &'static str,
    pub region: Region,
    pub mask_bbox: Region,
    pub backend: FillKind,
    pub confidence_before: f64,
    pub confidence_after: Option<f64>,
    pub status: MarkValidationStatus,
    pub residual_region: Option<Region>,
}

/// The filled image plus every mark's removal record.
#[derive(Clone, Debug)]
pub struct VisibleRemovalResult {
    pub image: Image,
    pub marks: Vec<MarkRemovalResult>,
}

impl VisibleRemovalResult {
    /// Labels whose masks were filled.
    pub fn labels(&self) -> Vec<&'static str> {
        self.marks.iter().map(|m| m.label).collect()
    }

    /// A known residual takes precedence over an unknown one.
    pub fn status(&self) -> VisibleRemovalStatus {
        if self.marks.is_empty() {
            VisibleRemovalStatus::NoWatermark
        } else if self.marks.iter().any(|m| m.status == MarkValidationStatus::Partial) {
            VisibleRemovalStatus::Partial
        } else if self.marks.iter().any(|m| m.status == MarkValidationStatus::Unvalidated) {
            VisibleRemovalStatus::Unvalidated
        } else {
            VisibleRemovalStatus::Cleaned
        }
    }
}

/// The ordered set of known visible marks.
#[derive(Debug)]
pub struct Registry {
    marks: Vec<KnownMark>,
}

fn text_row(
    key: &'static str,
    label: &'static str,
    location: &'static str,
    manufacturer: &'static str,
    platform: &'static str,
    engine: Box<dyn VisibleEngine>,
) -> KnownMark {
    KnownMark {
        label_regime: Some("tc260"),
        platform: Some(platform),
        provenance_signals: &["aigc"],
        tc260_producer_codes: raiw_core::tc260::producer_codes_for_mark(key),
        ..KnownMark::new(key, label, location, key, manufacturer, engine)
    }
}

fn untagged(mut row: KnownMark) -> KnownMark {
    row.label_regime = None;
    row.provenance_signals = &[];
    row
}

impl Registry {
    /// A registry over explicit rows (tests and embedders).
    pub fn from_marks(marks: Vec<KnownMark>) -> Self {
        Self { marks }
    }

    /// Build the shipped registry, in the identify specificity order.
    pub fn build() -> Result<Self> {
        let gemini = KnownMark {
            provenance_platform_tokens: &["google", "gemini"],
            ..KnownMark::new(
                "gemini",
                "Google Gemini visible watermark (sparkle)",
                "bottom-right",
                "gemini",
                "google",
                Box::new(GeminiMark::new()?),
            )
        };
        let samsung = KnownMark {
            provenance_signals: &["samsung_genai"],
            ..untagged(text_row(
                "samsung",
                "Samsung Galaxy AI text",
                "bottom-left",
                "samsung",
                "Samsung Galaxy AI (visible 'Contenuti generati dall'AI' mark detected)",
                Box::new(engines::samsung::engine()),
            ))
        };
        let liblib_pill = KnownMark {
            label_regime: Some("tc260"),
            provenance_signals: &["aigc"],
            can_corroborate: false,
            ..KnownMark::new(
                "liblib_pill",
                "LiblibAI AI生成 pill",
                "top-left",
                "liblib",
                "liblib",
                Box::new(LibLibPillEngine::default()),
            )
        };
        let microsoft = KnownMark {
            provenance_platform_tokens: &["microsoft"],
            ..untagged(text_row(
                "microsoft",
                "Microsoft top-right AI badge",
                "top-right",
                "microsoft",
                "Microsoft (visible top-right AI badge detected)",
                Box::new(engines::microsoft::engine()),
            ))
        };
        let jimeng_pill = KnownMark {
            label_regime: Some("tc260"),
            can_corroborate: false,
            ..KnownMark::new(
                "jimeng_pill",
                "Jimeng AI生成 pill",
                "top-left",
                "jimeng",
                "bytedance",
                Box::new(PillEngine::new(true)),
            )
        };
        let kling = KnownMark {
            provenance_platform_tokens: &["kling"],
            ..text_row(
                "kling",
                "Kling AI 可灵AI / KlingAI 3.0/4.0 text",
                "bottom-right",
                "kuaishou",
                "Kuaishou Kling AI (visible 可灵AI / KlingAI mark detected)",
                Box::new(engines::kling::engine()),
            )
        };
        let marks = vec![
            gemini,
            text_row(
                "doubao",
                "Doubao 豆包AI生成 text",
                "bottom-right",
                "bytedance",
                "ByteDance Doubao (visible 豆包AI生成 mark detected)",
                Box::new(engines::doubao::engine()),
            ),
            text_row(
                "jimeng",
                "Jimeng 即梦AI wordmark",
                "bottom-right",
                "bytedance",
                "ByteDance Jimeng / Dreamina (visible 即梦AI mark detected)",
                Box::new(engines::jimeng::engine()),
            ),
            text_row(
                "qwen",
                "Qwen 千问AI生成 text or three-lobe symbol",
                "bottom-right",
                "alibaba",
                "Alibaba Cloud Qwen (visible text or symbol mark detected)",
                Box::new(QwenEngine::default()),
            ),
            text_row(
                "wan",
                "Wan logo and wordmark",
                "bottom-right",
                "alibaba",
                "Alibaba Wan (visible Wan mark detected)",
                Box::new(engines::wan::engine()),
            ),
            kling,
            text_row(
                "yuanbao",
                "Tencent Yuanbao 元宝 / AI生成 mark",
                "bottom-right",
                "tencent",
                "Tencent Yuanbao (visible 元宝 / AI生成 mark detected)",
                Box::new(engines::yuanbao::engine()),
            ),
            untagged(text_row(
                "dola",
                "Dola AI image wordmark",
                "bottom-right",
                "bytedance",
                "Dola AI (visible Dola AI disclosure)",
                Box::new(engines::dola::engine()),
            )),
            text_row(
                "workbuddy",
                "AI生成 / WORKBUDDY disclosure",
                "bottom-right",
                "tencent",
                "WorkBuddy (visible AI content disclosure)",
                Box::new(engines::workbuddy::engine()),
            ),
            samsung,
            text_row(
                "runninghub",
                "RunningHub AI生成 text",
                "top-left",
                "runninghub",
                "RunningHub (visible RunningHub AI生成 mark detected)",
                Box::new(engines::runninghub::engine()),
            ),
            text_row(
                "baidu",
                "Baidu 百度 AI生成 text",
                "bottom-right",
                "baidu",
                "Baidu (visible 百度 AI生成 mark detected)",
                Box::new(engines::baidu::engine()),
            ),
            text_row(
                "liblib",
                "LiblibAI wordmark",
                "bottom-center",
                "liblib",
                "LiblibAI (visible LiblibAI mark detected)",
                Box::new(engines::liblib::engine()),
            ),
            liblib_pill,
            microsoft,
            untagged(text_row(
                "generic_ai_label",
                "Generic bare AI生成 text (no vendor wordmark)",
                "bottom-right",
                "unknown",
                "Generic AI-generated content label (bare 'AI生成' mark; specific vendor undetermined)",
                Box::new(engines::generic_ai_label::engine()),
            )),
            untagged(text_row(
                "openart",
                "OpenArt wordmark",
                "frame-center",
                "openart",
                "OpenArt (visible OpenArt mark detected)",
                Box::new(engines::openart::engine()),
            )),
            untagged(text_row(
                "genspark",
                "Genspark badge and wordmark",
                "frame-center",
                "genspark",
                "Genspark (visible Genspark mark detected)",
                Box::new(engines::genspark::engine()),
            )),
            jimeng_pill,
        ];
        Ok(Self { marks })
    }

    /// The process-wide shipped registry.
    pub fn standard() -> Result<&'static Self> {
        static REGISTRY: OnceLock<std::result::Result<Registry, String>> = OnceLock::new();
        REGISTRY
            .get_or_init(|| Self::build().map_err(|e| e.to_string()))
            .as_ref()
            .map_err(|e| VisibleError::Unsupported { op: "registry", reason: e.clone() })
    }

    pub fn known_marks(&self) -> &[KnownMark] {
        &self.marks
    }

    /// Keys of all registered marks (the CLI's `--mark` choices).
    pub fn mark_keys(&self) -> Vec<&'static str> {
        self.marks.iter().map(|m| m.key).collect()
    }

    pub fn get_mark(&self, key: &str) -> Result<&KnownMark> {
        self.marks.iter().find(|m| m.key == key).ok_or_else(|| VisibleError::UnknownMark(key.to_owned()))
    }

    /// `(key, platform)` for every mark that names a platform, in registry order.
    pub fn visible_mark_platforms(&self) -> Vec<(&'static str, &'static str)> {
        self.marks.iter().filter_map(|m| m.platform.map(|p| (m.key, p))).collect()
    }

    /// The registry row a TC260 `ContentProducer` identity names.
    pub fn tc260_producer_mark(&self, code: &str) -> Option<&KnownMark> {
        let mark = raiw_core::tc260::producer_for_code(code)?.image_mark?;
        self.get_mark(mark).ok()
    }

    fn product_of(&self, key: &str) -> Option<&'static str> {
        self.marks.iter().find(|m| m.key == key).map(|m| m.product)
    }

    fn provenance_confirms_product(&self, product: &str, provenance: &BTreeSet<String>) -> bool {
        provenance.iter().any(|k| self.product_of(k) == Some(product))
    }

    /// Detect every scanned mark; `provenance` relaxes the gate of every mark of a confirmed product.
    pub fn detect_marks(
        &self,
        image: &Image,
        include_explicit: bool,
        provenance: &BTreeSet<String>,
    ) -> Result<Vec<MarkDetection>> {
        self.marks
            .iter()
            .filter(|m| include_explicit || m.in_auto)
            .map(|m| m.detect(image, self.provenance_confirms_product(m.product, provenance)))
            .collect()
    }

    /// The trust level mark `key`'s gate resolves to; only same-product evidence confirms.
    pub fn resolve_trust(
        &self,
        key: &str,
        sensitivity: Sensitivity,
        provenance: &BTreeSet<String>,
        strict_keys: &BTreeSet<&str>,
    ) -> Trust {
        if sensitivity == Sensitivity::Strict {
            return Trust::Strict;
        }
        let Some(product) = self.product_of(key) else { return Trust::Strict };
        let corroborated = strict_keys.iter().any(|k| {
            *k != key
                && self.marks.iter().any(|m| m.key == *k && m.can_corroborate && m.product == product)
        });
        if self.provenance_confirms_product(product, provenance) || corroborated {
            Trust::Confirmed
        } else {
            Trust::Strict
        }
    }

    /// Other products from the pill's manufacturer, whose detection vetoes it.
    pub fn pill_suppressors(&self) -> BTreeSet<&'static str> {
        let Ok(pill) = self.get_mark("jimeng_pill") else { return BTreeSet::new() };
        self.marks
            .iter()
            .filter(|m| m.manufacturer == pill.manufacturer && m.product != pill.product)
            .map(|m| m.key)
            .collect()
    }

    /// Whether to auto-remove the Jimeng `AI生成` pill or verified label.
    pub fn keep_pill(
        &self,
        keys: &BTreeSet<&str>,
        provenance: &BTreeSet<String>,
        footprint_flat: bool,
        verified_label: bool,
    ) -> bool {
        if keys.contains("liblib_pill") || self.pill_suppressors().iter().any(|k| keys.contains(k)) {
            return false;
        }
        if keys.contains("jimeng") {
            return true;
        }
        if provenance.contains("jimeng") {
            return footprint_flat || verified_label;
        }
        false
    }

    /// Perception: every `in_auto` mark's verdict at both trust levels from one scan.
    pub fn build_candidates(&self, image: &Image) -> Result<Vec<Candidate>> {
        let mut cands = Vec::with_capacity(self.marks.len());
        for m in self.marks.iter().filter(|m| m.in_auto) {
            let (strict, relaxed) = m.detect_both(image)?;
            let mut features =
                if strict.detected || relaxed.detected { m.features(image)? } else { BTreeMap::new() };
            if relaxed.engine.as_ref().is_some_and(EngineDetection::verified_label) {
                features.insert("verified_label", 1.0);
            }
            cands.push(Candidate {
                key: m.key,
                label: m.label,
                detected_strict: strict.detected,
                detected_relaxed: relaxed.detected,
                features,
                strict_detection: Some(strict),
                relaxed_detection: Some(relaxed),
            });
        }
        Ok(cands)
    }

    /// The pure removal arbiter: perception plus context to the ordered marks to remove.
    pub fn decide(&self, candidates: &[Candidate], context: &Context) -> Vec<Decision> {
        let strict_keys: BTreeSet<&str> = candidates
            .iter()
            .filter(|c| c.detected_strict && !context.vetoed_marks.contains(c.key))
            .map(|c| c.key)
            .collect();
        let mut fired: Vec<Decision> = candidates
            .iter()
            .filter(|c| !context.vetoed_marks.contains(c.key))
            .filter_map(|c| {
                let trust = self.resolve_trust(c.key, context.sensitivity, &context.provenance, &strict_keys);
                let relax = trust != Trust::Strict;
                let ok = if relax { c.detected_relaxed } else { c.detected_strict };
                ok.then(|| Decision { candidate: c.clone(), relax })
            })
            .collect();
        let keys: BTreeSet<&str> = fired.iter().map(|d| d.candidate.key).collect();
        if let Some(pill) = fired.iter().find(|d| d.candidate.key == "jimeng_pill") {
            let feature = |name: &str| pill.candidate.features.get(name).copied().unwrap_or(0.0) != 0.0;
            if !self.keep_pill(&keys, &context.provenance, feature("footprint_flat"), feature("verified_label")) {
                fired.retain(|d| d.candidate.key != "jimeng_pill");
            }
        }
        fired
    }

    /// Fill every decided mark once, then validate each result without editing it.
    pub fn remove_auto_marks_detailed(
        &self,
        image: &Image,
        context: &Context,
        backend: Backend,
        fill: &dyn Fill,
    ) -> Result<VisibleRemovalResult> {
        let mut result = image.clone();
        let mut removals: Vec<MarkRemovalResult> = Vec::new();
        let mut resolved: Option<FillKind> = None;
        for d in self.decide(&self.build_candidates(image)?, context) {
            let mark = self.get_mark(d.candidate.key)?;
            let cached = if removals.is_empty() {
                if d.relax { d.candidate.relaxed_detection.as_ref() } else { d.candidate.strict_detection.as_ref() }
            } else {
                None
            };
            let localization = mark.localize(&result, d.relax, false, cached)?;
            let Some(mask) = localization.mask else { continue };
            let bbox = bounding_rect(&mask);
            if bbox.w <= 0 || bbox.h <= 0 {
                continue;
            }
            let kind = *resolved.get_or_insert_with(|| fill.resolve(backend));
            result = fill.fill(&result, &mask, kind)?;
            let (status, confidence_after, residual_region) = match mark.detect(&result, d.relax) {
                Ok(after) if after.detected && after.region.overlaps(&bbox) => {
                    (MarkValidationStatus::Partial, Some(after.confidence), Some(after.region))
                }
                Ok(after) => (MarkValidationStatus::Cleaned, Some(after.confidence), None),
                Err(err) => {
                    tracing::warn!(
                        "Post-removal validation failed for {} after {} fill in mask {:?}: {err}",
                        mark.key,
                        kind.as_str(),
                        bbox
                    );
                    (MarkValidationStatus::Unvalidated, None, None)
                }
            };
            removals.push(MarkRemovalResult {
                key: mark.key,
                label: mark.label,
                location: mark.location,
                region: localization.region,
                mask_bbox: bbox,
                backend: kind,
                confidence_before: localization.confidence,
                confidence_after,
                status,
                residual_region,
            });
        }
        Ok(VisibleRemovalResult { image: result, marks: removals })
    }

    /// Compatibility wrapper: the filled image and the removed labels.
    pub fn remove_auto_marks(
        &self,
        image: &Image,
        context: &Context,
        backend: Backend,
        fill: &dyn Fill,
    ) -> Result<(Image, Vec<&'static str>)> {
        let report = self.remove_auto_marks_detailed(image, context, backend, fill)?;
        let labels = report.labels();
        Ok((report.image, labels))
    }
}

/// The Gemini engine is shared process-wide; exposing it keeps `identify` on one instance.
pub fn gemini_engine() -> Result<&'static GeminiEngine> {
    GeminiEngine::shared()
}
