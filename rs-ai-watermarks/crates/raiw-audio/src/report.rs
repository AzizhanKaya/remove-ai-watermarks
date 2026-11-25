//! Inspection report types (`audio identify --json`) and their verdict rules.

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

use indexmap::IndexMap;
use serde::{Serialize, Serializer};

use crate::decoders::{Decoder, ModelReading};
use crate::io::AudioStreamInfo;

/// Container metadata markers in insertion order (`dict[str, str]`).
pub type Markers = IndexMap<String, String>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WatermarkStatus {
    Detected,
    NotDetected,
    Unverified,
}

impl WatermarkStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Detected => "detected",
            Self::NotDetected => "not_detected",
            Self::Unverified => "unverified",
        }
    }
}

impl fmt::Display for WatermarkStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A decoded channel ordinal or the multichannel mono downmix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Channel {
    Index(usize),
    Mix,
}

impl Serialize for Channel {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Index(i) => serializer.serialize_u64(*i as u64),
            Self::Mix => serializer.serialize_str("mix"),
        }
    }
}

impl fmt::Display for Channel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Index(i) => write!(f, "{i}"),
            Self::Mix => f.write_str("mix"),
        }
    }
}

/// Every channel, then the downmix for multichannel sources.
pub fn channels_of(count: usize) -> Vec<Channel> {
    let mut channels: Vec<Channel> = (0..count).map(Channel::Index).collect();
    if count > 1 {
        channels.push(Channel::Mix);
    }
    channels
}

/// A decoder reading over native, half-open sample-frame bounds.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AudioWindowReading {
    pub start_frame: usize,
    pub end_frame: usize,
    pub reading: Option<ModelReading>,
    pub error: Option<String>,
    pub inapplicable: bool,
}

impl AudioWindowReading {
    pub fn new(start_frame: usize, end_frame: usize, outcome: CheckOutcome) -> Self {
        Self {
            start_frame,
            end_frame,
            reading: outcome.reading,
            error: outcome.error,
            inapplicable: outcome.inapplicable,
        }
    }

    fn detected(&self) -> bool {
        self.reading.as_ref().is_some_and(ModelReading::detected)
    }
}

/// `(reading, error, inapplicable)` from one decoder check.
#[derive(Debug, Clone, PartialEq)]
pub struct CheckOutcome {
    pub reading: Option<ModelReading>,
    pub error: Option<String>,
    pub inapplicable: bool,
}

impl CheckOutcome {
    pub const fn read(reading: ModelReading) -> Self {
        Self {
            reading: Some(reading),
            error: None,
            inapplicable: false,
        }
    }

    pub fn failed(error: impl Into<String>) -> Self {
        Self {
            reading: None,
            error: Some(error.into()),
            inapplicable: false,
        }
    }

    pub fn inapplicable(error: impl Into<String>) -> Self {
        Self {
            reading: None,
            error: Some(error.into()),
            inapplicable: true,
        }
    }
}

/// One compatible-model check, for a channel or the mono downmix.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AudioDetection {
    pub scheme: Decoder,
    pub channel: Channel,
    pub status: WatermarkStatus,
    pub whole_clip: Option<ModelReading>,
    pub windows: Vec<AudioWindowReading>,
    pub whole_clip_error: Option<String>,
    pub decoder_sample_rate: Option<u32>,
    pub whole_segments: Vec<AudioWindowReading>,
    pub inapplicable: bool,
}

impl AudioDetection {
    /// A placeholder for a requested decoder that is not installed.
    pub const fn missing(scheme: Decoder) -> Self {
        Self {
            scheme,
            channel: Channel::Index(0),
            status: WatermarkStatus::Unverified,
            whole_clip: None,
            windows: Vec::new(),
            whole_clip_error: None,
            decoder_sample_rate: None,
            whole_segments: Vec::new(),
            inapplicable: false,
        }
    }

    /// Any execution error outside explicitly inapplicable minimum-window checks.
    pub fn has_execution_error(&self, include_segments: bool) -> bool {
        let failed = |w: &AudioWindowReading| truthy(w.error.as_deref()) && !w.inapplicable;
        !self.inapplicable
            && (truthy(self.whole_clip_error.as_deref())
                || (include_segments && self.whole_segments.iter().any(failed))
                || self.windows.iter().any(failed))
    }
}

/// Python truthiness of an optional message.
fn truthy(message: Option<&str>) -> bool {
    message.is_some_and(|m| !m.is_empty())
}

impl AudioDetection {
    /// Whether any whole-clip, segment or window check carries an error message.
    pub fn any_error(&self) -> bool {
        truthy(self.whole_clip_error.as_deref())
            || self
                .windows
                .iter()
                .chain(&self.whole_segments)
                .any(|w| truthy(w.error.as_deref()))
    }
}

/// Metadata and named decoder evidence, without a general clean verdict.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AudioProvenanceReport {
    pub source: PathBuf,
    pub sha256: String,
    pub stream: AudioStreamInfo,
    pub metadata_markers: Markers,
    pub watermark_status: WatermarkStatus,
    pub detections: Vec<AudioDetection>,
    pub caveats: Vec<String>,
    pub recommended_recipe: Option<String>,
}

/// `_status(whole, windows, whole_segments)`.
pub fn status(
    whole: Option<&ModelReading>,
    windows: &[AudioWindowReading],
    segments: &[AudioWindowReading],
) -> WatermarkStatus {
    if whole.is_some_and(ModelReading::detected)
        || segments.iter().any(AudioWindowReading::detected)
        || windows.iter().any(AudioWindowReading::detected)
    {
        return WatermarkStatus::Detected;
    }
    let whole_verified = if segments.is_empty() {
        whole.is_some()
    } else {
        segments.iter().all(|s| s.reading.is_some())
    };
    if whole_verified && !windows.is_empty() && windows.iter().all(|w| w.reading.is_some()) {
        return WatermarkStatus::NotDetected;
    }
    WatermarkStatus::Unverified
}

/// `_aggregate(detections)`.
pub fn aggregate(detections: &[AudioDetection]) -> WatermarkStatus {
    if detections.iter().any(|d| d.status == WatermarkStatus::Detected) {
        WatermarkStatus::Detected
    } else if !detections.is_empty()
        && detections
            .iter()
            .all(|d| d.status == WatermarkStatus::NotDetected)
    {
        WatermarkStatus::NotDetected
    } else {
        WatermarkStatus::Unverified
    }
}

/// `missing_decoder_reason(report)`: install hints for requested decoders that were absent.
pub fn missing_decoder_reason(report: &AudioProvenanceReport) -> Option<String> {
    let missing: Vec<&str> = report
        .caveats
        .iter()
        .filter(|c| c.contains("decoding is unavailable; install "))
        .map(String::as_str)
        .collect();
    (!missing.is_empty()).then(|| missing.join("; "))
}

/// `verification_failure_reason(report)`.
pub fn verification_failure_reason(report: &AudioProvenanceReport) -> Option<String> {
    if let Some(missing) = missing_decoder_reason(report) {
        return Some(missing);
    }
    report
        .detections
        .iter()
        .any(|d| d.has_execution_error(true))
        .then(|| "One or more audio decoder checks failed during output verification.".to_owned())
}

/// `verification_outcome(before, after)`: `Some(true)` only when every source-positive
/// `(scheme, channel, decoder_sample_rate)` check has complete negative output coverage.
pub fn verification_outcome(before: &AudioProvenanceReport, after: &AudioProvenanceReport) -> Option<bool> {
    if after.watermark_status == WatermarkStatus::Detected
        || after
            .detections
            .iter()
            .any(|d| d.status == WatermarkStatus::Detected)
    {
        return Some(false);
    }
    if before.watermark_status != WatermarkStatus::Detected
        || after.watermark_status != WatermarkStatus::NotDetected
    {
        return None;
    }
    let applicable: Vec<&AudioDetection> = after.detections.iter().filter(|d| !d.inapplicable).collect();
    if applicable.is_empty()
        || applicable
            .iter()
            .any(|d| d.status != WatermarkStatus::NotDetected)
    {
        return None;
    }
    let mut by_key: HashMap<(Decoder, Channel, Option<u32>), Vec<&AudioDetection>> = HashMap::new();
    for det in applicable {
        by_key
            .entry((det.scheme, det.channel, det.decoder_sample_rate))
            .or_default()
            .push(det);
    }
    let positives: Vec<&AudioDetection> = before
        .detections
        .iter()
        .filter(|d| d.status == WatermarkStatus::Detected)
        .collect();
    if positives.is_empty() {
        return None;
    }
    for pos in positives {
        match by_key.get(&(pos.scheme, pos.channel, pos.decoder_sample_rate)) {
            Some(matched) if matched.iter().all(|m| m.status == WatermarkStatus::NotDetected) => {}
            _ => return None,
        }
    }
    Some(true)
}
