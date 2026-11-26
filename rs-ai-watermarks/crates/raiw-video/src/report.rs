//! Result types of the high-level video API (the `video.py` dataclasses).
//!
//! Field names and order follow the Python dataclasses, so serializing with
//! `raiw_core::to_py_json` reproduces `json.dumps(asdict(result), indent=2, default=str)`.

use std::path::PathBuf;

use raiw_audio::AudioProvenanceReport;
use raiw_metadata::AiMetadata;
use serde::Serialize;

use crate::visible::Region;

/// `VideoMetadataReport`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VideoMetadataReport {
    pub source: PathBuf,
    pub has_ai_metadata: bool,
    pub markers: AiMetadata,
}

/// What happened to the audio streams of an output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamAction {
    CopiedIfPresent,
    NotWritten,
    Processed,
    Absent,
}

/// The audio watermark verdict on an output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioWatermarkState {
    Unverified,
    NotDetected,
    Detected,
}

impl AudioWatermarkState {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unverified => "unverified",
            Self::NotDetected => "not_detected",
            Self::Detected => "detected",
        }
    }
}

/// `VideoAudioStatus`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct VideoAudioStatus {
    pub stream_action: StreamAction,
    pub watermark_status: AudioWatermarkState,
}

impl Default for VideoAudioStatus {
    fn default() -> Self {
        Self {
            stream_action: StreamAction::CopiedIfPresent,
            watermark_status: AudioWatermarkState::Unverified,
        }
    }
}

/// `_VideoAudioBoundary.audio`: explicit status, else copied-if-present or not-written.
pub fn audio_boundary(explicit: Option<VideoAudioStatus>, wrote_output: bool) -> VideoAudioStatus {
    explicit.unwrap_or(VideoAudioStatus {
        stream_action: if wrote_output {
            StreamAction::CopiedIfPresent
        } else {
            StreamAction::NotWritten
        },
        watermark_status: AudioWatermarkState::Unverified,
    })
}

/// `VideoMetadataResult`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VideoMetadataResult {
    pub source: PathBuf,
    pub output: PathBuf,
    pub detected: AiMetadata,
    pub remaining: AiMetadata,
}

/// `VideoProvenanceReport`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VideoProvenanceReport {
    pub source: PathBuf,
    pub is_ai_generated: Option<bool>,
    pub confidence: &'static str,
    pub platform: Option<String>,
    pub visible_mark: Option<String>,
    pub visible_detected_frames: usize,
    pub total_frames: Option<usize>,
    pub has_ai_metadata: bool,
    pub metadata_markers: AiMetadata,
    pub caveats: Vec<String>,
    pub invisible_watermark_hints: Vec<String>,
    pub audio_streams: Vec<AudioProvenanceReport>,
    pub recommended_recipe: Option<String>,
    pub visible_marks: Vec<String>,
}

/// `VideoVisibleResult`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VideoVisibleResult {
    pub source: PathBuf,
    pub output: Option<PathBuf>,
    pub mark: String,
    pub total_frames: usize,
    pub detected_frames: usize,
    pub removed_frames: usize,
    pub remaining_metadata: AiMetadata,
    pub marks: Vec<String>,
    pub trimmed_bumper_frames: usize,
}

/// `VideoEraseResult`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VideoEraseResult {
    pub source: PathBuf,
    pub output: PathBuf,
    pub regions: Vec<Region>,
    pub total_frames: usize,
    pub erased_frames: usize,
    pub frame_range: Option<(usize, usize)>,
    pub backend: String,
    pub remaining_metadata: AiMetadata,
    pub tracked: bool,
}

/// `RegenerationMetrics` from the invisible engine.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RegenerationMetrics {
    pub frames: usize,
    pub fps: f64,
    pub width: usize,
    pub height: usize,
    pub psnr_db: f64,
    pub temporal_residual_ratio: f64,
}

/// `VideoInvisibleResult` and its experimental subclasses (the extra fields are `None` on the
/// plain SynthID result and serialize only when present).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VideoInvisibleResult {
    pub source: PathBuf,
    pub output: PathBuf,
    pub noise_std: f64,
    pub metrics: RegenerationMetrics,
    pub remaining_metadata: AiMetadata,
    pub profile: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub experimental: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strength: Option<f64>,
}

impl VideoInvisibleResult {
    /// `visual_invisible_action`.
    pub const fn visual_invisible_action(&self) -> &'static str {
        "regenerated"
    }
}

/// `VideoAllResult`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VideoAllResult {
    pub source: PathBuf,
    pub output: PathBuf,
    pub visible_mark: Option<String>,
    pub total_frames: usize,
    pub visible_detected_frames: usize,
    pub visible_removed_frames: usize,
    pub detected_metadata: AiMetadata,
    pub remaining_metadata: AiMetadata,
    pub invisible_removed: bool,
    pub invisible_profile: Option<String>,
    pub audio_status: Option<VideoAudioStatus>,
    pub audio_processed_streams: Vec<usize>,
    pub audio_verification_scope: Vec<String>,
    pub audio_quality_passed: Option<bool>,
    pub visible_marks: Vec<String>,
    pub trimmed_bumper_frames: usize,
}

impl VideoAllResult {
    pub const fn visual_invisible_action(&self) -> &'static str {
        if self.invisible_removed {
            "regenerated"
        } else {
            "not_run"
        }
    }

    pub fn audio(&self) -> VideoAudioStatus {
        audio_boundary(self.audio_status, true)
    }
}

/// Batch mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum BatchMode {
    All,
    Visible,
    Metadata,
}

impl BatchMode {
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "all" => Some(Self::All),
            "visible" => Some(Self::Visible),
            "metadata" => Some(Self::Metadata),
            _ => None,
        }
    }
}

/// `VideoBatchItem`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VideoBatchItem {
    pub source: PathBuf,
    pub output: Option<PathBuf>,
    pub mode: BatchMode,
    pub changed: bool,
    pub visible_mark: Option<String>,
    pub invisible_removed: bool,
    pub error: Option<String>,
    pub audio_status: Option<VideoAudioStatus>,
    pub audio_verification_scope: Vec<String>,
    pub audio_quality_passed: Option<bool>,
    pub visible_marks: Vec<String>,
}

impl VideoBatchItem {
    pub fn failed(source: PathBuf, mode: BatchMode, error: String) -> Self {
        Self {
            source,
            output: None,
            mode,
            changed: false,
            visible_mark: None,
            invisible_removed: false,
            error: Some(error),
            audio_status: None,
            audio_verification_scope: Vec::new(),
            audio_quality_passed: None,
            visible_marks: Vec::new(),
        }
    }

    pub const fn visual_invisible_action(&self) -> &'static str {
        if self.invisible_removed {
            "regenerated"
        } else {
            "not_run"
        }
    }

    pub fn audio(&self) -> VideoAudioStatus {
        audio_boundary(self.audio_status, self.output.is_some())
    }
}

/// `VideoBatchResult`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VideoBatchResult {
    pub directory: PathBuf,
    pub output_directory: PathBuf,
    pub items: Vec<VideoBatchItem>,
}

impl VideoBatchResult {
    pub fn processed(&self) -> usize {
        self.items.iter().filter(|i| i.error.is_none()).count()
    }

    pub fn failed(&self) -> usize {
        self.items.iter().filter(|i| i.error.is_some()).count()
    }

    pub fn invisible_removed(&self) -> usize {
        self.items.iter().filter(|i| i.invisible_removed).count()
    }
}
