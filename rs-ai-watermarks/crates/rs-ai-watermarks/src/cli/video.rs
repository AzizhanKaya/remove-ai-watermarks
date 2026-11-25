//! `video` command group.

use std::path::PathBuf;

use clap::{Args, Subcommand};

use super::args::{
    AudioInspect, AudioProcessArgs, AudioScheme, Device, Invisible, MessageMethod, MessageScheme,
    MetadataArgs, StripMetadata, TemporalConsistency, Verify, VideoAudioArgs, VideoBatchMode, VideoMark,
    VideoProfile, VideoQuality, VideoRegenArgs, existing_dir, existing_file, optional_unit_interval,
};

/// Process AI watermarks in video files.
#[derive(Debug, Subcommand)]
pub enum VideoCmd {
    #[command(subcommand)]
    Message(MessageCmd),
    Audio(VideoAudioCmd),
    Identify(VideoIdentifyCmd),
    Metadata(VideoMetadataCmd),
    Invisible(VideoInvisibleCmd),
    Visible(VideoVisibleCmd),
    Erase(VideoEraseCmd),
    All(VideoAllCmd),
    Batch(VideoBatchCmd),
}

/// Inspect or disrupt an independently known public-model watermark message.
#[derive(Debug, Subcommand)]
pub enum MessageCmd {
    Identify(MessageIdentifyCmd),
    Detect(MessageDetectCmd),
    Remove(MessageRemoveCmd),
}

/// Compare all video frames with an independently known encoder message.
#[derive(Debug, Args)]
pub struct MessageIdentifyCmd {
    #[arg(value_parser = existing_file)]
    pub source: PathBuf,
    #[arg(long, value_enum, default_value_t = MessageScheme::Videoseal)]
    pub scheme: MessageScheme,
    /// Independent encoder message hex (64 hex digits for videoseal/pixelseal, 256 for chunkyseal).
    #[arg(long, required = true)]
    pub reference: String,
    /// Emit hash-bound known-message evidence as JSON.
    #[arg(long = "json")]
    pub as_json: bool,
}

/// Check a video for blind VideoSeal, PixelSeal, or ChunkySeal presence without a known hex key.
#[derive(Debug, Args)]
pub struct MessageDetectCmd {
    #[arg(value_parser = existing_file)]
    pub source: PathBuf,
    #[arg(long, value_enum, default_value_t = MessageScheme::Videoseal)]
    pub scheme: MessageScheme,
    /// Emit blind presence evidence as JSON.
    #[arg(long = "json")]
    pub as_json: bool,
}

/// Process a known target, verifying the delivered file before publication.
#[derive(Debug, Args)]
pub struct MessageRemoveCmd {
    #[arg(value_parser = existing_file)]
    pub source: PathBuf,
    #[arg(long, value_enum, default_value_t = MessageScheme::Videoseal)]
    pub scheme: MessageScheme,
    /// Independent encoder message hex (64 hex digits for videoseal/pixelseal, 256 for chunkyseal).
    #[arg(long, required = true)]
    pub reference: String,
    /// Output MP4, MOV, M4V, MKV, WebM, AVI, or FLV.
    #[arg(short, long, required = true)]
    pub output: PathBuf,
    #[arg(long, value_enum, default_value_t = MessageMethod::Auto)]
    pub method: MessageMethod,
    /// Explicit VAE noise; requires --method vae.
    #[arg(long = "noise-std", value_parser = optional_unit_interval)]
    pub noise_std: Option<f64>,
    #[arg(long = "long-side", default_value_t = 512, value_parser = clap::value_parser!(u32).range(8..))]
    pub long_side: u32,
    #[arg(long, value_enum, default_value_t = Device::Auto)]
    pub device: Device,
    /// Permit a final unverified or still-matching candidate.
    #[arg(long)]
    pub force: bool,
    #[command(flatten)]
    pub verify: Verify,
    /// Emit the result and delivered-file quality as JSON.
    #[arg(long = "json")]
    pub as_json: bool,
}

/// Process one or all audio tracks while preserving encoded video and untouched tracks.
#[derive(Debug, Args)]
pub struct VideoAudioCmd {
    #[arg(value_parser = existing_file)]
    pub source: PathBuf,
    /// Output MKV/MOV with PCM or MP4 with AAC, default SOURCE_audio_processed.mkv.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = AudioScheme::Audioseal)]
    pub scheme: AudioScheme,
    #[command(flatten)]
    pub process: AudioProcessArgs,
    /// Audio track ordinal, starting at zero.
    #[arg(long, default_value_t = 0)]
    pub stream: u32,
    /// Inspect and process every source-positive audio track instead of one --stream ordinal.
    #[arg(long = "all-streams")]
    pub all_streams: bool,
    /// Audio stream ordinal to include in multi-stream processing (repeatable; implies --all-streams).
    #[arg(long = "target-stream")]
    pub target_streams: Vec<u32>,
    /// Emit the delivered-video report as JSON.
    #[arg(long = "json")]
    pub as_json: bool,
}

/// Identify supported provenance and visible AI marks in video.
#[derive(Debug, Args)]
pub struct VideoIdentifyCmd {
    #[arg(value_parser = existing_file)]
    pub source: PathBuf,
    /// Skip visible-mark detection; inspect metadata only.
    #[arg(long = "no-visible")]
    pub no_visible: bool,
    #[command(flatten)]
    pub audio: AudioInspect,
    /// Emit the report as JSON.
    #[arg(long = "json")]
    pub as_json: bool,
}

/// Check or remove AI metadata without transcoding video streams.
#[derive(Debug, Args)]
pub struct VideoMetadataCmd {
    #[arg(value_parser = existing_file)]
    pub source: PathBuf,
    #[command(flatten)]
    pub mode: MetadataArgs,
    /// Output path (default: <source>_clean with the same container).
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Emit the metadata report or strip result as JSON.
    #[arg(long = "json")]
    pub as_json: bool,
}

/// Regenerate video pixels with an explicitly selected profile.
#[derive(Debug, Args)]
pub struct VideoInvisibleCmd {
    #[arg(value_parser = existing_file)]
    pub source: PathBuf,
    /// Output path (default: <source>_clean with the same container).
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    #[command(flatten)]
    pub regen: VideoRegenArgs,
    /// VAE inference device.
    #[arg(long, value_enum, default_value_t = Device::Auto)]
    pub device: Device,
    /// Output quality; master requires seedance, wan21-synthid, or wan21-seedance.
    #[arg(long, value_enum, default_value_t = VideoQuality::Delivery)]
    pub quality: VideoQuality,
    /// Video-pixel profile. 'auto' routes ByteDance/Seedance and SynthID clips automatically.
    #[arg(long, value_enum, default_value_t = VideoProfile::Auto)]
    pub profile: VideoProfile,
    /// Emit the regeneration result as JSON.
    #[arg(long = "json")]
    pub as_json: bool,
}

/// Remove a temporally stable visible AI wordmark from video.
#[derive(Debug, Args)]
pub struct VideoVisibleCmd {
    #[arg(value_parser = existing_file)]
    pub source: PathBuf,
    /// Output path (default: <source>_clean with the same container).
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Visible AI mark to remove. Auto scans every supported mark.
    #[arg(long, value_enum, default_value_t = VideoMark::Auto)]
    pub mark: VideoMark,
    #[command(flatten)]
    pub temporal_consistency: TemporalConsistency,
    #[command(flatten)]
    pub strip_metadata: StripMetadata,
    /// Detect and trim a trailing full-screen static AI outro bumper.
    #[arg(long = "trim-bumper")]
    pub trim_bumper: bool,
    /// Emit the visible removal result as JSON.
    #[arg(long = "json")]
    pub as_json: bool,
}

/// Erase user-directed (x,y,w,h) region(s) from a video across all or selected frames.
#[derive(Debug, Args)]
pub struct VideoEraseCmd {
    #[arg(value_parser = existing_file)]
    pub source: PathBuf,
    /// x,y,w,h box to erase (repeatable).
    #[arg(long = "region", required = true)]
    pub regions: Vec<String>,
    /// Inclusive 0-based START:END frame range (default: all frames).
    #[arg(long)]
    pub frames: Option<String>,
    /// Track moving watermark/object box(es) across video frames from the starting frame.
    #[arg(long)]
    pub track: bool,
    /// Output path (default: <source>_clean with the same container).
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Grow each box in pixels.
    #[arg(long, default_value_t = 3)]
    pub dilate: u32,
    #[command(flatten)]
    pub temporal_consistency: TemporalConsistency,
    #[command(flatten)]
    pub strip_metadata: StripMetadata,
    /// Emit the result as JSON.
    #[arg(long = "json")]
    pub as_json: bool,
}

/// Remove stable visible marks, AI metadata, and optional pixel/audio marks from video.
#[derive(Debug, Args)]
pub struct VideoAllCmd {
    #[arg(value_parser = existing_file)]
    pub source: PathBuf,
    /// Output path (default: <source>_clean with the same container).
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    #[command(flatten)]
    pub temporal_consistency: TemporalConsistency,
    #[command(flatten)]
    pub invisible: Invisible,
    #[command(flatten)]
    pub regen: VideoRegenArgs,
    #[command(flatten)]
    pub audio: VideoAudioArgs,
    /// Detect and trim a trailing full-screen static AI outro bumper.
    #[arg(long = "trim-bumper")]
    pub trim_bumper: bool,
    /// Emit the complete video cleaning result as JSON.
    #[arg(long = "json")]
    pub as_json: bool,
}

/// Process every supported video in a directory.
#[derive(Debug, Args)]
pub struct VideoBatchCmd {
    #[arg(value_parser = existing_dir)]
    pub directory: PathBuf,
    /// Output directory (default: <directory>_clean).
    #[arg(short = 'o', long = "output-dir")]
    pub output_dir: Option<PathBuf>,
    /// Video processing mode.
    #[arg(long, value_enum, default_value_t = VideoBatchMode::All)]
    pub mode: VideoBatchMode,
    #[command(flatten)]
    pub temporal_consistency: TemporalConsistency,
    #[command(flatten)]
    pub invisible: Invisible,
    #[command(flatten)]
    pub regen: VideoRegenArgs,
    #[command(flatten)]
    pub audio: VideoAudioArgs,
    /// Emit the video batch summary as JSON.
    #[arg(long = "json")]
    pub as_json: bool,
}

impl VideoAudioCmd {
    pub fn process(&self) -> &AudioProcessArgs {
        &self.process
    }
}
