//! `audio` command group.

use std::path::PathBuf;

use clap::{Args, Subcommand};

use super::args::{AudioBatchMode, AudioProcessArgs, AudioScheme, Invisible, KeepStandard, MetadataArgs};
use super::args::{existing_dir, existing_file};

/// Inspect audio watermarks and process supported research marks.
#[derive(Debug, Subcommand)]
pub enum AudioCmd {
    Identify(AudioIdentifyCmd),
    Metadata(AudioMetadataCmd),
    Invisible(AudioInvisibleCmd),
    All(AudioAllCmd),
    Batch(AudioBatchCmd),
}

/// Inspect a selected audio track with compatible optional decoders.
#[derive(Debug, Args)]
pub struct AudioIdentifyCmd {
    #[arg(value_parser = existing_file)]
    pub source: PathBuf,
    /// Compatible public models to check; all requests every named model, auto checks installed models.
    #[arg(long, value_enum, default_value_t = AudioScheme::Audioseal)]
    pub scheme: AudioScheme,
    /// Inspect metadata and stream geometry without audio watermark models.
    #[arg(long = "no-invisible")]
    pub no_invisible: bool,
    /// Audio track ordinal, starting at zero.
    #[arg(long, default_value_t = 0)]
    pub stream: u32,
    /// Emit the report as JSON.
    #[arg(long = "json")]
    pub as_json: bool,
}

/// Check or remove AI metadata without re-encoding audio samples.
#[derive(Debug, Args)]
pub struct AudioMetadataCmd {
    #[arg(value_parser = existing_file)]
    pub source: PathBuf,
    #[command(flatten)]
    pub mode: MetadataArgs,
    /// Output audio path (default: <source_stem>_clean<suffix>).
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Emit the report or result as JSON.
    #[arg(long = "json")]
    pub as_json: bool,
}

/// Apply an audio processing profile, keeping verification explicit.
#[derive(Debug, Args)]
pub struct AudioInvisibleCmd {
    #[arg(value_parser = existing_file)]
    pub source: PathBuf,
    /// Output WAV, FLAC, MP3, M4A, OGG, OGA, Opus, AAC, or MKA, default SOURCE_processed.wav.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Compatible public models to check.
    #[arg(long, value_enum, default_value_t = AudioScheme::Audioseal)]
    pub scheme: AudioScheme,
    #[command(flatten)]
    pub process: AudioProcessArgs,
    /// Audio track ordinal, starting at zero.
    #[arg(long, default_value_t = 0)]
    pub stream: u32,
    /// Evict cached neural audio models and empty GPU caches after processing.
    #[arg(long = "low-memory")]
    pub low_memory: bool,
    /// Emit the result as JSON.
    #[arg(long = "json")]
    pub as_json: bool,
}

/// Remove AI metadata and optionally clean compatible audio watermarks.
#[derive(Debug, Args)]
pub struct AudioAllCmd {
    #[arg(value_parser = existing_file)]
    pub source: PathBuf,
    /// Output audio path (default: <source_stem>_clean<suffix>).
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Compatible public models to check.
    #[arg(long, value_enum, default_value_t = AudioScheme::Auto)]
    pub scheme: AudioScheme,
    #[command(flatten)]
    pub invisible: Invisible,
    #[command(flatten)]
    pub process: AudioProcessArgs,
    /// Audio track ordinal, starting at zero.
    #[arg(long, default_value_t = 0)]
    pub stream: u32,
    #[command(flatten)]
    pub keep_standard: KeepStandard,
    /// Evict cached neural audio models and empty GPU caches after processing.
    #[arg(long = "low-memory")]
    pub low_memory: bool,
    /// Emit the result as JSON.
    #[arg(long = "json")]
    pub as_json: bool,
}

/// Process every supported audio file in a directory.
#[derive(Debug, Args)]
pub struct AudioBatchCmd {
    #[arg(value_parser = existing_dir)]
    pub directory: PathBuf,
    /// Output directory (default: <directory>_clean).
    #[arg(short = 'o', long = "output-dir")]
    pub output_dir: Option<PathBuf>,
    /// Audio batch processing mode.
    #[arg(long, value_enum, default_value_t = AudioBatchMode::All)]
    pub mode: AudioBatchMode,
    /// Compatible public models to check.
    #[arg(long, value_enum, default_value_t = AudioScheme::Auto)]
    pub scheme: AudioScheme,
    #[command(flatten)]
    pub invisible: Invisible,
    #[command(flatten)]
    pub process: AudioProcessArgs,
    #[command(flatten)]
    pub keep_standard: KeepStandard,
    /// Evict cached neural audio models and empty GPU caches after each file.
    #[arg(long = "low-memory")]
    pub low_memory: bool,
    /// Emit the batch summary as JSON.
    #[arg(long = "json")]
    pub as_json: bool,
}
