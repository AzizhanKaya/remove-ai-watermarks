//! Single-image commands and the mixed-media `batch`.

use std::path::PathBuf;

use clap::Args;

use super::args::{
    AllPreset, AllVendor, BatchMode, Detect, DiffusionArgs, FillBackend, ImageMark, ImageVendor,
    InpaintMethod, KeepStandard, MetadataArgs, Pipeline, Sensitivity, StripMetadata, TextRestorationArgs,
    existing_dir, existing_file,
};

/// Remove a known visible AI watermark from an image.
#[derive(Debug, Args)]
pub struct VisibleCmd {
    #[arg(value_parser = existing_file)]
    pub source: PathBuf,
    /// Output path (default: <source>_clean.<ext>).
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    #[command(flatten)]
    pub detect: Detect,
    /// Which known visible mark to target (auto picks every detected mark).
    #[arg(long, value_enum, default_value_t = ImageMark::Auto)]
    pub mark: ImageMark,
    /// Fill backend for visible-mark removal (auto: LaMa > MI-GAN > cv2).
    #[arg(long, value_enum, default_value_t = FillBackend::Auto)]
    pub backend: FillBackend,
    /// How hard to trust a borderline mark (auto relaxes only with corroboration).
    #[arg(long, value_enum, default_value_t = Sensitivity::Auto)]
    pub sensitivity: Sensitivity,
    #[command(flatten)]
    pub strip_metadata: StripMetadata,
}

/// Erase arbitrary region(s) from an image or video via inpainting.
#[derive(Debug, Args)]
pub struct EraseCmd {
    #[arg(value_parser = existing_file)]
    pub source: PathBuf,
    /// x,y,w,h box to erase (repeatable).
    #[arg(long = "region", required = true)]
    pub regions: Vec<String>,
    /// Inclusive 0-based START:END frame range when erasing a video clip.
    #[arg(long)]
    pub frames: Option<String>,
    /// Track moving watermark/object box(es) across video frames from the starting frame.
    #[arg(long)]
    pub track: bool,
    /// Output path (default: <source>_clean.<ext>).
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Inpaint backend.
    #[arg(long, value_enum, default_value_t = FillBackend::Cv2)]
    pub backend: FillBackend,
    /// cv2 inpaint method.
    #[arg(long = "inpaint-method", value_enum, default_value_t = InpaintMethod::Telea)]
    pub inpaint_method: InpaintMethod,
    /// Grow the box by this many px before inpainting.
    #[arg(long, default_value_t = 3)]
    pub dilate: i64,
    #[command(flatten)]
    pub strip_metadata: StripMetadata,
    /// Emit JSON output when erasing video.
    #[arg(long = "json")]
    pub as_json: bool,
}

/// Attempt to disrupt invisible AI watermarks through pixel regeneration.
#[derive(Debug, Args)]
pub struct InvisibleCmd {
    #[arg(value_parser = existing_file)]
    pub source: PathBuf,
    /// Output path (default: <source>_clean.<ext>).
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Strength cohort for the invisible-removal default; naming it implies the scrub runs.
    #[arg(long, value_enum, default_value_t = ImageVendor::Auto)]
    pub vendor: ImageVendor,
    /// Pipeline profile.
    #[arg(long, value_enum, default_value_t = Pipeline::QwenZimage)]
    pub pipeline: Pipeline,
    #[command(flatten)]
    pub diffusion: DiffusionArgs,
    #[command(flatten)]
    pub text: TextRestorationArgs,
}

/// Check or remove AI-generation metadata (images, video, and audio).
#[derive(Debug, Args)]
pub struct MetadataCmd {
    #[arg(value_parser = existing_file)]
    pub source: PathBuf,
    #[command(flatten)]
    pub mode: MetadataArgs,
    /// Output path (default: overwrite source).
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

/// Identify where an image was made and what watermarks it carries.
#[derive(Debug, Args)]
pub struct IdentifyCmd {
    #[arg(value_parser = existing_file)]
    pub source: PathBuf,
    /// Skip pixel-domain detectors (visible sparkle + invisible watermark); metadata-only.
    #[arg(long = "no-visible")]
    pub no_visible: bool,
    /// Emit the report as JSON instead of a table.
    #[arg(long = "json")]
    pub as_json: bool,
}

/// Classify an image's likely AI provider and model version from pixels.
#[derive(Debug, Args)]
pub struct ClassifySourceCmd {
    #[arg(value_parser = existing_file)]
    pub source: PathBuf,
    /// Emit the source classification as JSON.
    #[arg(long = "json")]
    pub as_json: bool,
}

/// Remove ALL watermarks: visible + invisible + metadata.
#[derive(Debug, Args)]
pub struct AllCmd {
    #[arg(value_parser = existing_file)]
    pub source: PathBuf,
    /// Output path (default: <source>_clean.<ext>).
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Cleaning preset: lossless-fast, balanced, or deep-clean.
    #[arg(long, value_enum)]
    pub preset: Option<AllPreset>,
    /// Known AI generator cohort.
    #[arg(long, value_enum, default_value_t = AllVendor::Auto)]
    pub vendor: AllVendor,
    #[command(flatten)]
    pub diffusion: DiffusionArgs,
    #[command(flatten)]
    pub text: TextRestorationArgs,
}

/// Process all supported media files (images, videos, audio, text) in a directory.
#[derive(Debug, Args)]
pub struct BatchCmd {
    #[arg(value_parser = existing_dir)]
    pub directory: PathBuf,
    /// Output directory (default: <dir>_clean/).
    #[arg(short = 'o', long = "output-dir")]
    pub output_dir: Option<PathBuf>,
    /// Processing mode.
    #[arg(long, value_enum, default_value_t = BatchMode::Visible)]
    pub mode: BatchMode,
    #[command(flatten)]
    pub diffusion: DiffusionArgs,
}

impl MetadataCmd {
    pub const fn keep_standard(&self) -> bool {
        self.mode.keep_standard.get()
    }
}

impl KeepStandard {
    pub const fn keep(self) -> bool {
        self.get()
    }
}
