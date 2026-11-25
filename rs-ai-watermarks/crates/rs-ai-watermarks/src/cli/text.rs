//! `text` command group.

use std::path::PathBuf;

use clap::{Args, Subcommand};

use super::args::{Invisible, RewriterBackend, existing_file};

/// Process AI provenance wrappers, character steganography, and token watermarks in text files.
#[derive(Debug, Subcommand)]
pub enum TextCmd {
    Identify(TextIdentifyCmd),
    Metadata(TextMetadataCmd),
    Invisible(TextInvisibleCmd),
    All(TextAllCmd),
}

/// Identify C2PA text/HTML wrappers, VSRMark soft bindings, and Unicode steganography in text.
#[derive(Debug, Args)]
pub struct TextIdentifyCmd {
    #[arg(value_parser = existing_file)]
    pub source: PathBuf,
    /// Emit the report as JSON.
    #[arg(long = "json")]
    pub as_json: bool,
}

/// Check or remove C2PA text/HTML wrappers, VSRMark soft bindings, and Unicode steganography.
#[derive(Debug, Args)]
pub struct TextMetadataCmd {
    #[arg(value_parser = existing_file)]
    pub source: PathBuf,
    /// Check for AI text metadata and steganography (don't modify).
    #[arg(long)]
    pub check: bool,
    /// Remove AI text metadata and steganography.
    #[arg(long)]
    pub remove: bool,
    /// Output path (default: <source>_clean.<ext>).
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

/// Span-protected regeneration options shared by `text invisible` and `text all`.
#[derive(Debug, Clone, Args)]
pub struct RewriteArgs {
    #[arg(long, default_value_t = 0)]
    pub seed: i64,
    /// Keep rejected prose unchanged and report partial regeneration.
    #[arg(long = "allow-partial")]
    pub allow_partial: bool,
    /// Checkpoint path or model ID for the selected backend.
    #[arg(long = "rewriter-model")]
    pub rewriter_model: Option<String>,
    /// Local Ollama Chat Completions base URL (default: http://127.0.0.1:11434/v1).
    #[arg(long = "ollama-url")]
    pub ollama_url: Option<String>,
    /// With Ollama, regenerate retained sentences in a second pass.
    #[arg(long = "strict-residual")]
    pub strict_residual: bool,
}

/// Remove statistical token watermarks by regenerating unlocked prose spans.
#[derive(Debug, Args)]
pub struct TextInvisibleCmd {
    #[arg(value_parser = existing_file)]
    pub source: PathBuf,
    /// Output path (default: <source>_clean.<ext>).
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    #[arg(long = "rewriter-backend", value_enum, default_value_t = RewriterBackend::Transformers)]
    pub rewriter_backend: RewriterBackend,
    #[command(flatten)]
    pub rewrite: RewriteArgs,
}

/// Strip deterministic text provenance and optionally regenerate prose spans.
#[derive(Debug, Args)]
pub struct TextAllCmd {
    #[arg(value_parser = existing_file)]
    pub source: PathBuf,
    /// Output path (default: <source>_clean.<ext>).
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    #[command(flatten)]
    pub invisible: Invisible,
    #[command(flatten)]
    pub rewrite: RewriteArgs,
}
