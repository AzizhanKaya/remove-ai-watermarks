//! Command-line interface mirroring the Python `remove-ai-watermarks` Click tree.

pub mod args;
mod audio;
mod image;
mod text;
mod video;

use clap::{CommandFactory, Parser, Subcommand};

pub use audio::AudioCmd;
pub use image::{
    AllCmd, BatchCmd, ClassifySourceCmd, EraseCmd, IdentifyCmd, InvisibleCmd, MetadataCmd, VisibleCmd,
};
pub use text::TextCmd;
pub use video::VideoCmd;

/// Exit code for a hard processing or write failure.
pub const EXIT_FAILURE: i32 = 1;
/// Exit code shared by both no-signal conditions and an empty batch directory.
pub const EXIT_NO_SIGNAL: i32 = 2;

#[derive(Debug, Parser)]
#[command(
    name = "rs-ai-watermarks",
    version,
    about = "Remove visible and invisible AI watermarks, plus metadata provenance marks, from images, video, audio, and text."
)]
pub struct Cli {
    /// Enable verbose logging.
    #[arg(short, long, global = true)]
    pub verbose: bool,

    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Remove ALL watermarks: visible + invisible + metadata.
    All(AllCmd),
    /// Inspect audio watermarks and process supported research marks.
    #[command(subcommand)]
    Audio(AudioCmd),
    /// Process all supported media files (images, videos, audio, text) in a directory.
    Batch(BatchCmd),
    /// Classify an image's likely AI provider and model version from pixels.
    ClassifySource(ClassifySourceCmd),
    /// Erase arbitrary region(s) from an image or video via inpainting.
    Erase(EraseCmd),
    /// Identify where an image was made and what watermarks it carries.
    Identify(IdentifyCmd),
    /// Attempt to disrupt invisible AI watermarks through pixel regeneration.
    Invisible(InvisibleCmd),
    /// Run a Model Context Protocol server on stdin/stdout.
    Mcp,
    /// Check or remove AI-generation metadata (images, video, and audio).
    Metadata(MetadataCmd),
    /// Process AI provenance wrappers, character steganography, and token watermarks in text files.
    #[command(subcommand)]
    Text(TextCmd),
    /// Process AI watermarks in video files.
    #[command(subcommand)]
    Video(VideoCmd),
    /// Remove a known visible AI watermark from an image.
    Visible(VisibleCmd),
    /// Integration plumbing: print the operation catalog or execute an API-only operation.
    #[command(name = "__integration", hide = true, subcommand)]
    Integration(IntegrationCmd),
}

#[derive(Debug, Subcommand)]
pub enum IntegrationCmd {
    /// Print the operation catalog as JSON.
    Catalog,
    /// Execute one operation with JSON arguments on stdin.
    Execute { operation: String },
}

/// Every CLI leaf (and API-only backend) as an integration operation.
pub fn operation_catalog() -> Vec<raiw_mcp::Operation> {
    raiw_mcp::operation_catalog(&Cli::command(), args::TOGGLES, args::BOUNDS)
}

/// The MCP server bound to this binary.
pub fn mcp_server(runner: Box<dyn raiw_mcp::Runner>) -> raiw_mcp::Server {
    raiw_mcp::Server::new(operation_catalog(), runner)
}

/// Parse arguments, run the selected command and return the process exit code.
pub fn run() -> i32 {
    let cli = Cli::parse();
    init_logging(cli.verbose);
    let Some(command) = cli.command else {
        let _ = Cli::command().print_help();
        return 0;
    };
    match execute(command) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("Error: {err:#}");
            EXIT_FAILURE
        }
    }
}

fn execute(command: Command) -> anyhow::Result<i32> {
    let name = match command {
        Command::Mcp => return serve_mcp(),
        Command::Integration(action) => return integration(action),
        Command::All(_) => "all",
        Command::Audio(_) => "audio",
        Command::Batch(_) => "batch",
        Command::ClassifySource(_) => "classify-source",
        Command::Erase(_) => "erase",
        Command::Identify(_) => "identify",
        Command::Invisible(_) => "invisible",
        Command::Metadata(_) => "metadata",
        Command::Text(_) => "text",
        Command::Video(_) => "video",
        Command::Visible(_) => "visible",
    };
    anyhow::bail!("`{name}` is not wired into this build yet")
}

fn serve_mcp() -> anyhow::Result<i32> {
    let server = mcp_server(Box::new(raiw_mcp::ProcessRunner::current()?));
    server.serve(std::io::stdin().lock(), std::io::stdout().lock())?;
    Ok(0)
}

fn integration(action: IntegrationCmd) -> anyhow::Result<i32> {
    use raiw_core::{JsonStyle, to_py_json};
    let catalog = operation_catalog();
    match action {
        IntegrationCmd::Catalog => {
            println!("{}", to_py_json(&catalog, JsonStyle::COMPACT));
            Ok(0)
        }
        IntegrationCmd::Execute { operation } => {
            let Some(operation) = catalog.iter().find(|op| op.id == operation) else {
                anyhow::bail!("Unknown operation");
            };
            let arguments: serde_json::Value = serde_json::from_reader(std::io::stdin().lock())?;
            operation.validate(&arguments)?;
            match operation.id.as_str() {
                "api_draft_text" => {
                    anyhow::bail!("Missing backend: OCR draft proposals are not available in this build")
                }
                other => anyhow::bail!("Unsupported API operation: {other}"),
            }
        }
    }
}

fn init_logging(verbose: bool) {
    let level = if verbose { "debug" } else { "warn" };
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| level.into()),
        )
        .with_writer(std::io::stderr)
        .try_init();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_tree_is_consistent() {
        Cli::command().debug_assert();
    }

    #[test]
    fn toggles_follow_last_spelling() {
        let cli = Cli::try_parse_from([
            "raiw",
            "visible",
            "Cargo.toml",
            "--keep-metadata",
            "--strip-metadata",
        ]);
        let Some(Command::Visible(cmd)) = cli.expect("parses").command else {
            panic!("visible")
        };
        assert!(cmd.strip_metadata.get());
        let cli = Cli::try_parse_from(["raiw", "visible", "Cargo.toml", "--keep-metadata"]).unwrap();
        let Some(Command::Visible(cmd)) = cli.command else {
            panic!("visible")
        };
        assert!(!cmd.strip_metadata.get());
    }

    #[test]
    fn directories_are_rejected_for_single_file_commands() {
        let err = Cli::try_parse_from(["raiw", "identify", "."]).unwrap_err();
        assert_eq!(err.exit_code(), 2);
    }
}
