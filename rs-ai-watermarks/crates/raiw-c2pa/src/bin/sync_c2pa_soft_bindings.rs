//! Synchronize the generated C2PA soft-binding registry table.
//!
//! `cargo run -p raiw-c2pa --features sync --bin sync-c2pa-soft-bindings -- [--check]
//! [--source URL|FILE] [--revision SHA] [--output PATH]`

use std::error::Error;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use raiw_c2pa::soft_binding::sync::{
    DEFAULT_REVISION_URL, DEFAULT_SOURCE_URL, PINNED_SOURCE_URL, is_commit_sha, parse_registry, render_module,
};
use raiw_core::sha256_bytes;

type BoxError = Box<dyn Error>;

const DEFAULT_OUTPUT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/soft_binding/generated.rs");
const USER_AGENT: &str = "remove-ai-watermarks-c2pa-registry-sync";

#[derive(Debug)]
struct Args {
    source: String,
    revision: Option<String>,
    output: PathBuf,
    check: bool,
}

fn parse_args() -> Result<Args, BoxError> {
    let mut args = Args {
        source: DEFAULT_SOURCE_URL.into(),
        revision: None,
        output: DEFAULT_OUTPUT.into(),
        check: false,
    };
    let mut raw = std::env::args().skip(1);
    while let Some(flag) = raw.next() {
        let mut value = || raw.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--source" => args.source = value()?,
            "--revision" => args.revision = Some(value()?),
            "--output" => args.output = value()?.into(),
            "--check" => args.check = true,
            "-h" | "--help" => {
                println!(
                    "usage: sync-c2pa-soft-bindings [--source URL|FILE] [--revision SHA] [--output PATH] [--check]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unrecognized argument {other}").into()),
        }
    }
    Ok(args)
}

fn host(url: &str) -> &str {
    url.split_once("://").map_or("", |(_, rest)| {
        rest.split(['/', '?', '#', ':']).next().unwrap_or_default()
    })
}

/// Read a URL or local file; a GitHub token is sent only to `api.github.com`.
fn read_bytes(source: &str) -> Result<Vec<u8>, BoxError> {
    if !(source.starts_with("http://") || source.starts_with("https://")) {
        return Ok(std::fs::read(source)?);
    }
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(30)))
        .build()
        .into();
    let mut request = agent.get(source).header("User-Agent", USER_AGENT);
    let token = std::env::var("GITHUB_TOKEN")
        .or_else(|_| std::env::var("GH_TOKEN"))
        .ok()
        .filter(|t| !t.is_empty());
    if let Some(token) = token.filter(|_| host(source) == "api.github.com") {
        request = request.header("Authorization", format!("Bearer {token}"));
    }
    Ok(request.call()?.body_mut().read_to_vec()?)
}

fn latest_revision() -> Result<String, BoxError> {
    let revisions: serde_json::Value = serde_json::from_slice(&read_bytes(DEFAULT_REVISION_URL)?)?;
    let revision = revisions
        .get(0)
        .and_then(|entry| entry.get("sha"))
        .and_then(serde_json::Value::as_str);
    match revision {
        Some(sha) if is_commit_sha(sha) => Ok(sha.to_owned()),
        Some(_) => Err("GitHub returned an invalid registry revision".into()),
        None => Err("GitHub did not return a registry revision".into()),
    }
}

/// Fetch the official list through its immutable revision, never `main` directly.
fn source_snapshot(source: &str, revision: Option<&str>) -> Result<(Vec<u8>, String), BoxError> {
    if source == DEFAULT_SOURCE_URL {
        let revision = match revision {
            Some(revision) => revision.to_owned(),
            None => latest_revision()?,
        };
        if !is_commit_sha(&revision) {
            return Err("the official source revision must be a 40-character commit SHA".into());
        }
        return Ok((
            read_bytes(&PINNED_SOURCE_URL.replace("{revision}", &revision))?,
            revision,
        ));
    }
    let bytes = read_bytes(source)?;
    let revision = revision.map_or_else(|| format!("sha256:{}", sha256_bytes(&bytes)), str::to_owned);
    Ok((bytes, revision))
}

fn run() -> Result<ExitCode, BoxError> {
    let args = parse_args()?;
    let (bytes, revision) = source_snapshot(&args.source, args.revision.as_deref())?;
    let rows = parse_registry(&serde_json::from_slice(&bytes)?)?;
    let rendered = render_module(&rows, &args.source, &revision);
    if args.check {
        if std::fs::read_to_string(&args.output).ok().as_deref() != Some(rendered.as_str()) {
            eprintln!("ERROR: C2PA soft-binding snapshot is stale; run this tool without --check");
            return Ok(ExitCode::FAILURE);
        }
        eprintln!(
            "INFO: C2PA soft-binding snapshot is current: {} entries at {revision}",
            rows.len()
        );
        return Ok(ExitCode::SUCCESS);
    }
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.output, rendered)?;
    eprintln!(
        "INFO: Updated {} with {} entries at {revision}",
        args.output.display(),
        rows.len()
    );
    Ok(ExitCode::SUCCESS)
}

fn main() -> ExitCode {
    run().unwrap_or_else(|error| {
        eprintln!("ERROR: C2PA soft-binding sync failed: {error}");
        ExitCode::FAILURE
    })
}
