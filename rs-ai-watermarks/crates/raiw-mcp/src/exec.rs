//! Isolated execution: every operation runs as one child process of the CLI binary.

use std::collections::BTreeMap;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::Serialize;
use serde_json::Value;

use crate::catalog::Operation;
use crate::{Arguments, InvalidArguments};

/// A finished child process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Completed {
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
}

/// Launches CLI commands; tests substitute a fake.
pub trait Runner: Send + Sync {
    fn run(&self, args: &[String], stdin: Option<&str>) -> io::Result<Completed>;
}

/// Runs the real CLI binary with a quiet, wide, colorless terminal.
#[derive(Debug, Clone)]
pub struct ProcessRunner {
    executable: PathBuf,
}

impl ProcessRunner {
    pub fn new(executable: impl Into<PathBuf>) -> Self {
        Self {
            executable: executable.into(),
        }
    }

    /// The currently running binary, which is the CLI itself.
    pub fn current() -> io::Result<Self> {
        std::env::current_exe().map(Self::new)
    }
}

impl Runner for ProcessRunner {
    fn run(&self, args: &[String], stdin: Option<&str>) -> io::Result<Completed> {
        tracing::info!(?args, "running CLI command");
        let mut child = Command::new(&self.executable)
            .args(args)
            .env("COLUMNS", "120")
            .env("NO_COLOR", "1")
            .env("TERM", "dumb")
            .stdin(if stdin.is_some() {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        if let (Some(payload), Some(mut pipe)) = (stdin, child.stdin.take()) {
            pipe.write_all(payload.as_bytes())?;
        }
        let output = child.wait_with_output()?;
        let code = output.status.code().unwrap_or(1);
        tracing::info!(code, "CLI command exited");
        Ok(Completed {
            code,
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }
}

/// Result of [`run_operation`]: `output` is set only when the child actually wrote it.
#[derive(Debug, Clone, Serialize)]
pub struct OperationResult {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub output: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum OperationError {
    #[error(transparent)]
    Invalid(#[from] InvalidArguments),
    #[error(transparent)]
    Io(#[from] io::Error),
}

/// Hidden CLI entry that executes API-only operations from JSON on stdin.
pub const EXECUTE_COMMAND: [&str; 2] = ["__integration", "execute"];

/// Run one catalog operation in an isolated child, redacting the Hugging Face token.
pub fn run_operation(
    operation: &Operation,
    arguments: &Value,
    runner: &dyn Runner,
) -> Result<OperationResult, OperationError> {
    let values = operation.validate(arguments)?;
    let (args, mut output) = operation.plan(&Value::Object(values.clone()))?;
    let before = output_state(output.as_deref());
    let completed = if args.is_empty() {
        let mut execute: Vec<String> = EXECUTE_COMMAND.iter().map(|s| (*s).to_owned()).collect();
        execute.push(operation.id.clone());
        runner.run(
            &execute,
            Some(&serde_json::to_string(&values).unwrap_or_default()),
        )?
    } else {
        runner.run(&args, None)?
    };
    let (mut stdout, mut stderr) = (completed.stdout, completed.stderr);
    if let Some(token) = secret(&values) {
        stdout = stdout.replace(token, "[redacted]");
        stderr = stderr.replace(token, "[redacted]");
    }
    if output_state(output.as_deref()) == before {
        output = None;
    }
    Ok(OperationResult {
        exit_code: completed.code,
        stdout,
        stderr,
        output,
    })
}

pub fn secret(values: &Arguments) -> Option<&str> {
    values
        .get("hf_token")
        .and_then(Value::as_str)
        .filter(|t| !t.is_empty())
}

type FileState = BTreeMap<PathBuf, (u64, i128, u64)>;

fn output_state(output: Option<&str>) -> FileState {
    let mut state = FileState::new();
    if let Some(output) = output {
        collect(Path::new(output), &mut state);
    }
    state
}

fn collect(path: &Path, state: &mut FileState) {
    let Ok(meta) = std::fs::metadata(path) else { return };
    if meta.is_dir() {
        let Ok(entries) = std::fs::read_dir(path) else {
            return;
        };
        for entry in entries.flatten() {
            collect(&entry.path(), state);
        }
    } else if meta.is_file() {
        state.insert(path.to_path_buf(), (inode(&meta), mtime_ns(&meta), meta.len()));
    }
}

#[cfg(unix)]
fn inode(meta: &std::fs::Metadata) -> u64 {
    std::os::unix::fs::MetadataExt::ino(meta)
}

#[cfg(not(unix))]
fn inode(_meta: &std::fs::Metadata) -> u64 {
    0
}

fn mtime_ns(meta: &std::fs::Metadata) -> i128 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos() as i128)
}
