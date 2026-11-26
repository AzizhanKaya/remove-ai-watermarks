//! Child-process plumbing shared by every ffmpeg stage.
//!
//! Diagnostics go to an anonymous temporary file rather than a pipe, so ffmpeg can never block
//! on stderr backpressure while the producer is still writing frames.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};

use crate::error::{Result, VideoError};

const STDERR_LIMIT: u64 = 64 * 1024;
const STDERR_TRUNCATION: &[u8] = b"\n...[ffmpeg stderr truncated]...\n";

/// `_read_bounded_stderr`: the whole file, or a bounded head and tail around a marker.
pub(crate) fn read_bounded_stderr(file: &mut File) -> io::Result<String> {
    let size = file.seek(SeekFrom::End(0))?;
    let mut raw = Vec::new();
    if size <= STDERR_LIMIT {
        file.seek(SeekFrom::Start(0))?;
        file.read_to_end(&mut raw)?;
    } else {
        let payload = STDERR_LIMIT as usize - STDERR_TRUNCATION.len();
        let head = payload / 2;
        let tail = payload - head;
        file.seek(SeekFrom::Start(0))?;
        raw.resize(head, 0);
        file.read_exact(&mut raw)?;
        raw.extend_from_slice(STDERR_TRUNCATION);
        file.seek(SeekFrom::End(-(tail as i64)))?;
        let mut tail_bytes = vec![0; tail];
        file.read_exact(&mut tail_bytes)?;
        raw.extend_from_slice(&tail_bytes);
    }
    Ok(String::from_utf8_lossy(&raw).into_owned())
}

/// The first `limit` characters of trimmed diagnostics, as Python's `stderr.strip()[:500]`.
pub(crate) fn head_chars(text: &str, limit: usize) -> String {
    text.trim().chars().take(limit).collect()
}

/// Run a finite command with stdout discarded and bounded stderr captured.
pub(crate) fn run_with_stderr_file(command: &mut Command) -> Result<(ExitStatus, String)> {
    let mut stderr = tempfile::tempfile()?;
    let status = command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(stderr.try_clone()?)
        .status()?;
    let text = read_bounded_stderr(&mut stderr)?;
    tracing::info!(?command, %status, stderr = %text, "ffmpeg finished");
    Ok((status, text))
}

/// Run a finite command, capturing stdout and stderr in memory.
pub(crate) fn run_captured(command: &mut Command) -> Result<std::process::Output> {
    let output = command.stdin(Stdio::null()).output()?;
    tracing::info!(?command, status = %output.status, "subprocess finished");
    Ok(output)
}

/// `_RawVideoEncoder`: an ffmpeg process reading frames from stdin.
#[derive(Debug)]
pub struct RawVideoEncoder {
    child: Child,
    stdin: Option<ChildStdin>,
    stderr: File,
}

impl RawVideoEncoder {
    /// `start_raw_video_encoder`.
    pub fn start(mut command: Command) -> Result<Self> {
        tracing::info!(?command, "starting ffmpeg video encode");
        let stderr = tempfile::tempfile()?;
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(stderr.try_clone()?)
            .spawn()?;
        let Some(stdin) = child.stdin.take() else {
            let _ = child.kill();
            let _ = child.wait();
            return Err(VideoError::runtime("Could not open ffmpeg input pipe"));
        };
        Ok(Self {
            child,
            stdin: Some(stdin),
            stderr,
        })
    }

    /// The frame pipe; `None` once closed.
    pub fn stdin(&mut self) -> Result<&mut ChildStdin> {
        self.stdin
            .as_mut()
            .ok_or_else(|| VideoError::runtime("ffmpeg input pipe is already closed"))
    }

    /// `write_raw_frame_bytes`.
    pub fn write_frame(&mut self, bytes: &[u8]) -> Result<()> {
        use std::io::Write;
        self.stdin()?
            .write_all(bytes)
            .map_err(|error| self.pipe_error(&error))
    }

    fn pipe_error(&mut self, error: &io::Error) -> VideoError {
        let diagnostics = read_bounded_stderr(&mut self.stderr).unwrap_or_default();
        VideoError::runtime(format!(
            "ffmpeg stopped reading frames ({error}): {}",
            head_chars(&diagnostics, 500)
        ))
    }

    /// `finish_raw_video_encoder`: close the pipe and fail when ffmpeg rejects the encode.
    pub fn finish(mut self, output: &std::path::Path, operation: &str) -> Result<()> {
        drop(self.stdin.take());
        let status = self.child.wait()?;
        let stderr = read_bounded_stderr(&mut self.stderr)?;
        tracing::info!(operation, %status, stderr = %stderr, "ffmpeg encode finished");
        if !status.success() {
            return Err(VideoError::runtime(format!(
                "ffmpeg failed to encode {}: {}",
                output.display(),
                head_chars(&stderr, 500)
            )));
        }
        Ok(())
    }

    /// `abort_raw_video_encoder`: stop an incomplete encode.
    pub fn abort(mut self) {
        self.kill();
    }

    fn kill(&mut self) {
        drop(self.stdin.take());
        if matches!(self.child.try_wait(), Ok(None)) {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
    }
}

impl Drop for RawVideoEncoder {
    fn drop(&mut self) {
        if self.stdin.is_some() || matches!(self.child.try_wait(), Ok(None)) {
            self.kill();
        }
    }
}
