//! Child-process bookkeeping shared by the decoder and encoder pipes.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::process::{Child, Command, ExitStatus, Stdio};

use crate::error::{AudioError, Result};

/// A running ffmpeg whose stderr is spooled to an anonymous file (never a blocking pipe).
#[derive(Debug)]
pub struct Spawned {
    pub child: Child,
    stderr: File,
}

impl Spawned {
    pub fn spawn(mut command: Command, stdin: Stdio, stdout: Stdio) -> Result<Self> {
        let stderr = tempfile::tempfile()?;
        let child = command
            .stdin(stdin)
            .stdout(stdout)
            .stderr(Stdio::from(stderr.try_clone()?))
            .spawn()?;
        Ok(Self { child, stderr })
    }

    /// `finish_process`: wait, then fail with the captured stderr on a nonzero status.
    pub fn finish(&mut self) -> Result<()> {
        let status = self.child.wait()?;
        if status.success() {
            return Ok(());
        }
        let detail = self.stderr_text();
        Err(AudioError::failed(format!(
            "ffmpeg audio processing failed ({}): {}",
            python_status(status),
            detail.trim()
        )))
    }

    pub fn stderr_text(&mut self) -> String {
        let mut bytes = Vec::new();
        if self.stderr.seek(SeekFrom::Start(0)).is_ok() {
            let _ = self.stderr.read_to_end(&mut bytes);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// `stop_process`: end a still-running child.
    pub fn stop(&mut self) {
        if matches!(self.child.try_wait(), Ok(None)) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

impl Drop for Spawned {
    fn drop(&mut self) {
        self.stop();
    }
}

/// `Popen.returncode`: the exit code, or the negated signal number.
fn python_status(status: ExitStatus) -> i32 {
    if let Some(code) = status.code() {
        return code;
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            return -signal;
        }
    }
    -1
}
