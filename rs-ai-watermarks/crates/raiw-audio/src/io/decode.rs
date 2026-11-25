//! `read_audio`: interleaved float32 blocks streamed from an ffmpeg child.

use std::io::Read;
use std::path::Path;
use std::process::{ChildStdout, Command, Stdio};

use super::process::Spawned;
use super::{AudioStreamInfo, FfmpegTools, bounded_chunk};
use crate::error::{AudioError, Result};
use crate::frames::Frames;

/// Native-rate float32 frames of one stream; dropping the reader stops the decoder.
#[derive(Debug)]
pub struct AudioReader {
    process: Spawned,
    stdout: Option<ChildStdout>,
    channels: usize,
    chunk_frames: usize,
    total: usize,
    done: bool,
}

impl AudioReader {
    pub fn open(
        tools: &FfmpegTools,
        source: &Path,
        info: &AudioStreamInfo,
        chunk_frames: usize,
    ) -> Result<Self> {
        let chunk_frames = bounded_chunk(info, chunk_frames)?;
        let source = std::fs::canonicalize(source).unwrap_or_else(|_| source.to_path_buf());
        let mut command = Command::new(&tools.ffmpeg);
        command
            .args(["-nostdin", "-v", "error", "-i"])
            .arg(&source)
            .args(["-map", &format!("0:{}", info.index), "-vn", "-sn", "-dn"])
            .args([
                "-ac",
                &info.channels.to_string(),
                "-ar",
                &info.sample_rate.to_string(),
            ])
            .args(["-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"]);
        let mut process = Spawned::spawn(command, Stdio::null(), Stdio::piped())?;
        let Some(stdout) = process.child.stdout.take() else {
            process.stop();
            return Err(AudioError::failed("Audio decoder did not open its output pipe"));
        };
        Ok(Self {
            process,
            stdout: Some(stdout),
            channels: info.channels,
            chunk_frames,
            total: 0,
            done: false,
        })
    }

    /// Frames decoded so far.
    pub const fn frames_read(&self) -> usize {
        self.total
    }

    fn read_block(&mut self) -> Result<Option<Frames>> {
        let frame_bytes = 4 * self.channels;
        let want = self.chunk_frames * frame_bytes;
        let mut buffer = vec![0u8; want];
        let mut filled = 0;
        if let Some(stdout) = self.stdout.as_mut() {
            while filled < want {
                match stdout.read(&mut buffer[filled..]) {
                    Ok(0) => break,
                    Ok(n) => filled += n,
                    Err(err) if err.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(err) => return Err(err.into()),
                }
            }
        }
        if filled == 0 {
            self.stdout = None;
            self.process.finish()?;
            if self.total == 0 {
                return Err(AudioError::invalid("Audio stream contains no decoded samples"));
            }
            return Ok(None);
        }
        if filled % frame_bytes != 0 {
            return Err(AudioError::failed(
                "Audio decoder returned an incomplete sample frame",
            ));
        }
        buffer.truncate(filled);
        let samples: Vec<f32> = buffer
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let frames = Frames::new(samples, self.channels)?;
        self.total += frames.frames();
        Ok(Some(frames))
    }
}

impl Iterator for AudioReader {
    type Item = Result<Frames>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        match self.read_block() {
            Ok(Some(frames)) => Some(Ok(frames)),
            Ok(None) => {
                self.done = true;
                None
            }
            Err(err) => {
                self.done = true;
                self.stdout = None;
                self.process.stop();
                Some(Err(err))
            }
        }
    }
}

/// Decode a whole stream into memory.
pub fn read_all(tools: &FfmpegTools, source: &Path, info: &AudioStreamInfo) -> Result<Frames> {
    let mut all = Frames::empty(info.channels);
    for block in AudioReader::open(tools, source, info, super::DEFAULT_CHUNK_FRAMES)? {
        all.append(&block?);
    }
    Ok(all)
}
