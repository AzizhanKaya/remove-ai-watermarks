//! Frame capture (port of `video_decode.py`).
//!
//! OpenCV's FFmpeg backend converts with swscale's BT.601 matrix and bicubic chroma upsampling,
//! ignoring the stream's color tags. The reader asks ffmpeg for exactly that conversion, so 8-bit
//! frames match `cv2.VideoCapture` byte for byte; 16-bit frames use the same `bgr48le` pipe as the
//! Python `_FFmpeg16BitCapture`.

use std::fs::File;
use std::io::{BufReader, ErrorKind, Read};
use std::marker::PhantomData;
use std::path::Path;
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

use raiw_cv::{Element, Mat};

use crate::error::{Result, VideoError};
use crate::probe::{VideoGeometry, probe_video_geometry};
use crate::process::{head_chars, read_bounded_stderr};
use crate::tools::{Tool, require_video_tool};

static FFMPEG_THREADS: AtomicUsize = AtomicUsize::new(0);

/// Cap the threads every ffmpeg child uses (`0` keeps ffmpeg's automatic choice).
pub fn set_ffmpeg_threads(threads: usize) {
    FFMPEG_THREADS.store(threads, Ordering::Relaxed);
}

/// `-threads N` when a cap is set.
pub(crate) fn thread_args() -> Vec<String> {
    match FFMPEG_THREADS.load(Ordering::Relaxed) {
        0 => Vec::new(),
        n => vec!["-threads".to_owned(), n.to_string()],
    }
}

/// The pixel type of a decoded BGR frame: `u8` (`bgr24`) or `u16` (`bgr48le`).
pub trait FramePixel: Element {
    const PIX_FMT: &'static str;
    const BYTES: usize;
    fn decode(bytes: &[u8]) -> Vec<Self>;
}

impl FramePixel for u8 {
    const PIX_FMT: &'static str = "bgr24";
    const BYTES: usize = 1;
    fn decode(bytes: &[u8]) -> Vec<Self> {
        bytes.to_vec()
    }
}

impl FramePixel for u16 {
    const PIX_FMT: &'static str = "bgr48le";
    const BYTES: usize = 2;
    fn decode(bytes: &[u8]) -> Vec<Self> {
        bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect()
    }
}

/// Little-endian bytes of a frame, as written to an ffmpeg pipe.
pub fn frame_bytes<T: FramePixel>(frame: &Mat<T>) -> std::borrow::Cow<'_, [u8]> {
    if T::BYTES == 1 || cfg!(target_endian = "little") {
        std::borrow::Cow::Borrowed(frame.as_bytes())
    } else {
        let mut out = Vec::with_capacity(frame.data().len() * T::BYTES);
        for value in frame.data() {
            let wide = value.to_f64() as u16;
            out.extend_from_slice(&wide.to_le_bytes());
        }
        std::borrow::Cow::Owned(out)
    }
}

/// A sequential BGR frame reader over an ffmpeg pipe.
#[derive(Debug)]
pub struct VideoReader<T: FramePixel = u8> {
    child: Child,
    stdout: BufReader<ChildStdout>,
    stderr: File,
    geometry: VideoGeometry,
    buffer: Vec<u8>,
    frames_read: usize,
    finished: bool,
    _pixel: PhantomData<T>,
}

/// `open_video_capture`: an 8-bit BGR reader.
pub fn open_video_capture(source: &Path) -> Result<VideoReader<u8>> {
    VideoReader::open(source)
}

/// `open_video_capture_16bit`: a 16-bit BGR48LE reader.
pub fn open_video_capture_16bit(source: &Path) -> Result<VideoReader<u16>> {
    VideoReader::open(source)
}

impl<T: FramePixel> VideoReader<T> {
    /// Probe the stream geometry and start decoding; fails when no video stream is readable.
    pub fn open(source: &Path) -> Result<Self> {
        let geometry = probe_video_geometry(source)?
            .filter(|g| g.width > 0 && g.height > 0)
            .ok_or_else(|| {
                VideoError::runtime(format!("OpenCV could not decode video: {}", source.display()))
            })?;
        Self::spawn(source, geometry)
    }

    fn spawn(source: &Path, geometry: VideoGeometry) -> Result<Self> {
        let ffmpeg = require_video_tool(Tool::Ffmpeg)?;
        let mut command = Command::new(ffmpeg);
        command
            .args(["-nostdin", "-v", "error"])
            .args(thread_args())
            .arg("-i")
            .arg(source);
        command.args(["-map", "0:v:0"]);
        if T::BYTES == 1 {
            command.args(["-vf", "scale=in_color_matrix=bt601:in_range=auto:flags=bicubic"]);
        }
        command.args([
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            T::PIX_FMT,
            "pipe:1",
        ]);
        let stderr = tempfile::tempfile()?;
        let mut child = command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(stderr.try_clone()?)
            .spawn()?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| VideoError::runtime("Could not open ffmpeg output pipe"))?;
        let frame_bytes = geometry.width * geometry.height * 3 * T::BYTES;
        Ok(Self {
            child,
            stdout: BufReader::with_capacity(frame_bytes.clamp(1 << 16, 1 << 24), stdout),
            stderr,
            geometry,
            buffer: vec![0; frame_bytes],
            frames_read: 0,
            finished: false,
            _pixel: PhantomData,
        })
    }

    pub fn geometry(&self) -> VideoGeometry {
        self.geometry
    }

    pub fn width(&self) -> usize {
        self.geometry.width
    }

    pub fn height(&self) -> usize {
        self.geometry.height
    }

    /// `CAP_PROP_FPS`.
    pub fn fps(&self) -> f64 {
        self.geometry.fps
    }

    /// Frames delivered so far.
    pub fn frames_read(&self) -> usize {
        self.frames_read
    }

    /// Fill the internal buffer with the next frame; `false` at the end of the stream.
    fn advance(&mut self) -> Result<bool> {
        if self.finished {
            return Ok(false);
        }
        let mut got = 0;
        while got < self.buffer.len() {
            match self.stdout.read(&mut self.buffer[got..]) {
                Ok(0) => break,
                Ok(n) => got += n,
                Err(error) if error.kind() == ErrorKind::Interrupted => {}
                Err(error) => return Err(error.into()),
            }
        }
        if got == 0 {
            self.finished = true;
            self.reap()?;
            return Ok(false);
        }
        if got != self.buffer.len() {
            self.finished = true;
            let _ = self.child.kill();
            let _ = self.child.wait();
            return Err(VideoError::runtime("Incomplete BGR frame read from ffmpeg"));
        }
        self.frames_read += 1;
        Ok(true)
    }

    fn reap(&mut self) -> Result<()> {
        let status = self.child.wait()?;
        if !status.success() && self.frames_read == 0 {
            let stderr = read_bounded_stderr(&mut self.stderr).unwrap_or_default();
            tracing::warn!(status = %status, stderr = %stderr, "ffmpeg decoded no frames");
        }
        Ok(())
    }

    /// `read()`: the next decoded frame, or `None` at the end.
    pub fn read(&mut self) -> Result<Option<Mat<T>>> {
        if !self.advance()? {
            return Ok(None);
        }
        let data = T::decode(&self.buffer);
        Ok(Some(Mat::from_vec(
            self.geometry.width,
            self.geometry.height,
            3,
            data,
        )?))
    }

    /// `grab()` without `retrieve()`: advance one frame and drop its pixels.
    pub fn grab(&mut self) -> Result<bool> {
        self.advance()
    }

    /// Diagnostics ffmpeg wrote so far, bounded.
    pub fn diagnostics(&mut self) -> String {
        head_chars(&read_bounded_stderr(&mut self.stderr).unwrap_or_default(), 500)
    }
}

impl<T: FramePixel> Iterator for VideoReader<T> {
    type Item = Result<Mat<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.read().transpose()
    }
}

impl<T: FramePixel> Drop for VideoReader<T> {
    fn drop(&mut self) {
        if matches!(self.child.try_wait(), Ok(None)) {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
    }
}

/// `_iter_decoded_bgr_frames`: every decoded frame, up to `max_frames`.
pub fn iter_decoded_bgr_frames(
    source: &Path,
    max_frames: Option<usize>,
) -> Result<impl Iterator<Item = Result<Mat<u8>>>> {
    let reader = open_video_capture(source)?;
    Ok(reader.take(max_frames.unwrap_or(usize::MAX)))
}
