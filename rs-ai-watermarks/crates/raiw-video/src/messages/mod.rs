//! Experimental known-message readouts for public video watermark models (`video_messages.py`,
//! `_internal/video_message_models.py`, `_internal/video_message_regeneration.py`).
//!
//! The expected encoder message must be supplied independently of decoding the input. These
//! APIs do not classify AI origin or detect arbitrary unknown marks.

pub mod scores;

use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use raiw_core::constants::{ISOBMFF_VIDEO_EXTENSIONS, VIDEO_EXTENSIONS};
use raiw_core::fs::suffix_lower;
use raiw_core::{StagedSibling, sha256_file};
use raiw_cv::Mat;
use serde::Serialize;
use serde_json::Value;

use crate::api::VideoRuntime;
use crate::audio::{StreamRow, stream_inventory, verify_stream_packets};
use crate::decode::{VideoReader, frame_bytes};
use crate::encode::{
    MuxOptions, RawVideoOptions, StagedVideoOutput, command_from, mux_encoded_video, raw_video_command,
};
use crate::error::{Result, VideoError};
use crate::invisible::{InvisibleOptions, fit_size, regenerate_video_candidate};
use crate::probe::{
    Rational, probe_video_encode_profile, probe_video_geometry, probe_video_timestamps, run_ffprobe,
};
use crate::process::{RawVideoEncoder, head_chars, run_with_stderr_file};
use crate::quality::{FidelityOptions, measure_video_fidelity};
use crate::sampling::{SampledFrames, SamplingOptions, select_sampled_indices};
use crate::tools::{Tool, require_video_tool};
pub use scores::{MessageAccumulator, MessageWindow, VideoMessageReading, reference_bits};

pub const DEFAULT_VIDEO_MESSAGE_LONG_SIDE: usize = 512;
const CODEC_MIN_PSNR_DB: f64 = 25.0;
const CODEC_MIN_SSIM: f64 = 0.75;
const VAE_MIN_PSNR_DB: f64 = 18.0;
const VAE_MIN_SSIM: f64 = 0.35;
const BATCH_FRAMES: usize = 8;
const BATCH_VALUES: usize = 8_000_000;
const PROCESSING_FPS: f64 = 12.0;

/// A public VideoSeal-family model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageScheme {
    VideoSeal,
    PixelSeal,
    ChunkySeal,
}

impl MessageScheme {
    pub fn parse(scheme: &str) -> Result<Self> {
        match scheme {
            "videoseal" => Ok(Self::VideoSeal),
            "pixelseal" => Ok(Self::PixelSeal),
            "chunkyseal" => Ok(Self::ChunkySeal),
            _ => Err(VideoError::value(
                "Video message scheme must be videoseal, pixelseal, or chunkyseal",
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::VideoSeal => "videoseal",
            Self::PixelSeal => "pixelseal",
            Self::ChunkySeal => "chunkyseal",
        }
    }

    /// `SCHEME_MESSAGE_BITS`.
    pub fn bits(self) -> usize {
        match self {
            Self::VideoSeal | Self::PixelSeal => 256,
            Self::ChunkySeal => 1024,
        }
    }
}

impl fmt::Display for MessageScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One loaded detector (`VideoMessageDecoder`).
pub trait VideoMessageModel: Send + Sync + fmt::Debug {
    fn scheme(&self) -> MessageScheme;
    /// Raw rows `[1 + bits]` (presence first) of RGB `[0, 1]` NHWC frames.
    fn predict(&self, frames: &[f32], count: usize, height: usize, width: usize) -> Result<Vec<Vec<f32>>>;
}

/// `load_detector`: verified public assets, loaded on demand.
pub trait VideoMessageModels: Send + Sync + fmt::Debug {
    fn load(&self, scheme: MessageScheme) -> Result<Arc<dyn VideoMessageModel>>;
}

/// `decode_presence_and_logits`: validated presence and message logits of one batch.
fn presence_and_logits(
    model: &dyn VideoMessageModel,
    batch: &FrameBatch,
) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
    if batch.count == 0
        || !batch
            .values
            .iter()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
    {
        return Err(VideoError::value(
            "Video decoding requires nonempty finite TxHxWx3 RGB frames in [0, 1]",
        ));
    }
    let raw = model.predict(&batch.values, batch.count, batch.height, batch.width)?;
    let bits = model.scheme().bits();
    if raw.len() != batch.count
        || raw
            .iter()
            .any(|row| row.len() != bits + 1 || row.iter().any(|v| !v.is_finite()))
    {
        return Err(VideoError::runtime(
            "Video watermark model returned invalid video logits",
        ));
    }
    Ok(raw.into_iter().map(|mut row| (row.remove(0), row)).unzip())
}

/// One packed batch of RGB `[0, 1]` frames.
struct FrameBatch {
    values: Vec<f32>,
    count: usize,
    width: usize,
    height: usize,
}

fn declared_frames(source: &Path) -> Result<Option<u64>> {
    let raw = run_ffprobe(
        source,
        &["-select_streams", "v:0", "-show_entries", "stream=nb_frames"],
        "declared video frame count",
        "default=nokey=1:noprint_wrappers=1",
    )?;
    Ok(raw
        .and_then(|text| text.trim().parse::<u64>().ok())
        .filter(|n| *n > 0))
}

/// `_video_batches`: decode every frame into bounded batches; reject incomplete streams.
fn for_each_batch(source: &Path, mut visit: impl FnMut(&FrameBatch) -> Result<()>) -> Result<usize> {
    let expected = declared_frames(source)?;
    let mut reader =
        VideoReader::<u8>::open(source).map_err(|_| VideoError::value("Source has no video stream"))?;
    let mut batch = FrameBatch {
        values: Vec::new(),
        count: 0,
        width: 0,
        height: 0,
    };
    let mut total = 0usize;
    while let Some(frame) = reader.read()? {
        let (width, height) = frame.size();
        let size = width * height * 3;
        if batch.count > 0
            && ((width, height) != (batch.width, batch.height)
                || batch.count >= BATCH_FRAMES
                || batch.values.len() + size > BATCH_VALUES)
        {
            visit(&batch)?;
            batch.values.clear();
            batch.count = 0;
        }
        (batch.width, batch.height) = (width, height);
        batch.values.reserve(size);
        for px in frame.data().chunks_exact(3) {
            batch
                .values
                .extend([px[2], px[1], px[0]].map(|v| f32::from(v) / 255.0));
        }
        batch.count += 1;
        total += 1;
    }
    if batch.count > 0 {
        visit(&batch)?;
    }
    if total == 0 || expected.is_some_and(|n| n != total as u64) {
        return Err(VideoError::runtime(
            "Video stream did not decode to its complete declared frame count",
        ));
    }
    Ok(total)
}

fn existing_source(source: &Path) -> Result<PathBuf> {
    let path = std::path::absolute(source)?;
    let path = std::fs::canonicalize(&path).unwrap_or(path);
    if !path.is_file() {
        return Err(VideoError::value("Video source must be an existing file"));
    }
    Ok(path)
}

/// Hash-bound evidence for one known message, without a general origin verdict.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VideoMessageReport {
    pub source: PathBuf,
    pub sha256: String,
    pub scheme: MessageScheme,
    pub reference_hex: String,
    pub status: String,
    pub reading: Option<VideoMessageReading>,
    pub caveats: Vec<String>,
}

fn message_scope(scheme: MessageScheme) -> Vec<String> {
    vec![
        format!(
            "Correlation to the caller's independently known {}-bit encoder message, under the pinned public model.",
            scheme.bits()
        ),
        "AI origin, unknown messages, provider deployments, and complete information erasure remain unverified.".into(),
        "Only the first video stream is inspected. Audio is outside these checks.".into(),
        "The receiver checks every decoded frame, four float64 aggregations, and sliding 12-frame windows.".into(),
    ]
}

const NOT_AVAILABLE: &str = "Local checks require remove-ai-watermarks[video-detect].";

/// `identify_video_message`: compare every frame to an independently known encoder message.
pub fn identify_video_message(
    runtime: &VideoRuntime,
    source: &Path,
    reference_hex: &str,
    scheme: MessageScheme,
    check_message: bool,
) -> Result<VideoMessageReport> {
    let bits = reference_bits(reference_hex, scheme.bits())?;
    let mut caveats = message_scope(scheme);
    let path = existing_source(source)?;
    let digest = sha256_file(&path)?;
    let unverified = |caveats: Vec<String>| VideoMessageReport {
        source: path.clone(),
        sha256: digest.clone(),
        scheme,
        reference_hex: reference_hex.to_lowercase(),
        status: "unverified".into(),
        reading: None,
        caveats,
    };
    if !check_message {
        caveats.push("Message decoding was skipped.".into());
        return Ok(unverified(caveats));
    }
    let Some(models) = runtime.neural.messages.as_ref() else {
        caveats.push(NOT_AVAILABLE.into());
        return Ok(unverified(caveats));
    };
    let model = models.load(scheme)?;
    let mut accumulator = MessageAccumulator::new(bits)?;
    for_each_batch(&path, |batch| {
        let (_, logits) = presence_and_logits(model.as_ref(), batch)?;
        accumulator.update(&logits)
    })?;
    let reading = accumulator.finish()?;
    if sha256_file(&path)? != digest {
        return Err(VideoError::runtime(
            "Video source changed during message inspection",
        ));
    }
    Ok(VideoMessageReport {
        status: if reading.target_matched() {
            "matched"
        } else {
            "not_matched"
        }
        .into(),
        reading: Some(reading),
        caveats,
        ..unverified(Vec::new())
    })
}

/// Blind public-model watermark presence and recovered payload report.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VideoWatermarkPresenceReport {
    pub source: PathBuf,
    pub sha256: String,
    pub scheme: MessageScheme,
    pub status: String,
    pub frames: usize,
    pub mean_presence_prob: Option<f64>,
    pub bit_sign_consistency: Option<f64>,
    pub decoded_hex: Option<String>,
    pub caveats: Vec<String>,
}

fn bits_to_hex(bits: &[bool]) -> String {
    let bytes: Vec<u8> = bits
        .chunks(8)
        .map(|byte| byte.iter().fold(0u8, |acc, b| (acc << 1) | u8::from(*b)))
        .collect();
    hex::encode(bytes)
}

/// numpy's pairwise float64 summation (`np.add.reduce` over a contiguous axis).
fn pairwise_sum(values: &[f64]) -> f64 {
    const BLOCK: usize = 128;
    let n = values.len();
    if n < 8 {
        return values.iter().fold(-0.0, |acc, v| acc + v);
    }
    if n <= BLOCK {
        let mut r: [f64; 8] = std::array::from_fn(|i| values[i]);
        let full = n - n % 8;
        for chunk in values[8..full].chunks_exact(8) {
            for (acc, v) in r.iter_mut().zip(chunk) {
                *acc += v;
            }
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for v in &values[full..] {
            res += v;
        }
        return res;
    }
    let half = n / 2;
    let half = half - half % 8;
    pairwise_sum(&values[..half]) + pairwise_sum(&values[half..])
}

/// `detect_video_watermark_presence`: blind VideoSeal, PixelSeal or ChunkySeal presence.
pub fn detect_video_watermark_presence(
    runtime: &VideoRuntime,
    source: &Path,
    scheme: MessageScheme,
) -> Result<VideoWatermarkPresenceReport> {
    let path = existing_source(source)?;
    let digest = sha256_file(&path)?;
    let bits = scheme.bits();
    let mut caveats = vec![
        format!(
            "Blind {scheme} presence check combines channel-0 probability and {bits}-bit cross-frame stability."
        ),
        "Covers only the pinned public model family; proprietary provider keys or SynthID remain unverified."
            .into(),
    ];
    let Some(models) = runtime.neural.messages.as_ref() else {
        caveats.push(NOT_AVAILABLE.into());
        return Ok(VideoWatermarkPresenceReport {
            source: path,
            sha256: digest,
            scheme,
            status: "unverified".into(),
            frames: 0,
            mean_presence_prob: None,
            bit_sign_consistency: None,
            decoded_hex: None,
            caveats,
        });
    };
    let model = models.load(scheme)?;
    let mut total = 0usize;
    let mut prob_sum = 0.0f64;
    let mut logit_sum = vec![0.0f64; bits];
    let mut sign_sum = vec![0.0f64; bits];
    for_each_batch(&path, |batch| {
        let (presence, message) = presence_and_logits(model.as_ref(), batch)?;
        let probs: Vec<f64> = presence
            .iter()
            .map(|p| 1.0 / (1.0 + (-f64::from(*p).clamp(-40.0, 40.0)).exp()))
            .collect();
        prob_sum += pairwise_sum(&probs);
        let mut batch_logits = vec![0.0f64; bits];
        let mut batch_signs = vec![0.0f64; bits];
        for row in &message {
            for (j, v) in row.iter().enumerate() {
                let v = f64::from(*v);
                batch_logits[j] += v;
                batch_signs[j] += if v >= 0.0 { 1.0 } else { -1.0 };
            }
        }
        for j in 0..bits {
            logit_sum[j] += batch_logits[j];
            sign_sum[j] += batch_signs[j];
        }
        total += batch.count;
        Ok(())
    })?;
    if sha256_file(&path)? != digest {
        return Err(VideoError::runtime(
            "Video source changed during presence inspection",
        ));
    }
    let frames = total.max(1) as f64;
    let mean_prob = prob_sum / frames;
    let consistency =
        pairwise_sum(&sign_sum.iter().map(|s| (s / frames).abs()).collect::<Vec<_>>()) / bits as f64;
    let detected = if total >= 4 {
        mean_prob >= 0.70 && consistency >= 0.65
    } else {
        mean_prob >= 0.85
    };
    let recovered = detected.then(|| bits_to_hex(&logit_sum.iter().map(|v| *v > 0.0).collect::<Vec<_>>()));
    Ok(VideoWatermarkPresenceReport {
        source: path,
        sha256: digest,
        scheme,
        status: if detected { "detected" } else { "not_detected" }.into(),
        frames: total,
        mean_presence_prob: Some(mean_prob),
        bit_sign_consistency: Some(consistency),
        decoded_hex: recovered,
        caveats,
    })
}

/// Options for [`remove_video_message`].
#[derive(Clone, Debug, PartialEq)]
pub struct MessageRemovalOptions {
    pub reference_hex: String,
    pub scheme: MessageScheme,
    /// `auto`, `codec` or `vae`.
    pub method: String,
    pub noise_std: Option<f64>,
    pub long_side: usize,
    pub device: String,
    pub temporal_stabilize: bool,
    pub force: bool,
    pub verify: bool,
}

impl Default for MessageRemovalOptions {
    fn default() -> Self {
        Self {
            reference_hex: String::new(),
            scheme: MessageScheme::VideoSeal,
            method: "auto".into(),
            noise_std: None,
            long_side: DEFAULT_VIDEO_MESSAGE_LONG_SIDE,
            device: "auto".into(),
            temporal_stabilize: false,
            force: false,
            verify: true,
        }
    }
}

/// A saved known-target result, with delivered-file quality and explicit scope.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VideoMessageRemovalResult {
    pub source: PathBuf,
    pub output: Option<PathBuf>,
    pub before: VideoMessageReport,
    pub after: Option<VideoMessageReport>,
    pub verified_removed: Option<bool>,
    pub skipped: bool,
    pub method: Option<String>,
    pub noise_std: Option<f64>,
    pub quality: Option<Value>,
    pub copied_audio_streams: Vec<usize>,
    pub quality_passed: Option<bool>,
}

/// `VideoMessageVerificationIncomplete`: a candidate did not pass the requested checks.
#[derive(Clone, Debug, PartialEq)]
pub struct MessageVerificationIncomplete {
    pub before: VideoMessageReport,
    pub after: VideoMessageReport,
    pub reason: Option<String>,
}

impl fmt::Display for MessageVerificationIncomplete {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Saved video message check is {}; existing output was preserved",
            self.after.status
        )?;
        if let Some(reason) = &self.reason {
            write!(f, ". {reason}")?;
        }
        Ok(())
    }
}

impl std::error::Error for MessageVerificationIncomplete {}

fn incomplete(before: &VideoMessageReport, after: &VideoMessageReport, reason: Option<&str>) -> VideoError {
    VideoError::MessageVerification(Box::new(MessageVerificationIncomplete {
        before: before.clone(),
        after: after.clone(),
        reason: reason.map(str::to_owned),
    }))
}

/// `_quality_passes`: method-specific PSNR/SSIM floors of the delivered file.
fn quality_passes(method: &str, quality: &Value) -> bool {
    let finite = |key: &str| quality.get(key).and_then(Value::as_f64).filter(|v| v.is_finite());
    let Some(ssim) = finite("source_ssim") else {
        return false;
    };
    let codec = method == "codec";
    if quality.get("lossless_pixels").and_then(Value::as_bool) == Some(true) {
        return ssim >= if codec { CODEC_MIN_SSIM } else { VAE_MIN_SSIM };
    }
    let Some(psnr) = finite("source_psnr_db") else {
        return false;
    };
    if codec {
        psnr >= CODEC_MIN_PSNR_DB && ssim >= CODEC_MIN_SSIM
    } else {
        psnr >= VAE_MIN_PSNR_DB && ssim >= VAE_MIN_SSIM
    }
}

/// Validated single-video geometry, audio streams and monotonic timestamps of a source.
struct ProcessingSource {
    geometry: (usize, usize, f64),
    audio: Vec<StreamRow>,
    timestamps: Vec<f64>,
}

fn row_str<'a>(row: &'a StreamRow, key: &str) -> Option<&'a str> {
    row.get(key).and_then(Value::as_str)
}

/// `_processing_source`.
fn processing_source(source: &Path) -> Result<ProcessingSource> {
    let streams = stream_inventory(source)?;
    let videos: Vec<&StreamRow> = streams
        .iter()
        .filter(|s| row_str(s, "codec_type") == Some("video"))
        .collect();
    if videos.len() != 1
        || streams
            .iter()
            .any(|s| !matches!(row_str(s, "codec_type"), Some("video" | "audio")))
    {
        return Err(VideoError::value(
            "Video message processing requires one video stream and optional audio streams",
        ));
    }
    let geometry = probe_video_geometry(source)?
        .filter(|g| g.width > 0 && g.height > 0 && g.fps > 0.0)
        .ok_or_else(|| VideoError::value(format!("Video has no usable geometry: {}", source.display())))?;
    let timestamps = probe_video_timestamps(source)?;
    let tolerance = row_str(videos[0], "time_base")
        .and_then(Rational::parse)
        .map_or(0.0, Rational::to_f64)
        + 0.000_001;
    if timestamps.len() < 2 || timestamps.windows(2).any(|w| w[1] < w[0] - tolerance) {
        return Err(VideoError::value(
            "Video message processing requires at least two monotonic video timestamps",
        ));
    }
    Ok(ProcessingSource {
        geometry: (geometry.width, geometry.height, geometry.fps),
        audio: streams
            .into_iter()
            .filter(|s| row_str(s, "codec_type") == Some("audio"))
            .collect(),
        timestamps,
    })
}

fn audio_language(row: &StreamRow) -> Option<&Value> {
    row.get("tags").and_then(|t| t.get("language"))
}

/// `_restore_audio_metadata`: put back audio language and disposition after the pixel encode.
fn restore_audio_metadata(output: &Path, audio: &[StreamRow]) -> Result<()> {
    if audio.is_empty() {
        return Ok(());
    }
    let staged = StagedSibling::new(output)?;
    let mut command = Command::new(require_video_tool(Tool::Ffmpeg)?);
    command
        .args(["-v", "error", "-y", "-copyts", "-i"])
        .arg(output)
        .args([
            "-map",
            "0",
            "-c",
            "copy",
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
        ])
        .args(["-avoid_negative_ts", "disabled"]);
    for (ordinal, stream) in audio.iter().enumerate() {
        if let Some(language) = audio_language(stream).and_then(Value::as_str) {
            command.args([format!("-metadata:s:a:{ordinal}"), format!("language={language}")]);
        }
        let disposition = stream
            .get("disposition")
            .and_then(Value::as_object)
            .map(|d| {
                d.iter()
                    .filter(|(_, v)| v.as_i64().is_some_and(|n| n != 0) || v.as_bool() == Some(true))
                    .map(|(k, _)| k.as_str())
                    .collect::<Vec<_>>()
                    .join("+")
            })
            .unwrap_or_default();
        let disposition = if disposition.is_empty() {
            "0".to_owned()
        } else {
            disposition
        };
        command.args([format!("-disposition:a:{ordinal}"), disposition]);
    }
    if ISOBMFF_VIDEO_EXTENSIONS.contains(&suffix_lower(output).as_str()) {
        command.args(["-movflags", "+faststart"]);
    }
    command.arg(staged.path());
    let (status, stderr) = run_with_stderr_file(&mut command)?;
    if !status.success() {
        return Err(VideoError::runtime(format!(
            "Could not restore copied audio metadata: {}",
            head_chars(&stderr, 2000)
        )));
    }
    staged.commit()?;
    Ok(())
}

/// `_check_copied_audio`: identical packets, samples, languages and dispositions.
fn check_copied_audio(
    runtime: &VideoRuntime,
    source: &Path,
    output: &Path,
    audio: &[StreamRow],
) -> Result<Vec<usize>> {
    let delivered: Vec<StreamRow> = stream_inventory(output)?
        .into_iter()
        .filter(|s| row_str(s, "codec_type") == Some("audio"))
        .collect();
    if delivered.len() != audio.len() {
        return Err(VideoError::runtime(
            "Video message processing changed the audio stream count",
        ));
    }
    let index = |row: &StreamRow| row.get("index").and_then(Value::as_u64).map_or(0, |v| v as usize);
    if !audio.is_empty() {
        let tools = runtime
            .audio
            .tools
            .as_ref()
            .ok_or_else(|| VideoError::runtime("Audio checks require ffmpeg and ffprobe"))?;
        for (ordinal, (before, after)) in audio.iter().zip(&delivered).enumerate() {
            verify_stream_packets(source, output, index(before), Some(index(after)))?;
            let info = raiw_audio::io::probe_audio(tools, source, ordinal)?;
            let fidelity = raiw_audio::quality::measure_fidelity(tools, source, &info, output, ordinal)?;
            if fidelity.max_abs_error != 0.0 {
                return Err(VideoError::runtime(
                    "Video message processing changed decoded audio samples",
                ));
            }
            if audio_language(before) != audio_language(after) {
                return Err(VideoError::runtime(
                    "Video message processing changed an audio language",
                ));
            }
            if before.get("disposition") != after.get("disposition") {
                return Err(VideoError::runtime(
                    "Video message processing changed an audio disposition",
                ));
            }
        }
    }
    Ok(audio.iter().map(index).collect())
}

/// `encode_video_frames`: encode sampled frames at `fps`, copy audio, omit source metadata.
fn encode_video_frames(frames: SampledFrames<u8>, source: &Path, output: &Path, fps: f64) -> Result<()> {
    let mut frames = frames.peekable();
    let (width, height) = match frames.peek() {
        Some(Ok(first)) => first.size(),
        Some(Err(_)) => return frames.next().map_or(Ok(()), |r| r.map(|_| ())),
        None => {
            return Err(VideoError::value(
                "At least one frame is required for video encoding",
            ));
        }
    };
    let profile = probe_video_encode_profile(source)?;
    let staged = StagedVideoOutput::new(output)?;
    let encoded = staged.encoded_video.path().to_path_buf();
    let raw = RawVideoOptions::new(width, height, fps, 18, &profile);
    let mut encoder = RawVideoEncoder::start(command_from(&raw_video_command(&encoded, &raw)?)?)?;
    let written = frames.try_for_each(|frame| -> Result<()> {
        let frame: Mat<u8> = frame?;
        if frame.size() != (width, height) {
            return Err(VideoError::value("Video frames must have matching dimensions"));
        }
        encoder.write_frame(&frame_bytes(&frame))
    });
    if let Err(error) = written {
        encoder.abort();
        return Err(error);
    }
    encoder.finish(&encoded, "SynthID removal encode")?;
    mux_encoded_video(
        &encoded,
        source,
        staged.final_output.path(),
        &MuxOptions {
            strip_metadata: true,
            ..MuxOptions::default()
        },
    )?;
    staged.commit()?;
    Ok(())
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// `remove_video_message`: process a known public-model target behind a saved-file gate.
///
/// `auto` tries codec-only, a VAE round trip, then fixed latent-noise profiles and publishes
/// the first complete negative; `force` permits only the final failed candidate.
pub fn remove_video_message(
    runtime: &VideoRuntime,
    source: &Path,
    output: &Path,
    options: &MessageRemovalOptions,
) -> Result<VideoMessageRemovalResult> {
    if !options.verify && !options.force {
        return Err(VideoError::value("Disabling verification requires force=True"));
    }
    if !matches!(options.method.as_str(), "auto" | "codec" | "vae") {
        return Err(VideoError::value(
            "Video message method must be auto, codec, or vae",
        ));
    }
    if options
        .noise_std
        .is_some_and(|n| options.method != "vae" || !n.is_finite() || !(0.0..=1.0).contains(&n))
    {
        return Err(VideoError::value(
            "Explicit noise_std requires method='vae' and a finite value from 0 to 1",
        ));
    }
    if options.long_side < 8 || !matches!(options.device.as_str(), "auto" | "cpu" | "cuda" | "mps") {
        return Err(VideoError::value("Video message geometry or device is invalid"));
    }
    let original = std::path::absolute(source)?;
    let original = std::fs::canonicalize(&original).unwrap_or(original);
    let target = output.to_path_buf();
    let same = std::path::absolute(&target)? == original
        || (target.exists()
            && original.exists()
            && std::fs::canonicalize(&target).ok().as_deref() == Some(original.as_path()));
    if same {
        return Err(VideoError::value("Video output must differ from the source"));
    }
    if !VIDEO_EXTENSIONS.contains(&suffix_lower(&target).as_str()) {
        return Err(VideoError::value(
            "Video message outputs must be MP4, MOV, M4V, MKV, WebM, AVI, or FLV",
        ));
    }
    let before = identify_video_message(
        runtime,
        &original,
        &options.reference_hex,
        options.scheme,
        options.verify,
    )?;
    if before.status != "matched" && !options.force {
        return Ok(VideoMessageRemovalResult {
            source: original,
            output: None,
            before,
            after: None,
            verified_removed: None,
            skipped: true,
            method: None,
            noise_std: None,
            quality: None,
            copied_audio_streams: Vec::new(),
            quality_passed: None,
        });
    }
    let clip = processing_source(&original)?;
    let (width, height, source_fps) = clip.geometry;
    let effective_fps = PROCESSING_FPS.min(source_fps);
    let expected_frames = select_sampled_indices(&clip.timestamps, effective_fps, None).len();
    let profiles: Vec<(&str, Option<f64>)> = match options.method.as_str() {
        "auto" => vec![
            ("codec", None),
            ("vae", Some(0.0)),
            ("vae", Some(0.15)),
            ("vae", Some(0.25)),
        ],
        "vae" => vec![("vae", Some(options.noise_std.unwrap_or(0.15)))],
        _ => vec![("codec", None)],
    };
    if let Some(parent) = target.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let last = profiles.len() - 1;
    for (index, (method, noise)) in profiles.into_iter().enumerate() {
        let candidate = MessageCandidate {
            runtime,
            original: &original,
            target: &target,
            options,
            clip: &clip,
            before: &before,
            effective_fps,
            expected_frames,
            final_candidate: index == last,
        };
        match candidate.run(method, noise, (width, height)) {
            Err(VideoError::MessageVerification(_)) if index < last => {}
            outcome => return outcome,
        }
    }
    Err(VideoError::runtime("No video message processing profile ran"))
}

/// One processing profile attempt of [`remove_video_message`].
struct MessageCandidate<'a> {
    runtime: &'a VideoRuntime,
    original: &'a Path,
    target: &'a Path,
    options: &'a MessageRemovalOptions,
    clip: &'a ProcessingSource,
    before: &'a VideoMessageReport,
    effective_fps: f64,
    expected_frames: usize,
    final_candidate: bool,
}

impl MessageCandidate<'_> {
    fn run(
        &self,
        method: &str,
        noise: Option<f64>,
        (width, height): (usize, usize),
    ) -> Result<VideoMessageRemovalResult> {
        let (original, options, clip) = (self.original, self.options, self.clip);
        let staged = StagedSibling::new(self.target)?;
        if method == "codec" {
            let frames = SampledFrames::open(
                original,
                SamplingOptions {
                    source_fps: clip.geometry.2,
                    duration: None,
                    effective_fps: self.effective_fps,
                    size: fit_size(width, height, options.long_side)?,
                    timestamps: Some(clip.timestamps.clone()),
                },
            )?;
            encode_video_frames(frames, original, staged.path(), self.effective_fps)?;
        } else {
            let vae = self.runtime.neural.vae.as_ref().ok_or_else(|| {
                VideoError::Unavailable(
                    "Video VAE processing requires remove-ai-watermarks[video,diffusion]".into(),
                )
            })?;
            let regen = InvisibleOptions {
                profile: "synthid".into(),
                noise_std: noise.unwrap_or(0.0),
                long_side: options.long_side,
                fps: Some(PROCESSING_FPS),
                batch_size: 1,
                seed: 0,
                device: options.device.clone(),
                temporal_stabilize: options.temporal_stabilize,
                ..InvisibleOptions::default()
            };
            regenerate_video_candidate(
                vae.as_ref(),
                original,
                staged.path(),
                &regen,
                None,
                Some(clip.timestamps.clone()),
            )?;
        }
        restore_audio_metadata(staged.path(), &clip.audio)?;
        let after = identify_video_message(
            self.runtime,
            staged.path(),
            &options.reference_hex,
            options.scheme,
            options.verify,
        )?;
        if options.verify && after.status != "not_matched" && (!options.force || !self.final_candidate) {
            return Err(incomplete(self.before, &after, None));
        }
        let copied = check_copied_audio(self.runtime, original, staged.path(), &clip.audio)?;
        let fidelity = measure_video_fidelity(
            original,
            staged.path(),
            &FidelityOptions {
                source_geometry: clip.geometry,
                duration: None,
                preserve_timestamps: false,
                source_timestamps: Some(clip.timestamps.clone()),
            },
        )?;
        if fidelity.frames != self.expected_frames
            || after
                .reading
                .as_ref()
                .is_some_and(|r| r.frames != self.expected_frames)
        {
            return Err(VideoError::runtime(
                "Delivered video frame count does not match the source timeline",
            ));
        }
        let mut quality = serde_json::to_value(&fidelity)?;
        if let Some(map) = quality.as_object_mut() {
            map.insert("file".into(), Value::String(file_name(self.target)));
            if fidelity.source_psnr_db.is_infinite() {
                map.insert("source_psnr_db".into(), Value::Null);
                map.insert("lossless_pixels".into(), Value::Bool(true));
            }
        }
        let quality_ok = quality_passes(method, &quality);
        if !quality_ok && (!options.force || !self.final_candidate) {
            return Err(incomplete(
                self.before,
                &after,
                Some("Delivered video failed the minimum quality gate"),
            ));
        }
        if sha256_file(original)? != self.before.sha256 || sha256_file(staged.path())? != after.sha256 {
            return Err(VideoError::runtime(
                "Video source or candidate changed during processing",
            ));
        }
        let verified = if self.before.status == "matched" && after.status == "not_matched" {
            Some(true)
        } else if after.status == "matched" {
            Some(false)
        } else {
            None
        };
        staged.commit()?;
        Ok(VideoMessageRemovalResult {
            source: original.to_path_buf(),
            output: Some(self.target.to_path_buf()),
            before: self.before.clone(),
            after: Some(VideoMessageReport {
                source: self.target.to_path_buf(),
                ..after
            }),
            verified_removed: verified,
            skipped: false,
            method: Some(method.to_owned()),
            noise_std: noise,
            quality: Some(quality),
            copied_audio_streams: copied,
            quality_passed: Some(quality_ok),
        })
    }
}
