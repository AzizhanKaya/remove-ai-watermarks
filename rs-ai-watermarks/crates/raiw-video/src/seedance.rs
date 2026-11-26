//! The experimental Seedance texture-v1 recipe (`video_seedance.py`) and the timestamped
//! native-geometry encode, delivery transcode, clock repair and scoring it shares with Wan 2.1.
//!
//! A 512 px VAE round trip of every frame is stabilized, color-anchored at native size, and
//! only its sigma-1 low band replaces the source's; the high band of the source is kept.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::Command;

use raiw_core::constants::ISOBMFF_VIDEO_EXTENSIONS;
use raiw_core::fs::suffix_lower;
use raiw_cv::{Border, Interpolation, Mat, gaussian_blur, resize};
use serde_json::Value;

use crate::decode::{VideoReader, frame_bytes};
use crate::encode::{
    MuxOptions, RawVideoOptions, StagedVideoOutput, atomic_video_output, command_from, container_format_args,
    mux_encoded_video, profile_args, raw_video_command, video_codec_args, video_end_repair,
};
use crate::error::{Result, VideoError};
use crate::invisible::{LatentRound, VaePixel, VideoVae, fit_size};
use crate::nut::{NutPixelFormat, TimestampedNutWriter};
use crate::probe::{
    VideoEncodeProfile, probe_video_encode_profile, probe_video_geometry, probe_video_timestamps, run_ffprobe,
};
use crate::process::{RawVideoEncoder, head_chars, run_with_stderr_file};
use crate::report::RegenerationMetrics;
use crate::temporal::VideoPixel;
use crate::temporal::color::anchor_low_freq_color;
use crate::temporal::flow::{backward_motion_map, motion_residual_energy};
use crate::temporal::regen::{RegenStabilizeOptions, RegeneratedStabilizer};
use crate::tools::{Tool, require_video_tool};
use crate::visible::encode::timestamp_time_base;

pub const SEEDANCE_EXPERIMENTAL_PROFILE: &str = "seedance-texture-v1";
pub const SEEDANCE_VAE_MODEL: &str = "stabilityai/sd-vae-ft-mse";
const SEEDANCE_GUIDE_LONG_SIDE: usize = 512;
const SEEDANCE_BATCH: usize = 4;
const HDR_TRANSFERS: [&str; 2] = ["smpte2084", "arib-std-b67"];
const CLOCK_TOLERANCE: f64 = 0.000_002;

/// `delivery` (H.264 High / HEVC CRF 18) or `master` (lossless-coded YUV).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeedanceQuality {
    Delivery,
    Master,
}

impl SeedanceQuality {
    pub fn parse(quality: &str) -> Result<Self> {
        match quality {
            "delivery" => Ok(Self::Delivery),
            "master" => Ok(Self::Master),
            _ => Err(VideoError::value("quality must be delivery or master")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Delivery => "delivery",
            Self::Master => "master",
        }
    }
}

/// Validated geometry and frame clock of an experimental-profile source.
#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentalSource {
    pub width: usize,
    pub height: usize,
    pub fps: f64,
    pub timestamps: Vec<f64>,
}

/// `validate_experimental_mp4_source`: 8-bit SDR (or 10-bit when allowed), even dimensions, timestamps.
pub fn validate_experimental_mp4_source(
    source: &Path,
    label: &str,
    profile: &VideoEncodeProfile,
    allow_high_depth: bool,
) -> Result<ExperimentalSource> {
    let high_depth = allow_high_depth && profile.high_depth();
    let hdr = profile
        .color_transfer
        .as_deref()
        .is_some_and(|t| HDR_TRANSFERS.contains(&t));
    if (profile.component_depth != Some(8) || hdr) && !high_depth {
        return Err(VideoError::value(format!("{label} requires SDR 8-bit video")));
    }
    let geometry = probe_video_geometry(source)?
        .ok_or_else(|| VideoError::value(format!("Could not open video: {}", source.display())))?;
    if geometry.width % 2 != 0 || geometry.height % 2 != 0 {
        return Err(VideoError::value(format!(
            "{label} requires even video dimensions"
        )));
    }
    let timestamps = probe_video_timestamps(source)?;
    if timestamps.is_empty() {
        return Err(VideoError::value(format!(
            "{label} requires source frame timestamps"
        )));
    }
    Ok(ExperimentalSource {
        width: geometry.width,
        height: geometry.height,
        fps: geometry.fps,
        timestamps,
    })
}

/// `_restore_texture`: the fixed sigma-1 split, keeping the source high band at native depth.
pub fn restore_texture<T: VaePixel>(source: &Mat<T>, guide: &Mat<T>) -> Result<Mat<T>> {
    let (width, height) = source.size();
    let original: Mat<f64> = source.convert();
    let reconstructed: Mat<f64> = if guide.size() == (width, height) {
        guide.convert()
    } else {
        resize(guide, width, height, Interpolation::Lanczos4)?.convert()
    };
    let low_original = gaussian_blur(&original, (0, 0), 1.0, 1.0, Border::Reflect101)?;
    let low_guide = gaussian_blur(&reconstructed, (0, 0), 1.0, 1.0, Border::Reflect101)?;
    let data = low_guide
        .data()
        .iter()
        .zip(original.data())
        .zip(low_original.data())
        .map(|((g, o), l)| T::from_f64((g + o - l).round_ties_even().clamp(0.0, T::PEAK)))
        .collect();
    Ok(Mat::from_vec(width, height, 3, data)?)
}

/// `_clock`: the first video stream's clock (and packet clocks when asked) as ffprobe JSON.
fn clock(path: &Path, include_packets: bool) -> Result<Value> {
    let mut entries = "stream=time_base,start_pts,duration_ts,duration".to_owned();
    if include_packets {
        entries.push_str(":packet=pts,dts,duration");
    }
    let raw = run_ffprobe(
        path,
        &["-select_streams", "v:0", "-show_entries", &entries],
        "experimental Seedance video clock",
        "json",
    )?
    .ok_or_else(|| VideoError::runtime(format!("Could not read video clock: {}", path.display())))?;
    Ok(serde_json::from_str(&raw)?)
}

fn clock_duration(clock: &Value) -> Result<f64> {
    clock
        .pointer("/streams/0/duration")
        .and_then(|v| v.as_str().and_then(|s| s.parse().ok()).or_else(|| v.as_f64()))
        .ok_or_else(|| VideoError::runtime("Video stream reports no duration"))
}

fn is_isobmff(path: &Path) -> bool {
    ISOBMFF_VIDEO_EXTENSIONS.contains(&suffix_lower(path).as_str())
}

fn run_command(command: &mut Command) -> Result<()> {
    let (status, stderr) = run_with_stderr_file(command)?;
    if !status.success() {
        return Err(VideoError::runtime(format!(
            "Experimental Seedance encoding failed: {}",
            head_chars(&stderr, 2000)
        )));
    }
    Ok(())
}

/// `_match_video_end`: retime the terminal packet so the stream ends where the source does.
fn match_video_end(encoded: &Path, source: &Path) -> Result<()> {
    let original = clock(source, false)?;
    let target = clock_duration(&original)?;
    if (clock_duration(&clock(encoded, false)?)? - target).abs() <= CLOCK_TOLERANCE {
        return Ok(());
    }
    let repair = video_end_repair(&original, &clock(encoded, true)?)?;
    let ffmpeg = require_video_tool(Tool::Ffmpeg)?;
    let parent = encoded.parent().unwrap_or_else(|| Path::new("."));
    let directory = tempfile::Builder::new()
        .prefix("seedance-clock-")
        .tempdir_in(parent)?;
    let retimed = directory.path().join(format!("retimed{}", suffix_lower(encoded)));
    run_command(
        Command::new(ffmpeg)
            .args(["-v", "error", "-copyts", "-i"])
            .arg(encoded)
            .args(["-map", "0", "-c", "copy", "-bsf:v", &repair.bitstream_filter()])
            .args(["-movie_timescale", &repair.movie_timescale.to_string()])
            .args(["-avoid_negative_ts", "disabled", "-movflags", "+faststart"])
            .args(container_format_args(encoded))
            .arg(&retimed),
    )?;
    if (clock_duration(&clock(&retimed, false)?)? - target).abs() > CLOCK_TOLERANCE {
        return Err(VideoError::runtime("Could not preserve terminal video duration"));
    }
    std::fs::rename(&retimed, encoded)?;
    Ok(())
}

/// The source profile retargeted to the timestamp time base and the delivered pixel format.
fn target_profile(source: &Path, high_depth: bool) -> Result<(VideoEncodeProfile, crate::probe::Rational)> {
    let mut profile = probe_video_encode_profile(source)?;
    let time_base = timestamp_time_base(profile.time_base.as_deref());
    if !(high_depth && profile.high_depth()) {
        profile.pixel_format = "yuv420p".into();
    }
    profile.time_base = Some(time_base.to_string());
    Ok((profile, time_base))
}

/// `_encode_frames`: pipe native frames with their exact source PTS through NUT, then mux.
pub fn encode_timestamped_frames<T: VaePixel>(
    frames: &mut dyn Iterator<Item = Result<Mat<T>>>,
    source: &Path,
    output: &Path,
    fps: f64,
    timestamps: &[f64],
    crf: u32,
) -> Result<()> {
    let first = frames
        .next()
        .ok_or_else(|| VideoError::value("At least one frame is required for video encoding"))??;
    let (width, height) = first.size();
    let high_depth = T::BYTES == 2;
    let (profile, time_base) = target_profile(source, high_depth)?;
    let staged = StagedVideoOutput::new(output)?;
    let encoded = staged.encoded_video.path().to_path_buf();
    let mut raw = RawVideoOptions::new(width, height, fps, crf, &profile);
    raw.input_pix_fmt = T::RAW_PIX_FMT;
    raw.timestamped_input = true;
    raw.copy_input_timestamps = true;
    let mut encoder = RawVideoEncoder::start(command_from(&raw_video_command(&encoded, &raw)?)?)?;
    let start_pts = crate::py::round_int(timestamps.first().copied().unwrap_or(0.0) / time_base.to_f64());
    let pix = if high_depth {
        NutPixelFormat::Bgr48Le
    } else {
        NutPixelFormat::Bgr24
    };
    let written = (|| -> Result<()> {
        let mut writer =
            TimestampedNutWriter::new(encoder.stdin()?, width, height, time_base, start_pts, pix)?;
        let mut stamps = timestamps.iter();
        for frame in std::iter::once(Ok(first)).chain(frames) {
            let frame = frame?;
            let ts = stamps
                .next()
                .ok_or_else(|| VideoError::value("zip() argument 2 is shorter than argument 1"))?;
            if frame.size() != (width, height) {
                return Err(VideoError::value("Video frames must have matching dimensions"));
            }
            writer.write(&frame_bytes(&frame), *ts)?;
        }
        if stamps.next().is_some() {
            return Err(VideoError::value("zip() argument 2 is longer than argument 1"));
        }
        writer.finish()?;
        Ok(())
    })();
    if let Err(error) = written {
        encoder.abort();
        return Err(error);
    }
    encoder.finish(&encoded, "Seedance encode")?;
    let isobmff = is_isobmff(output);
    if isobmff {
        match_video_end(&encoded, source)?;
    }
    mux_encoded_video(
        &encoded,
        source,
        staged.final_output.path(),
        &MuxOptions {
            strip_metadata: true,
            copy_input_timestamps: true,
            ..MuxOptions::default()
        },
    )?;
    if isobmff {
        match_video_end(staged.final_output.path(), source)?;
    }
    staged.commit()?;
    Ok(())
}

/// `_encode_delivery`: transcode the lossless master to CRF 18 with exact timestamps.
fn encode_delivery(master: &Path, source: &Path, output: &Path) -> Result<()> {
    let ffmpeg = require_video_tool(Tool::Ffmpeg)?;
    let (profile, time_base) = target_profile(source, true)?;
    let high_depth = profile.high_depth();
    let suffix = suffix_lower(output);
    let mut codec = video_codec_args(&suffix, 18, &profile, "software", &ffmpeg, None);
    if !high_depth && suffix != ".webm" {
        codec.extend(["-profile:v".to_owned(), "high".to_owned()]);
    }
    let isobmff = is_isobmff(output);
    let mut command = Command::new(&ffmpeg);
    command
        .args(["-y", "-v", "error", "-copyts", "-i"])
        .arg(master)
        .args(["-map", "0:v:0", "-map", "0:a?"])
        .args(&codec)
        .args(profile_args(&profile))
        .args(["-c:a", "copy", "-fps_mode", "passthrough"])
        .args(crate::decode::thread_args());
    if isobmff && time_base.num == 1 {
        command.args(["-video_track_timescale", &time_base.den.to_string()]);
    }
    command.args([
        "-avoid_negative_ts",
        "disabled",
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
    ]);
    if isobmff {
        command.args(["-movflags", "+faststart"]);
    }
    run_command(command.args(container_format_args(output)).arg(output))?;
    if isobmff {
        match_video_end(output, source)?;
    }
    Ok(())
}

/// `_verify_isobmff_timing`: exact timestamps and terminal duration on ISOBMFF outputs.
pub fn verify_isobmff_timing(source: &Path, staged: &Path, timestamps: &[f64], label: &str) -> Result<()> {
    if !is_isobmff(staged) {
        return Ok(());
    }
    if probe_video_timestamps(staged)? != timestamps {
        return Err(VideoError::runtime(format!("{label} output timestamps changed")));
    }
    let staged_duration = clock_duration(&clock(staged, false)?)?;
    let source_duration = clock_duration(&clock(source, false)?)?;
    if (staged_duration - source_duration).abs() > CLOCK_TOLERANCE {
        return Err(VideoError::runtime(format!("{label} output duration changed")));
    }
    Ok(())
}

fn gray_f32<T: VideoPixel>(frame: &Mat<T>) -> Result<Mat<u8>> {
    let narrow: Mat<u8> = if T::BYTES == 1 {
        frame.map(|v| v.to_f64() as u8)
    } else {
        frame.map(|v| ((v.to_f64() as u16) >> 8) as u8)
    };
    Ok(raiw_cv::cvt_color(&narrow, raiw_cv::ColorConversion::BgrToGray)?)
}

/// `_measure`: decode source and delivered file in lockstep for PSNR and gray motion residual.
pub fn measure_native<T: VaePixel>(
    source: &Path,
    output: &Path,
    fps: f64,
    count: usize,
) -> Result<RegenerationMetrics> {
    let geometry = probe_video_geometry(source)?
        .ok_or_else(|| VideoError::value(format!("Could not open video: {}", source.display())))?;
    let mut originals = VideoReader::<T>::open(source)?;
    let mut delivered = VideoReader::<T>::open(output)?;
    let (mut squared_error, mut pixels, mut baseline, mut residual) = (0.0f64, 0usize, 0.0f64, 0.0f64);
    let mut previous: Option<(Mat<u8>, Mat<f32>, Mat<f32>)> = None;
    let mut frames = 0usize;
    loop {
        let (original, candidate) = match (originals.read()?, delivered.read()?) {
            (None, None) => break,
            (Some(a), Some(b)) => (a, b),
            _ => return Err(VideoError::value("zip() arguments have different lengths")),
        };
        if original.size() != candidate.size() {
            return Err(VideoError::runtime(
                "Experimental Seedance output geometry changed",
            ));
        }
        squared_error += original
            .data()
            .iter()
            .zip(candidate.data())
            .map(|(a, b)| {
                let d = a.to_f64() as f32 - b.to_f64() as f32;
                f64::from(d * d)
            })
            .sum::<f64>();
        pixels += original.data().len();
        let current = gray_f32(&original)?;
        let (current_f, current_out): (Mat<f32>, Mat<f32>) =
            (current.convert(), gray_f32(&candidate)?.convert());
        if let Some((pg, pf, po)) = &previous {
            let maps = backward_motion_map(&current, pg)?;
            baseline += motion_residual_energy(&current_f, pf, &maps)?;
            residual += motion_residual_energy(&current_out, po, &maps)?;
        }
        previous = Some((current, current_f, current_out));
        frames += 1;
    }
    if frames != count {
        return Err(VideoError::runtime(
            "Experimental Seedance decoded frame count changed",
        ));
    }
    let mse = squared_error / pixels.max(1) as f64;
    let peak = T::PEAK;
    Ok(RegenerationMetrics {
        frames,
        fps,
        width: geometry.width,
        height: geometry.height,
        psnr_db: if mse > 0.0 {
            10.0 * (peak * peak / mse).log10()
        } else {
            f64::INFINITY
        },
        temporal_residual_ratio: temporal_ratio(residual, baseline),
    })
}

fn temporal_ratio(residual: f64, baseline: f64) -> f64 {
    if baseline > 1e-12 {
        residual / baseline
    } else if residual <= 1e-12 {
        1.0
    } else {
        f64::INFINITY
    }
}

/// The Seedance frame stream: one decode feeds both the guide batches and the native window.
struct SeedanceFrames<'a, T: VaePixel> {
    reader: VideoReader<T>,
    round: LatentRound<'a>,
    guide: (usize, usize),
    native: (usize, usize),
    window: VecDeque<Mat<T>>,
    ready: VecDeque<Mat<T>>,
    stabilizer: Option<RegeneratedStabilizer<T>>,
    done: bool,
}

impl<T: VaePixel> SeedanceFrames<'_, T> {
    fn restore(&mut self, guide: &Mat<T>) -> Result<()> {
        let source = self
            .window
            .pop_front()
            .ok_or_else(|| VideoError::value("zip() argument 2 is longer than argument 1"))?;
        let (w, h) = self.native;
        let up = if guide.size() == (w, h) {
            guide.clone()
        } else {
            resize(guide, w, h, Interpolation::Lanczos4)?
        };
        let anchored = anchor_low_freq_color(&source, &up, None)?;
        self.ready.push_back(restore_texture(&source, &anchored)?);
        Ok(())
    }

    fn fill(&mut self) -> Result<()> {
        let mut batch = Vec::with_capacity(SEEDANCE_BATCH);
        while batch.len() < SEEDANCE_BATCH {
            let Some(frame) = self.reader.read()? else {
                break;
            };
            batch.push(frame);
        }
        if batch.is_empty() {
            self.done = true;
            if let Some(stabilizer) = self.stabilizer.take()
                && let Some((_, guide)) = stabilizer.finish()?
            {
                self.restore(&guide)?;
            }
            return Ok(());
        }
        let (gw, gh) = self.guide;
        let resized = batch
            .iter()
            .map(|f| {
                if f.size() == (gw, gh) {
                    Ok(f.clone())
                } else {
                    resize(f, gw, gh, Interpolation::Lanczos4)
                }
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let guides = self.round.regenerate(&resized)?;
        for (source, guide) in batch.into_iter().zip(guides) {
            let (w, h) = guide.size();
            let small = if source.size() == (w, h) {
                source.clone()
            } else {
                resize(&source, w, h, Interpolation::Area)?
            };
            self.window.push_back(source);
            let stabilizer = self
                .stabilizer
                .as_mut()
                .ok_or_else(|| VideoError::runtime("Seedance stabilizer already finished"))?;
            if let Some((_, stabilized)) = stabilizer.push(small, guide)? {
                self.restore(&stabilized)?;
            }
        }
        Ok(())
    }
}

impl<T: VaePixel> Iterator for SeedanceFrames<'_, T> {
    type Item = Result<Mat<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.ready.is_empty() && !self.done {
            if let Err(error) = self.fill() {
                self.done = true;
                return Some(Err(error));
            }
        }
        self.ready.pop_front().map(Ok)
    }
}

/// Encode a lossless master, write the delivery or master output, verify the clock and score.
#[allow(clippy::too_many_arguments)]
pub(crate) fn finalize_experimental_output<T: VaePixel>(
    frames: &mut dyn Iterator<Item = Result<Mat<T>>>,
    source: &Path,
    master: &Path,
    staged: &Path,
    quality: SeedanceQuality,
    clip: &ExperimentalSource,
    label: &str,
) -> Result<RegenerationMetrics> {
    encode_timestamped_frames(frames, source, master, clip.fps, &clip.timestamps, 0)?;
    match quality {
        SeedanceQuality::Delivery => encode_delivery(master, source, staged)?,
        SeedanceQuality::Master => std::fs::rename(master, staged)?,
    }
    verify_isobmff_timing(source, staged, &clip.timestamps, label)?;
    measure_native::<T>(source, staged, clip.fps, clip.timestamps.len())
}

/// `regenerate_seedance_experimental`: the texture-transfer recipe on 8-bit SDR or 10-bit video.
pub fn regenerate_seedance_experimental(
    vae: &dyn VideoVae,
    source: &Path,
    output: &Path,
    device: &str,
    quality: SeedanceQuality,
) -> Result<RegenerationMetrics> {
    if !matches!(device, "auto" | "cuda" | "mps" | "cpu") {
        return Err(VideoError::value("device must be auto, cuda, mps, or cpu"));
    }
    if vae.model() != SEEDANCE_VAE_MODEL || vae.requested_device() != device {
        return Err(VideoError::value(
            "The supplied video VAE runtime does not match the Seedance model and device",
        ));
    }
    let profile = probe_video_encode_profile(source)?;
    let clip = validate_experimental_mp4_source(source, "Seedance processing", &profile, true)?;
    tracing::info!(
        "Applying {SEEDANCE_EXPERIMENTAL_PROFILE} ({}, {}); no per-file CapCut verification is performed",
        quality.as_str(),
        profile.pixel_format
    );
    if profile.high_depth() {
        run_seedance::<u16>(vae, source, output, quality, &clip)
    } else {
        run_seedance::<u8>(vae, source, output, quality, &clip)
    }
}

fn run_seedance<T: VaePixel>(
    vae: &dyn VideoVae,
    source: &Path,
    output: &Path,
    quality: SeedanceQuality,
    clip: &ExperimentalSource,
) -> Result<RegenerationMetrics> {
    let staged = atomic_video_output(output)?;
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let directory = tempfile::Builder::new()
        .prefix("seedance-experimental-")
        .tempdir_in(parent)?;
    let master: PathBuf = directory.path().join(format!("master{}", suffix_lower(source)));
    let guide = fit_size(clip.width, clip.height, SEEDANCE_GUIDE_LONG_SIDE)?;
    vae.set_tiling(false);
    let mut frames = SeedanceFrames::<T> {
        reader: VideoReader::open(source)?,
        round: LatentRound {
            vae,
            noise_std: 0.0,
            seed: 0,
            noise: None,
        },
        guide,
        native: (clip.width, clip.height),
        window: VecDeque::new(),
        ready: VecDeque::new(),
        stabilizer: Some(RegeneratedStabilizer::new(RegenStabilizeOptions::default())?),
        done: false,
    };
    let metrics = finalize_experimental_output(
        &mut frames,
        source,
        &master,
        staged.path(),
        quality,
        clip,
        "Seedance",
    )?;
    staged.commit()?;
    Ok(metrics)
}
