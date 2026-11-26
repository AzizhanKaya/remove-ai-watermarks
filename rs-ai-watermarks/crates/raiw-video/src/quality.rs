//! Delivered-video fidelity (port of `video_quality.py`).
//!
//! Measurements share the producer's sampling rule. They do not verify a watermark.

use std::path::Path;

use raiw_cv::image_quality::structural_similarity_gray;
use raiw_cv::{ColorConversion, Interpolation, Mat, cvt_color, extract_channel, resize};
use serde::Serialize;

use crate::decode::open_video_capture;
use crate::error::{Result, VideoError};
use crate::probe::{probe_video_duration, probe_video_timestamps};
use crate::py::round_to;
use crate::sampling::{SampledFrames, SamplingOptions};
use crate::temporal::flow::{backward_motion_map, motion_residual_energy};

/// `delivered_geometry`: `(width, height, fps)` as the capture reports them.
pub fn delivered_geometry(path: &Path) -> Result<(usize, usize, f64)> {
    let reader = open_video_capture(path)
        .map_err(|_| VideoError::value(format!("Could not open video: {}", path.display())))?;
    let g = reader.geometry();
    if g.width == 0 || g.height == 0 || g.fps <= 0.0 {
        return Err(VideoError::value(format!(
            "Video has no usable geometry or frame rate: {}",
            path.display()
        )));
    }
    Ok((g.width, g.height, g.fps))
}

/// The `measure_video_fidelity` report, in Python key order.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VideoFidelity {
    pub file: String,
    pub source: String,
    pub source_width: usize,
    pub source_height: usize,
    pub source_fps: f64,
    pub width: usize,
    pub height: usize,
    pub fps: f64,
    pub frames: usize,
    pub max_timestamp_error_seconds: Option<f64>,
    pub video_duration_error_seconds: Option<f64>,
    pub pixel_ratio: f64,
    pub source_psnr_db: f64,
    pub min_frame_psnr_db: f64,
    pub source_ssim: f64,
    pub source_color_ssim: f64,
    pub min_frame_ssim: f64,
    pub worst_window_ssim: f64,
    pub temporal_residual_ratio: f64,
    pub size_bytes: u64,
    pub file_bitrate_kbps: f64,
}

/// Options for [`measure_video_fidelity`].
#[derive(Clone, Debug)]
pub struct FidelityOptions {
    pub source_geometry: (usize, usize, f64),
    pub duration: Option<f64>,
    pub preserve_timestamps: bool,
    pub source_timestamps: Option<Vec<f64>>,
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn psnr_db(mse: f64) -> f64 {
    if mse == 0.0 {
        f64::INFINITY
    } else {
        round_to(20.0 * (255.0 / mse.sqrt()).log10(), 4)
    }
}

fn color_ssim(a: &Mat<u8>, b: &Mat<u8>) -> Result<f64> {
    let mut total = 0.0;
    for channel in 0..3 {
        total += structural_similarity_gray(&extract_channel(a, channel)?, &extract_channel(b, channel)?)?;
    }
    Ok(total / 3.0)
}

/// `measure_video_fidelity`: decode the delivered file and score it against the sampled source.
pub fn measure_video_fidelity(
    source: &Path,
    delivered: &Path,
    options: &FidelityOptions,
) -> Result<VideoFidelity> {
    let (source_width, source_height, source_fps) = options.source_geometry;
    let (width, height, fps) = delivered_geometry(delivered)?;
    let mut timing_error = None;
    let mut duration_error = None;
    let mut reference: Box<dyn Iterator<Item = Result<Mat<u8>>>> = if options.preserve_timestamps {
        if options.duration.is_some() {
            return Err(VideoError::value(
                "Timestamp fidelity requires the complete source",
            ));
        }
        let (source_pts, delivered_pts) = (
            probe_video_timestamps(source)?,
            probe_video_timestamps(delivered)?,
        );
        if source_pts.is_empty() || source_pts.len() != delivered_pts.len() {
            return Err(VideoError::value(
                "Source and delivered timestamps have different frame counts",
            ));
        }
        let error = source_pts
            .iter()
            .zip(&delivered_pts)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        if error > 0.000_002 {
            return Err(VideoError::value(format!(
                "Delivered frame timestamps changed by {error:.6} seconds"
            )));
        }
        timing_error = Some(error);
        let derr = (probe_video_duration(source)? - probe_video_duration(delivered)?).abs();
        if derr > 0.000_002 {
            return Err(VideoError::value(format!(
                "Delivered video duration changed by {derr:.6} seconds"
            )));
        }
        duration_error = Some(derr);
        Box::new(open_video_capture(source)?)
    } else {
        Box::new(SampledFrames::open(
            source,
            SamplingOptions {
                source_fps,
                duration: options.duration,
                effective_fps: fps,
                size: (source_width, source_height),
                timestamps: options.source_timestamps.clone(),
            },
        )?)
    };
    let mut delivered_frames = open_video_capture(delivered)?;

    let mut squared_error = 0.0;
    let mut max_frame_mse: f64 = 0.0;
    let mut pixel_count = 0usize;
    let mut ssim_scores = Vec::new();
    let mut color_scores = Vec::new();
    let (mut temporal_baseline, mut temporal_candidate) = (0.0, 0.0);
    let mut previous: Option<(Mat<u8>, Mat<f32>, Mat<f32>)> = None;
    let mut frame_count = 0usize;
    let needs_upscale = (width, height) != (source_width, source_height);
    loop {
        let reference_frame = reference.next().transpose()?;
        let delivered_frame = delivered_frames.read()?;
        let (reference_frame, delivered_frame) = match (reference_frame, delivered_frame) {
            (None, None) => break,
            (Some(r), Some(d)) => (r, d),
            _ => {
                return Err(VideoError::value(format!(
                    "{} and the sampled source ran out at different points after {frame_count} frames. Pass --duration \
                     to match the prefix this output was produced from, or check that it came from {}.",
                    file_name(delivered),
                    file_name(source)
                )));
            }
        };
        let candidate = if needs_upscale {
            resize(
                &delivered_frame,
                source_width,
                source_height,
                Interpolation::Lanczos4,
            )?
        } else {
            delivered_frame
        };
        let reference_f32: Mat<f32> = reference_frame.convert();
        let candidate_f32: Mat<f32> = candidate.convert();
        let frame_error: f64 = reference_f32
            .data()
            .iter()
            .zip(candidate_f32.data())
            .map(|(a, b)| {
                let d = a - b;
                f64::from(d * d)
            })
            .sum();
        squared_error += frame_error;
        let size = reference_frame.data().len();
        max_frame_mse = max_frame_mse.max(frame_error / size as f64);
        pixel_count += size;
        let current_gray = cvt_color(&reference_frame, ColorConversion::BgrToGray)?;
        let candidate_gray = cvt_color(&candidate, ColorConversion::BgrToGray)?;
        ssim_scores.push(structural_similarity_gray(&current_gray, &candidate_gray)?);
        color_scores.push(color_ssim(&reference_frame, &candidate)?);
        if let Some((previous_gray, previous_reference, previous_candidate)) = &previous {
            let maps = backward_motion_map(&current_gray, previous_gray)?;
            temporal_baseline += motion_residual_energy(&reference_f32, previous_reference, &maps)?;
            temporal_candidate += motion_residual_energy(&candidate_f32, previous_candidate, &maps)?;
        }
        previous = Some((current_gray, reference_f32, candidate_f32));
        frame_count += 1;
    }
    if frame_count < 2 {
        return Err(VideoError::value(format!(
            "{} paired fewer than two frames against {}",
            file_name(delivered),
            file_name(source)
        )));
    }
    let mse = squared_error / pixel_count as f64;
    let window = frame_count.min(12);
    let worst_window = ssim_scores
        .windows(window)
        .map(|w| w.iter().map(|v| v / window as f64).sum::<f64>())
        .fold(f64::INFINITY, f64::min);
    let size_bytes = std::fs::metadata(delivered)?.len();
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    Ok(VideoFidelity {
        file: file_name(delivered),
        source: file_name(source),
        source_width,
        source_height,
        source_fps: round_to(source_fps, 4),
        width,
        height,
        fps: round_to(fps, 4),
        frames: frame_count,
        max_timestamp_error_seconds: timing_error,
        video_duration_error_seconds: duration_error,
        pixel_ratio: round_to((width * height) as f64 / (source_width * source_height) as f64, 4),
        source_psnr_db: psnr_db(mse),
        min_frame_psnr_db: psnr_db(max_frame_mse),
        source_ssim: round_to(mean(&ssim_scores), 4),
        source_color_ssim: round_to(mean(&color_scores), 4),
        min_frame_ssim: round_to(ssim_scores.iter().copied().fold(f64::INFINITY, f64::min), 4),
        worst_window_ssim: round_to(worst_window, 4),
        temporal_residual_ratio: round_to(temporal_candidate / temporal_baseline.max(1e-6), 4),
        size_bytes,
        file_bitrate_kbps: round_to(size_bytes as f64 * 8.0 / (frame_count as f64 / fps) / 1000.0, 1),
    })
}

/// Floors for [`video_quality_passed`].
#[derive(Clone, Copy, Debug)]
pub struct QualityFloors {
    pub min_psnr_db: f64,
    pub min_ssim: f64,
    pub min_color_ssim: f64,
    pub min_frame_ssim: f64,
    pub min_window_ssim: f64,
    pub max_temporal_residual_ratio: f64,
}

impl Default for QualityFloors {
    fn default() -> Self {
        Self {
            min_psnr_db: 20.0,
            min_ssim: 0.75,
            min_color_ssim: 0.70,
            min_frame_ssim: 0.60,
            min_window_ssim: 0.65,
            max_temporal_residual_ratio: 2.5,
        }
    }
}

/// `video_quality_passed`: model-free fidelity floors including chroma, window and worst frame.
pub fn video_quality_passed(quality: Option<&VideoFidelity>, floors: &QualityFloors) -> bool {
    let Some(q) = quality else { return false };
    let psnr_ok = q.source_psnr_db.is_infinite()
        || (q.source_psnr_db.is_finite() && q.source_psnr_db >= floors.min_psnr_db);
    q.source_ssim.is_finite()
        && psnr_ok
        && q.source_ssim >= floors.min_ssim
        && q.source_color_ssim.is_finite()
        && q.source_color_ssim >= floors.min_color_ssim
        && q.min_frame_ssim.is_finite()
        && q.min_frame_ssim >= floors.min_frame_ssim
        && q.worst_window_ssim.is_finite()
        && q.worst_window_ssim >= floors.min_window_ssim
        && q.temporal_residual_ratio.is_finite()
        && q.temporal_residual_ratio <= floors.max_temporal_residual_ratio
}
