//! The coarse-to-fine multi-mark scan (`_scan_video_detectors`, `scan_video_marks`).

use std::path::Path;

use raiw_cv::Mat;
use rayon::prelude::*;

use super::detect::{Detector, detector_for};
use super::frame::{FrameLocalization, PreparedFrame};
use super::policy::{coarse_stride, should_promote_coarse_mark};
use crate::decode::open_video_capture;
use crate::error::{Result, VideoError};
use crate::probe::{probe_frame_timestamps, probe_video_timestamps};

const COARSE_MIN_CLIP_FRAMES: usize = 24;
const BATCH: usize = 8;

/// `VideoScan`: decoded geometry plus one candidate per frame.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoScan {
    pub width: usize,
    pub height: usize,
    pub fps: f64,
    pub detections: Vec<FrameLocalization>,
    pub timestamps: Vec<f64>,
}

impl VideoScan {
    pub fn total_frames(&self) -> usize {
        self.detections.len()
    }
}

fn run_detectors(
    batch: &[(usize, Mat<u8>)],
    detectors: &[(&str, Detector)],
) -> Result<Vec<Vec<FrameLocalization>>> {
    batch
        .par_iter()
        .map(|(index, frame)| {
            let mut prepared = PreparedFrame::new(frame)?;
            detectors
                .iter()
                .map(|(_, detect)| detect(frame, *index, &mut prepared))
                .collect()
        })
        .collect()
}

fn decoder_timestamps(source: &Path, start_time: f64, total: usize, fps: f64) -> Result<Vec<f64>> {
    let frames = probe_frame_timestamps(source)?;
    if frames.len() == total {
        return Ok(frames.into_iter().map(|t| t - start_time).collect());
    }
    Ok((0..total).map(|i| i as f64 / fps).collect())
}

/// One mark's scan, or the zero-detector probe under the key `""`.
pub type Scans = Vec<(String, VideoScan)>;

/// `_scan_video_detectors`: collect untrusted candidates with coarse-to-fine pruning.
pub fn scan_video_detectors(source: &Path, marks: &[&str], collect_timestamps: bool) -> Result<Scans> {
    let mut detectors: Vec<(&str, Detector)> = Vec::new();
    let mut unsupported: Vec<&str> = Vec::new();
    for mark in marks {
        match detector_for(mark) {
            Some(detect) => detectors.push((mark, detect)),
            None => unsupported.push(mark),
        }
    }
    if !unsupported.is_empty() {
        unsupported.sort_unstable();
        return Err(VideoError::value(format!(
            "Unsupported visible video mark: {}",
            unsupported.join(", ")
        )));
    }
    let not_decodable =
        || VideoError::runtime(format!("OpenCV could not decode video: {}", source.display()));
    let mut reader = open_video_capture(source).map_err(|_| not_decodable())?;
    let geometry = reader.geometry();
    let (width, height, fps) = (geometry.width, geometry.height, geometry.fps);
    if width == 0 || height == 0 || fps <= 0.0 {
        return Err(VideoError::runtime(format!(
            "Video has invalid stream geometry or frame rate: {}",
            source.display()
        )));
    }
    let probed = if collect_timestamps {
        probe_video_timestamps(source)?
    } else {
        Vec::new()
    };
    let short_clip = !probed.is_empty() && probed.len() < COARSE_MIN_CLIP_FRAMES;
    let use_coarse = !detectors.is_empty() && !short_clip;
    let stride = if use_coarse { coarse_stride() } else { 1 };
    let mut by_mark: Vec<Vec<(usize, FrameLocalization)>> = vec![Vec::new(); detectors.len()];

    let mut total = 0usize;
    let mut batch: Vec<(usize, Mat<u8>)> = Vec::new();
    let flush = |batch: &mut Vec<(usize, Mat<u8>)>,
                 by_mark: &mut Vec<Vec<(usize, FrameLocalization)>>|
     -> Result<()> {
        for ((index, _), results) in batch.iter().zip(run_detectors(batch, &detectors)?) {
            for (slot, result) in results.into_iter().enumerate() {
                by_mark[slot].push((*index, result));
            }
        }
        batch.clear();
        Ok(())
    };
    loop {
        let wanted = !detectors.is_empty() && total.is_multiple_of(stride);
        if !wanted {
            if !reader.grab()? {
                break;
            }
            total += 1;
            continue;
        }
        let Some(frame) = reader.read()? else { break };
        batch.push((total, frame));
        total += 1;
        if batch.len() >= BATCH {
            flush(&mut batch, &mut by_mark)?;
        }
    }
    flush(&mut batch, &mut by_mark)?;
    drop(reader);
    if total == 0 {
        return Err(VideoError::runtime(format!(
            "Video contains no decodable frames: {}",
            source.display()
        )));
    }

    let promoted: Vec<usize> = if !use_coarse {
        Vec::new()
    } else if total < COARSE_MIN_CLIP_FRAMES {
        if total > 1 {
            (0..detectors.len()).collect()
        } else {
            Vec::new()
        }
    } else {
        (0..detectors.len())
            .filter(|&slot| should_promote_coarse_mark(detectors[slot].0, &by_mark[slot]))
            .collect()
    };
    if !promoted.is_empty() {
        let subset: Vec<(&str, Detector)> = promoted.iter().map(|&slot| detectors[slot]).collect();
        let mut reader = open_video_capture(source).map_err(|_| {
            VideoError::runtime(format!("OpenCV could not reopen video: {}", source.display()))
        })?;
        let mut pending: Vec<(usize, Mat<u8>)> = Vec::new();
        let mut flush_promoted = |pending: &mut Vec<(usize, Mat<u8>)>| -> Result<()> {
            for ((index, _), results) in pending.iter().zip(run_detectors(pending, &subset)?) {
                for (position, result) in results.into_iter().enumerate() {
                    by_mark[promoted[position]].push((*index, result));
                }
            }
            pending.clear();
            Ok(())
        };
        for index in 0..total {
            if index % stride == 0 {
                if !reader.grab()? {
                    break;
                }
                continue;
            }
            let Some(frame) = reader.read()? else { break };
            pending.push((index, frame));
            if pending.len() >= BATCH {
                flush_promoted(&mut pending)?;
            }
        }
        flush_promoted(&mut pending)?;
    }

    let timestamps = if !collect_timestamps {
        Vec::new()
    } else if probed.len() == total {
        probed
    } else {
        if !probed.is_empty() {
            tracing::warn!(
                "ffprobe/decoder frame-count mismatch for {}: timestamps={} decoded={total}; using decoder timestamps",
                source.display(),
                probed.len()
            );
        }
        decoder_timestamps(source, geometry.start_time, total, fps)?
    };
    let assemble = |hits: Vec<(usize, FrameLocalization)>| {
        let mut detections: Vec<FrameLocalization> = (0..total).map(FrameLocalization::empty).collect();
        for (index, hit) in hits {
            if index < total {
                detections[index] = hit;
            }
        }
        VideoScan {
            width,
            height,
            fps,
            detections,
            timestamps: timestamps.clone(),
        }
    };
    if detectors.is_empty() {
        return Ok(vec![(String::new(), assemble(Vec::new()))]);
    }
    Ok(detectors
        .iter()
        .map(|(m, _)| (*m).to_owned())
        .zip(by_mark)
        .map(|(m, hits)| (m, assemble(hits)))
        .collect())
}

/// `probe_empty_video_scan`: geometry, frame count and timestamps without detectors.
pub fn probe_empty_video_scan(source: &Path, collect_timestamps: bool) -> Result<VideoScan> {
    let mut scans = scan_video_detectors(source, &[], collect_timestamps)?;
    scans
        .pop()
        .map(|(_, scan)| scan)
        .ok_or_else(|| VideoError::runtime("empty scan"))
}

/// `scan_video_marks`: decode once and collect candidates for every requested mark.
pub fn scan_video_marks(source: &Path, marks: &[&str], collect_timestamps: bool) -> Result<Scans> {
    scan_video_detectors(source, marks, collect_timestamps)
}
