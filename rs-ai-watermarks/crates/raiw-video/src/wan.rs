//! Wan 2.1 3D video-to-video window streaming (`video_wan.py`) behind the [`WanPipeline`] seam.
//!
//! Frames stream through overlapping causal windows (193 frames, 17 overlap by default), each
//! reflection-padded to 16 px and `1 + 4k` frames, regenerated, cropped, and blended into its
//! predecessor with a raised-cosine ramp. Sources above 1080p regenerate a 1080p guide whose
//! low band is transferred onto the native frame.

use std::collections::VecDeque;
use std::fmt;
use std::path::Path;

use raiw_cv::{Interpolation, Mat, resize};

use crate::encode::atomic_video_output;
use crate::error::{Result, VideoError};
use crate::invisible::VaePixel;
use crate::probe::probe_video_encode_profile;
use crate::report::RegenerationMetrics;
use crate::seedance::{
    ExperimentalSource, SeedanceQuality, encode_timestamped_frames, measure_native, restore_texture,
    validate_experimental_mp4_source, verify_isobmff_timing,
};
use crate::temporal::color::anchor_low_freq_color;

pub const WAN21_MODEL_ID: &str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers";
pub const WAN21_MODEL_REVISION: &str = "0fad780a534b6463e45facd96134c9f345acfa5b";
pub const WAN21_INFERENCE_STEPS: u32 = 20;
pub const WAN21_SYNTHID_STRENGTH: f64 = 0.38;
pub const WAN21_SEEDANCE_STRENGTH: f64 = 0.12;
pub const WAN21_SPATIAL_MULTIPLE: usize = 16;
pub const WAN21_DEFAULT_CHUNK_FRAMES: usize = 193;
pub const WAN21_COMPACT_CHUNK_FRAMES: usize = 81;
pub const WAN21_MPS_CHUNK_FRAMES: usize = 33;
pub const WAN21_DEFAULT_OVERLAP_FRAMES: usize = 17;
pub const WAN21_MPS_OVERLAP_FRAMES: usize = 9;
pub const WAN21_NATIVE_MAX_PIXELS: usize = 1920 * 1080;
pub const WAN21_MIN_CUDA_VRAM_BYTES: u64 = 8 * 1024 * 1024 * 1024;
pub const WAN21_HIGH_VRAM_BYTES: u64 = 16 * 1024 * 1024 * 1024;

/// One `WanVideoToVideoPipeline` call (guidance off, `output_type="np"`).
#[derive(Clone, Debug, PartialEq)]
pub struct WanCall<'a> {
    pub device: &'a str,
    pub height: usize,
    pub width: usize,
    pub steps: u32,
    pub strength: f64,
    pub seed: u64,
}

/// The 3D Wan 2.1 video-to-video seam (`VideoWanRuntime`).
pub trait WanPipeline: Send + Sync + fmt::Debug {
    fn model(&self) -> &str;
    fn revision(&self) -> &str;
    /// `resolve_wan_device`: the device Wan runs on for `requested`, or why it cannot run.
    fn resolve_device(&self, requested: &str) -> Result<String>;
    /// `_cuda_vram_below`: whether the accelerator has less than `limit` bytes (false if unknown).
    fn vram_below(&self, _limit: u64) -> bool {
        false
    }
    /// Regenerate padded RGB `[0, 1]` HWC frames (`1 + 4k` of them) into RGB `[0, 1]` frames.
    fn generate(&self, frames: &[Mat<f32>], call: &WanCall<'_>) -> Result<Vec<Mat<f32>>>;
}

/// `profile_strength`: the calibrated flow-matching denoising strength.
pub fn profile_strength(profile: &str) -> Result<f64> {
    match profile {
        "wan21-synthid" | "wan21-synthid-experimental" => Ok(WAN21_SYNTHID_STRENGTH),
        "wan21-seedance" | "wan21-seedance-experimental" => Ok(WAN21_SEEDANCE_STRENGTH),
        _ => Err(VideoError::value(
            "profile must be wan21-synthid-experimental or wan21-seedance-experimental",
        )),
    }
}

/// `can_run_wan_on_source`: source, device, VRAM and runtime eligibility.
pub fn can_run_wan_on_source(
    source: &Path,
    device: &str,
    wan: Option<&dyn WanPipeline>,
    for_auto: bool,
) -> bool {
    if !crate::api::is_regenerable(source) || !matches!(device, "auto" | "cuda" | "mps") {
        return false;
    }
    if for_auto && device == "mps" {
        return false;
    }
    let Some(wan) = wan else {
        return false;
    };
    if wan.model() != WAN21_MODEL_ID || wan.revision() != WAN21_MODEL_REVISION {
        return false;
    }
    match wan.resolve_device(device) {
        Ok(resolved) if for_auto && resolved == "cuda" && wan.vram_below(WAN21_MIN_CUDA_VRAM_BYTES) => {
            return false;
        }
        Ok(_) => {}
        Err(_) => return false,
    }
    probe_video_encode_profile(source)
        .and_then(|p| validate_experimental_mp4_source(source, "Experimental Wan 2.1 processing", &p, true))
        .is_ok()
}

fn validate_window_args(chunk_frames: usize, overlap_frames: usize) -> Result<()> {
    if chunk_frames < 17 || !(chunk_frames - 1).is_multiple_of(4) {
        return Err(VideoError::value("chunk_frames must be a 4k+1 integer >= 17"));
    }
    if overlap_frames >= chunk_frames {
        return Err(VideoError::value(
            "overlap_frames must satisfy 0 <= overlap_frames < chunk_frames",
        ));
    }
    Ok(())
}

/// `_effective_chunk_frames`: 33 on MPS, 81 above 720p or below 16 GiB, else as requested.
pub fn effective_chunk_frames(
    chunk_frames: usize,
    overlap_frames: usize,
    (width, height): (usize, usize),
    resolved_device: &str,
    wan: &dyn WanPipeline,
) -> usize {
    if chunk_frames != WAN21_DEFAULT_CHUNK_FRAMES {
        return chunk_frames;
    }
    if resolved_device == "mps" && overlap_frames < WAN21_MPS_CHUNK_FRAMES {
        return WAN21_MPS_CHUNK_FRAMES;
    }
    if overlap_frames >= WAN21_COMPACT_CHUNK_FRAMES {
        return chunk_frames;
    }
    if width * height > 1280 * 720 || wan.vram_below(WAN21_HIGH_VRAM_BYTES) {
        return WAN21_COMPACT_CHUNK_FRAMES;
    }
    chunk_frames
}

/// `_effective_overlap_frames`: 9 with the 33-frame MPS window.
pub fn effective_overlap_frames(overlap_frames: usize, chunk: usize, resolved_device: &str) -> usize {
    if resolved_device == "mps"
        && chunk == WAN21_MPS_CHUNK_FRAMES
        && overlap_frames == WAN21_DEFAULT_OVERLAP_FRAMES
    {
        WAN21_MPS_OVERLAP_FRAMES
    } else {
        overlap_frames
    }
}

/// `_wan_guide_size`: a 1080p-bounded, 16-aligned guide for larger sources.
pub fn wan_guide_size(width: usize, height: usize) -> Option<(usize, usize)> {
    if width * height <= WAN21_NATIVE_MAX_PIXELS {
        return None;
    }
    let scale = (WAN21_NATIVE_MAX_PIXELS as f64 / (width * height) as f64).sqrt();
    let fit = |v: usize| {
        ((v as f64 * scale) as usize / WAN21_SPATIAL_MULTIPLE * WAN21_SPATIAL_MULTIPLE)
            .max(WAN21_SPATIAL_MULTIPLE)
    };
    Some((fit(width), fit(height)))
}

/// `np.pad(mode="reflect")` index (edge excluded) for a padded coordinate.
fn reflect(index: isize, len: usize) -> usize {
    if len == 1 {
        return 0;
    }
    let period = 2 * (len as isize - 1);
    let m = index.rem_euclid(period);
    (if m < len as isize { m } else { period - m }) as usize
}

/// Window parameters fixed for one regeneration.
#[derive(Clone, Copy, Debug)]
struct WindowCall<'a> {
    wan: &'a dyn WanPipeline,
    device: &'a str,
    strength: f64,
    steps: u32,
    native: (usize, usize),
}

/// `_run_wan_window`: pad one window, run 3D diffusion, crop back, restore guides.
fn run_window<T: VaePixel>(window: &[Mat<T>], call: WindowCall<'_>, seed: u64) -> Result<Vec<Mat<T>>> {
    let (orig_w, orig_h) = call.native;
    let guide = wan_guide_size(orig_w, orig_h);
    let (work_w, work_h) = guide.unwrap_or((orig_w, orig_h));
    let work: Vec<Mat<T>> = match guide {
        Some((w, h)) => window
            .iter()
            .map(|f| resize(f, w, h, Interpolation::Area))
            .collect::<std::result::Result<_, _>>()?,
        None => window.to_vec(),
    };
    let pad_h = (WAN21_SPATIAL_MULTIPLE - work_h % WAN21_SPATIAL_MULTIPLE) % WAN21_SPATIAL_MULTIPLE;
    let pad_w = (WAN21_SPATIAL_MULTIPLE - work_w % WAN21_SPATIAL_MULTIPLE) % WAN21_SPATIAL_MULTIPLE;
    let (top, left) = (pad_h / 2, pad_w / 2);
    let (target_h, target_w) = (work_h + pad_h, work_w + pad_w);
    let peak = T::PEAK as f32;
    let mut frames: Vec<Mat<f32>> = Vec::with_capacity(work.len() + 3);
    for frame in &work {
        let src = frame.data();
        let mut data = Vec::with_capacity(target_h * target_w * 3);
        for y in 0..target_h {
            let sy = reflect(y as isize - top as isize, work_h);
            for x in 0..target_w {
                let sx = reflect(x as isize - left as isize, work_w);
                let px = &src[(sy * work_w + sx) * 3..][..3];
                data.extend([px[2], px[1], px[0]].map(|v| v.to_f64() as f32 / peak));
            }
        }
        frames.push(Mat::from_vec(target_w, target_h, 3, data)?);
    }
    let pad_t = (4 - (frames.len() - 1) % 4) % 4;
    if let Some(last) = frames.last().cloned() {
        frames.extend(std::iter::repeat_n(last, pad_t));
    }
    let generated = call.wan.generate(
        &frames,
        &WanCall {
            device: call.device,
            height: target_h,
            width: target_w,
            steps: call.steps,
            strength: call.strength,
            seed,
        },
    )?;
    if generated.len() < work.len()
        || generated
            .iter()
            .any(|g| g.size() != (target_w, target_h) || g.channels() != 3)
    {
        return Err(VideoError::runtime(format!(
            "Wan 2.1 output geometry did not match ({}, {orig_h}, {orig_w}, 3)",
            work.len()
        )));
    }
    let mut out = Vec::with_capacity(work.len());
    for (index, rgb) in generated.iter().take(work.len()).enumerate() {
        let src = rgb.data();
        let mut data = Vec::with_capacity(work_h * work_w * 3);
        for y in 0..work_h {
            for x in 0..work_w {
                let px = &src[((y + top) * target_w + x + left) * 3..][..3];
                data.extend(
                    [px[2], px[1], px[0]]
                        .map(|v| T::from_f64(f64::from((v * peak).round_ties_even().clamp(0.0, peak)))),
                );
            }
        }
        let cropped = Mat::from_vec(work_w, work_h, 3, data)?;
        out.push(match guide {
            Some(_) => {
                let up = resize(&cropped, orig_w, orig_h, Interpolation::Lanczos4)?;
                restore_texture(&window[index], &anchor_low_freq_color(&window[index], &up, None)?)?
            }
            None => cropped,
        });
    }
    Ok(out)
}

/// Window and overlap sizes of one streamed regeneration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WanWindowing {
    pub chunk_frames: usize,
    pub overlap_frames: usize,
}

impl Default for WanWindowing {
    fn default() -> Self {
        Self {
            chunk_frames: WAN21_DEFAULT_CHUNK_FRAMES,
            overlap_frames: WAN21_DEFAULT_OVERLAP_FRAMES,
        }
    }
}

/// `_stream_wan_windows`: retains one active window and the overlap tail.
pub struct WanWindows<'a, T: VaePixel, I: Iterator<Item = Result<Mat<T>>>> {
    input: I,
    call: WindowCall<'a>,
    seed: u64,
    windowing: WanWindowing,
    fade_in: Vec<f32>,
    peek: Option<Mat<T>>,
    in_tail: Vec<Mat<T>>,
    out_tail: Option<Vec<Mat<T>>>,
    window_index: u64,
    ready: VecDeque<Mat<T>>,
    started: bool,
    done: bool,
}

impl<T: VaePixel, I: Iterator<Item = Result<Mat<T>>>> fmt::Debug for WanWindows<'_, T, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WanWindows")
            .field("windowing", &self.windowing)
            .field("window_index", &self.window_index)
            .finish_non_exhaustive()
    }
}

impl<'a, T: VaePixel, I: Iterator<Item = Result<Mat<T>>>> WanWindows<'a, T, I> {
    pub fn new(
        input: I,
        wan: &'a dyn WanPipeline,
        device: &'a str,
        strength: f64,
        seed: u64,
        steps: u32,
        windowing: WanWindowing,
    ) -> Result<Self> {
        validate_window_args(windowing.chunk_frames, windowing.overlap_frames)?;
        let overlap = windowing.overlap_frames;
        let fade_in = (1..=overlap)
            .map(|i| {
                let t = (std::f32::consts::PI * i as f32 / (overlap as f32 + 1.0)).cos();
                0.5 - 0.5 * t
            })
            .collect();
        Ok(Self {
            input,
            call: WindowCall {
                wan,
                device,
                strength,
                steps,
                native: (0, 0),
            },
            seed,
            windowing,
            fade_in,
            peek: None,
            in_tail: Vec::new(),
            out_tail: None,
            window_index: 0,
            ready: VecDeque::new(),
            started: false,
            done: false,
        })
    }

    fn pull(&mut self) -> Result<Option<Mat<T>>> {
        self.input.next().transpose()
    }

    fn blend(&self, current: &mut [Mat<T>]) -> Result<()> {
        let Some(tail) = &self.out_tail else {
            return Ok(());
        };
        let peak = T::PEAK as f32;
        for ((frame, previous), fade) in current.iter_mut().zip(tail).zip(&self.fade_in) {
            let fade_out = 1.0 - fade;
            let data = previous
                .data()
                .iter()
                .zip(frame.data())
                .map(|(p, c)| {
                    let mixed = p.to_f64() as f32 * fade_out + c.to_f64() as f32 * fade;
                    T::from_f64(f64::from(mixed.round_ties_even().clamp(0.0, peak)))
                })
                .collect();
            *frame = Mat::from_vec(frame.width(), frame.height(), 3, data)?;
        }
        Ok(())
    }

    fn step(&mut self) -> Result<()> {
        if !self.started {
            self.started = true;
            let Some(first) = self.pull()? else {
                self.done = true;
                return Ok(());
            };
            self.call.native = first.size();
            self.peek = Some(first);
        }
        let Some(peek) = self.peek.take() else {
            self.done = true;
            return Ok(());
        };
        let WanWindowing {
            chunk_frames,
            overlap_frames,
        } = self.windowing;
        let step = chunk_frames - overlap_frames;
        let need = if self.window_index == 0 {
            chunk_frames
        } else {
            step
        };
        let mut window = std::mem::take(&mut self.in_tail);
        window.push(peek);
        for _ in 1..need {
            match self.pull()? {
                Some(frame) => window.push(frame),
                None => break,
            }
        }
        let next_peek = self.pull()?;
        let mut current = run_window(&window, self.call, self.seed + self.window_index)?;
        if overlap_frames > 0 {
            self.blend(&mut current)?;
        }
        if next_peek.is_none() {
            self.ready.extend(current);
            self.done = true;
            return Ok(());
        }
        let split = step.min(current.len());
        let tail = current.split_off(split);
        self.ready.extend(current);
        if overlap_frames > 0 {
            self.in_tail = window.split_off(step.min(window.len()));
            self.out_tail = Some(tail);
        } else {
            self.out_tail = None;
        }
        self.peek = next_peek;
        self.window_index += 1;
        Ok(())
    }
}

impl<T: VaePixel, I: Iterator<Item = Result<Mat<T>>>> Iterator for WanWindows<'_, T, I> {
    type Item = Result<Mat<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.ready.is_empty() && !self.done {
            if let Err(error) = self.step() {
                self.done = true;
                return Some(Err(error));
            }
        }
        self.ready.pop_front().map(Ok)
    }
}

/// Options of [`regenerate_wan_experimental`].
#[derive(Clone, Debug, PartialEq)]
pub struct WanOptions {
    pub profile: String,
    pub device: String,
    pub quality: SeedanceQuality,
    pub seed: u64,
    pub windowing: WanWindowing,
    pub inference_steps: Option<u32>,
}

impl Default for WanOptions {
    fn default() -> Self {
        Self {
            profile: "wan21-synthid-experimental".into(),
            device: "auto".into(),
            quality: SeedanceQuality::Delivery,
            seed: 0,
            windowing: WanWindowing::default(),
            inference_steps: None,
        }
    }
}

/// Counts decoded frames against the probed timestamps, as `counted_source`.
struct Counted<I> {
    inner: I,
    count: usize,
    expected: usize,
    checked: bool,
}

impl<T, I: Iterator<Item = Result<Mat<T>>>> Iterator for Counted<I> {
    type Item = Result<Mat<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner.next() {
            Some(item) => {
                self.count += 1;
                Some(item)
            }
            None if !self.checked => {
                self.checked = true;
                (self.count != self.expected).then(|| {
                    Err(VideoError::runtime(
                        "Decoded frame count did not match probed source frame timestamps",
                    ))
                })
            }
            None => None,
        }
    }
}

/// `regenerate_wan_experimental`: pinned 3D Wan 2.1 video-to-video with exact timestamps.
/// `frames` replaces the source decode with already-cleaned native frames (`_frame_stream`).
pub fn regenerate_wan_experimental<T: VaePixel>(
    wan: &dyn WanPipeline,
    source: &Path,
    output: &Path,
    options: &WanOptions,
    frames: Option<&mut dyn Iterator<Item = Result<Mat<T>>>>,
) -> Result<RegenerationMetrics> {
    let strength = profile_strength(&options.profile)?;
    if !matches!(options.device.as_str(), "auto" | "cuda" | "mps") {
        return Err(VideoError::value(
            "Wan 2.1 3D video diffusion requires device='auto', 'cuda', or 'mps'",
        ));
    }
    let steps = options.inference_steps.unwrap_or(WAN21_INFERENCE_STEPS);
    if steps < 1 {
        return Err(VideoError::value("inference_steps must be >= 1"));
    }
    validate_window_args(options.windowing.chunk_frames, options.windowing.overlap_frames)?;
    if wan.model() != WAN21_MODEL_ID || wan.revision() != WAN21_MODEL_REVISION {
        return Err(VideoError::value(
            "The supplied Wan 2.1 runtime does not match the pinned model and revision",
        ));
    }
    let profile = probe_video_encode_profile(source)?;
    if profile.high_depth() != (T::BYTES == 2) {
        return Err(VideoError::value(
            "Wan 2.1 frame depth does not match the source pixel format",
        ));
    }
    let clip: ExperimentalSource =
        validate_experimental_mp4_source(source, "Experimental Wan 2.1 processing", &profile, true)?;
    let resolved = wan.resolve_device(&options.device)?;
    tracing::warn!(
        "Applying experimental {} (strength={strength:.2}, {}); no per-file oracle verification is performed",
        options.profile,
        options.quality.as_str()
    );
    let chunk = effective_chunk_frames(
        options.windowing.chunk_frames,
        options.windowing.overlap_frames,
        (clip.width, clip.height),
        &resolved,
        wan,
    );
    let windowing = WanWindowing {
        chunk_frames: chunk,
        overlap_frames: effective_overlap_frames(options.windowing.overlap_frames, chunk, &resolved),
    };
    let staged = atomic_video_output(output)?;
    let mut decoded;
    let raw: &mut dyn Iterator<Item = Result<Mat<T>>> = if let Some(stream) = frames {
        stream
    } else {
        let mut reader = crate::decode::VideoReader::<T>::open(source)?;
        decoded = std::iter::from_fn(move || reader.read().transpose());
        &mut decoded
    };
    let counted = Counted {
        inner: raw,
        count: 0,
        expected: clip.timestamps.len(),
        checked: false,
    };
    let mut windows = WanWindows::new(counted, wan, &resolved, strength, options.seed, steps, windowing)?;
    let crf = match options.quality {
        SeedanceQuality::Delivery => 18,
        SeedanceQuality::Master => 0,
    };
    encode_timestamped_frames(
        &mut windows,
        source,
        staged.path(),
        clip.fps,
        &clip.timestamps,
        crf,
    )?;
    verify_isobmff_timing(source, staged.path(), &clip.timestamps, "Wan 2.1")?;
    let metrics = measure_native::<T>(source, staged.path(), clip.fps, clip.timestamps.len())?;
    staged.commit()?;
    Ok(metrics)
}
