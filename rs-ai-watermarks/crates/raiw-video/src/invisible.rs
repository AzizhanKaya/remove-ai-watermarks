//! Invisible-watermark regeneration orchestration (`video_invisible.py`, `video_synthid.py`, and
//! the profile routing of `video.py`).
//!
//! The neural pieces are trait seams the `gpu` feature wires to `raiw-gpu`: the 2D video VAE
//! ([`VideoVae`]), the 3D Wan 2.1 pipeline ([`WanPipeline`]) and the VideoSeal-family message
//! models ([`crate::messages::VideoMessageModels`]). Everything around them (sampling, batching,
//! the shared latent noise field, temporal stabilization, guide upscaling, encoding, metrics)
//! runs here and is tested with fakes.

use std::collections::VecDeque;
use std::fmt;
use std::path::Path;
use std::sync::Arc;

use raiw_core::constants::{VIDEO_INVISIBLE_PROFILES, VIDEO_PROFILE_ALIASES, WAN_EXPERIMENTAL_PROFILES};
use raiw_cv::resize::Resizable;
use raiw_cv::{Border, ColorConversion, Interpolation, Mat, add_weighted, cvt_color, gaussian_blur, resize};
use raiw_metadata::AiMetadata;

use crate::decode::frame_bytes;
use crate::encode::{
    MuxOptions, RawVideoOptions, StagedVideoOutput, command_from, mux_encoded_video, raw_video_command,
};
use crate::error::{Result, VideoError};
use crate::messages::VideoMessageModels;
use crate::probe::{probe_video_encode_profile, probe_video_geometry};
use crate::process::RawVideoEncoder;
use crate::provenance::is_bytedance_video_provenance;
use crate::report::RegenerationMetrics;
use crate::sampling::{FrameSampler, SampledFrames, SamplingOptions};
use crate::temporal::VideoPixel;
use crate::temporal::color::anchor_low_freq_color;
use crate::temporal::flow::{backward_motion_map, motion_residual_energy};
use crate::temporal::regen::{RegenStabilizeOptions, RegeneratedStabilizer};
pub use crate::wan::WanPipeline;

pub const DEFAULT_VIDEO_SYNTHID_VAE: &str = "stabilityai/sd-vae-ft-mse";
pub const DEFAULT_VIDEO_SYNTHID_VAE_REVISION: &str = "31f26fdeee1355a5c34592e401dd41e45d25a493";
pub const VIDEO_SYNTHID_VAE_SCALING_FACTOR: f64 = 0.18215;
pub const DEFAULT_VIDEO_SYNTHID_NOISE_STD: f64 = 0.15;
pub const DEFAULT_VIDEO_SYNTHID_LONG_SIDE: usize = 512;
pub const DEFAULT_VIDEO_SYNTHID_FPS: f64 = 12.0;
pub const VIDEO_SYNTHID_LATENT_MULTIPLE: usize = 8;
pub const VIDEO_SYNTHID_VERIFICATION_PROMPT: &str = "For the video attached to this message, was it created or edited by \
                                                     Google AI? Use the built-in SynthID content verification result.";
/// The VAE switches to tiled decoding above this guide long side.
pub const VAE_TILING_LONG_SIDE: usize = 768;

/// Latents in NCHW order.
#[derive(Clone, Debug, PartialEq)]
pub struct Latents {
    pub shape: [usize; 4],
    pub data: Vec<f32>,
}

/// The 2D SD-VAE seam (`VideoVaeRuntime`). Frames cross as RGB `[-1, 1]` HWC float planes;
/// latents are scaled by the runtime's validated factor on both sides.
pub trait VideoVae: Send + Sync + fmt::Debug {
    fn model(&self) -> &str;
    fn requested_device(&self) -> &str;
    /// The latent scaling factor the load gate validated (`VideoVaeRuntime.scaling_factor`).
    fn scaling_factor(&self) -> f64;
    /// `vae.encode(batch).latent_dist.mode() * scaling_factor`.
    fn encode(&self, rgb: &[Mat<f32>]) -> Result<Latents>;
    /// `vae.decode(latents / scaling_factor).sample` as RGB `[-1, 1]` HWC planes.
    fn decode(&self, latents: &Latents) -> Result<Vec<Mat<f32>>>;
    /// `torch.randn((1, *shape), generator=torch.Generator("cpu").manual_seed(seed))`.
    fn shared_noise(&self, shape: [usize; 3], seed: u64) -> Result<Vec<f32>>;
    /// Toggle tiled decoding for large guides (`enable_tiling` / `disable_tiling`).
    fn set_tiling(&self, _enabled: bool) {}
}

/// The neural runtimes a build provides (all empty without the `gpu` feature).
#[derive(Clone, Debug, Default)]
pub struct NeuralBackends {
    pub vae: Option<Arc<dyn VideoVae>>,
    pub wan: Option<Arc<dyn WanPipeline>>,
    pub messages: Option<Arc<dyn VideoMessageModels>>,
}

pub(crate) fn unavailable(what: &str) -> VideoError {
    VideoError::Unavailable(format!(
        "{what} needs the neural video runtime, which this build does not include (enable the `gpu` feature)"
    ))
}

/// `_fit_size`: fit to `long_side` keeping aspect and 8-pixel VAE alignment.
pub fn fit_size(width: usize, height: usize, long_side: usize) -> Result<(usize, usize)> {
    if width == 0 || height == 0 {
        return Err(VideoError::value("Video dimensions must be positive"));
    }
    if long_side < VIDEO_SYNTHID_LATENT_MULTIPLE {
        return Err(VideoError::value(format!(
            "Long side must be at least {VIDEO_SYNTHID_LATENT_MULTIPLE}"
        )));
    }
    let scale = long_side as f64 / width.max(height) as f64;
    let fit = |v: usize| {
        let rounded = crate::py::round_int(v as f64 * scale).max(0) as usize;
        (rounded / VIDEO_SYNTHID_LATENT_MULTIPLE * VIDEO_SYNTHID_LATENT_MULTIPLE)
            .max(VIDEO_SYNTHID_LATENT_MULTIPLE)
    };
    Ok((fit(width), fit(height)))
}

/// The 2D VAE tuning knobs (and the profile routing inputs).
#[derive(Clone, Debug, PartialEq)]
pub struct InvisibleOptions {
    pub profile: String,
    pub quality: String,
    pub noise_std: f64,
    pub long_side: usize,
    pub fps: Option<f64>,
    pub batch_size: usize,
    pub seed: u64,
    pub model: String,
    pub device: String,
    pub temporal_stabilize: bool,
    pub native_guide: bool,
    pub wan_steps: Option<u32>,
}

impl Default for InvisibleOptions {
    fn default() -> Self {
        Self {
            profile: "auto".into(),
            quality: "delivery".into(),
            noise_std: DEFAULT_VIDEO_SYNTHID_NOISE_STD,
            long_side: DEFAULT_VIDEO_SYNTHID_LONG_SIDE,
            fps: Some(DEFAULT_VIDEO_SYNTHID_FPS),
            batch_size: 4,
            seed: 0,
            model: DEFAULT_VIDEO_SYNTHID_VAE.into(),
            device: "auto".into(),
            temporal_stabilize: true,
            native_guide: false,
            wan_steps: None,
        }
    }
}

impl InvisibleOptions {
    /// `_has_2d_vae_overrides`.
    pub fn has_2d_vae_overrides(&self) -> bool {
        self.noise_std != DEFAULT_VIDEO_SYNTHID_NOISE_STD
            || self.long_side != DEFAULT_VIDEO_SYNTHID_LONG_SIDE
            || self.fps != Some(DEFAULT_VIDEO_SYNTHID_FPS)
            || self.batch_size != 4
            || self.model != DEFAULT_VIDEO_SYNTHID_VAE
            || !self.temporal_stabilize
            || self.native_guide
    }
}

/// `_normalize_video_profile`.
pub fn normalize_video_profile(profile: &str) -> &str {
    VIDEO_PROFILE_ALIASES
        .iter()
        .find(|(from, _)| *from == profile)
        .map_or(profile, |(_, to)| to)
}

/// Whether include-invisible is requested: explicitly (`true`), automatically, or not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InvisibleMode {
    Off,
    On,
    Auto,
}

/// `_validate_video_profile`.
pub fn validate_video_profile(options: &InvisibleOptions, enabled: InvisibleMode) -> Result<()> {
    if !VIDEO_INVISIBLE_PROFILES.contains(&options.profile.as_str()) {
        return Err(VideoError::value(
            "profile must be auto, synthid, seedance, wan21-synthid, or wan21-seedance (legacy *-experimental aliases are also accepted)",
        ));
    }
    let profile = normalize_video_profile(&options.profile);
    if !matches!(options.quality.as_str(), "delivery" | "master") {
        return Err(VideoError::value("quality must be delivery or master"));
    }
    let enabled_bool = enabled != InvisibleMode::Off;
    if !matches!(profile, "auto" | "synthid") && !enabled_bool {
        return Err(VideoError::value(
            "Selecting an invisible profile requires include_invisible=True (--invisible)",
        ));
    }
    if let Some(steps) = options.wan_steps {
        if steps < 1 {
            return Err(VideoError::value("wan_steps must be >= 1"));
        }
        if !enabled_bool {
            return Err(VideoError::value(
                "wan_steps requires include_invisible=True (--invisible)",
            ));
        }
        if matches!(profile, "synthid" | "seedance-experimental") {
            return Err(VideoError::value(
                "wan_steps is only supported for Wan 2.1 profiles (wan21-synthid, wan21-seedance, or auto)",
            ));
        }
    }
    let fixed_quality = matches!(
        profile,
        "auto" | "seedance-experimental" | "wan21-synthid-experimental" | "wan21-seedance-experimental"
    );
    if options.quality != "delivery" && (!enabled_bool || !fixed_quality) {
        return Err(VideoError::value(
            "quality=master requires the enabled seedance-experimental or wan21-*-experimental profile",
        ));
    }
    let overrides = options.has_2d_vae_overrides();
    if profile == "seedance-experimental" && (overrides || options.seed != 0) {
        return Err(VideoError::value(
            "Seedance uses a fixed recipe; SynthID tuning options must retain their defaults",
        ));
    }
    if WAN_EXPERIMENTAL_PROFILES.contains(&profile) && overrides {
        return Err(VideoError::value(
            "Wan 2.1 profiles use fixed native geometry and frame timing; 2D VAE SynthID tuning options must retain their defaults",
        ));
    }
    if options.wan_steps.is_some() && overrides {
        return Err(VideoError::value(
            "wan_steps cannot be combined with 2D VAE SynthID tuning options",
        ));
    }
    Ok(())
}

/// `_resolve_video_invisible_profile`: route `auto` by provenance and device capability.
pub fn resolve_video_invisible_profile(
    neural: &NeuralBackends,
    options: &InvisibleOptions,
    source: &Path,
    markers: &AiMetadata,
    visible_mark: Option<&str>,
) -> Result<String> {
    if options.profile != "auto" {
        return Ok(normalize_video_profile(&options.profile).to_owned());
    }
    let resolved = if options.has_2d_vae_overrides() || !crate::api::is_regenerable(source) {
        if options.wan_steps.is_some() {
            return Err(VideoError::value(
                "wan_steps cannot be combined with 2D VAE SynthID tuning options or unsupported containers",
            ));
        }
        "synthid"
    } else {
        let can_wan = crate::wan::can_run_wan_on_source(
            source,
            &options.device,
            neural.wan.as_deref(),
            options.wan_steps.is_none(),
        );
        if !can_wan && options.wan_steps.is_some() {
            return Err(VideoError::value(
                "wan_steps requires an eligible CUDA or MPS Wan 2.1 device and runtime",
            ));
        }
        if !can_wan && options.seed != 0 {
            "synthid"
        } else if is_bytedance_video_provenance(markers, visible_mark) {
            if can_wan {
                "wan21-seedance-experimental"
            } else {
                "seedance-experimental"
            }
        } else if can_wan {
            "wan21-synthid-experimental"
        } else {
            "synthid"
        }
    };
    if options.quality != "delivery" && resolved == "synthid" {
        return Err(VideoError::value(
            "quality=master requires the enabled seedance-experimental or wan21-*-experimental profile",
        ));
    }
    Ok(resolved.to_owned())
}

/// A frame depth the VAE accepts: `uint8` (`/ 127.5 - 1`) or `uint16` (`/ 32767.5 - 1`).
pub trait VaePixel: VideoPixel + Resizable {
    const UNIT_SCALE: f32;
    const QUANT_PEAK: f32;
    const RAW_PIX_FMT: &'static str;
}

impl VaePixel for u8 {
    const UNIT_SCALE: f32 = 127.5;
    const QUANT_PEAK: f32 = 255.0;
    const RAW_PIX_FMT: &'static str = "bgr24";
}

impl VaePixel for u16 {
    const UNIT_SCALE: f32 = 32767.5;
    const QUANT_PEAK: f32 = 65535.0;
    const RAW_PIX_FMT: &'static str = "bgr48le";
}

/// BGR frame to an RGB `[-1, 1]` plane.
pub fn bgr_to_unit_rgb<T: VaePixel>(frame: &Mat<T>) -> Result<Mat<f32>> {
    let mut data = Vec::with_capacity(frame.data().len());
    for px in frame.data().chunks_exact(3) {
        for &v in &[px[2], px[1], px[0]] {
            data.push(v.to_f64() as f32 / T::UNIT_SCALE - 1.0);
        }
    }
    Ok(Mat::from_vec(frame.width(), frame.height(), 3, data)?)
}

/// RGB `[-1, 1]` plane to a BGR frame: `round(clamp(x / 2 + 0.5, 0, 1) * peak)`.
pub fn unit_rgb_to_bgr<T: VaePixel>(plane: &Mat<f32>) -> Result<Mat<T>> {
    if plane.data().iter().any(|v| !v.is_finite()) {
        return Err(VideoError::runtime("Video VAE decoder returned nonfinite pixels"));
    }
    let quantize = |v: f32| {
        T::from_f64(f64::from(
            ((v / 2.0 + 0.5).clamp(0.0, 1.0) * T::QUANT_PEAK).round_ties_even(),
        ))
    };
    let mut data = Vec::with_capacity(plane.data().len());
    for px in plane.data().chunks_exact(3) {
        data.extend([quantize(px[2]), quantize(px[1]), quantize(px[0])]);
    }
    Ok(Mat::from_vec(plane.width(), plane.height(), 3, data)?)
}

/// `_encode_frame_latents` + `_decode_frame_latents` with one perturbation shared across time.
pub(crate) struct LatentRound<'a> {
    pub vae: &'a dyn VideoVae,
    pub noise_std: f64,
    pub seed: u64,
    /// `None` until the first batch fixes the latent shape; a zero field skips the draw.
    pub noise: Option<Vec<f32>>,
}

impl LatentRound<'_> {
    pub(crate) fn regenerate<T: VaePixel>(&mut self, frames: &[Mat<T>]) -> Result<Vec<Mat<T>>> {
        let rgb: Vec<Mat<f32>> = frames.iter().map(bgr_to_unit_rgb).collect::<Result<_>>()?;
        let mut latents = self.vae.encode(&rgb)?;
        if latents.data.iter().any(|v| !v.is_finite()) {
            return Err(VideoError::runtime(
                "Video VAE encoder returned nonfinite latents",
            ));
        }
        let [_, c, h, w] = latents.shape;
        let plane = c * h * w;
        if self.noise.is_none() {
            self.noise = Some(self.vae.shared_noise([c, h, w], self.seed)?);
        }
        let noise = self.noise.as_deref().unwrap_or_default();
        if noise.len() != plane {
            return Err(VideoError::runtime(
                "Shared latent noise does not match the latent shape",
            ));
        }
        let std = self.noise_std as f32;
        for (i, value) in latents.data.iter_mut().enumerate() {
            *value += std * noise[i % plane];
        }
        self.vae.decode(&latents)?.iter().map(unit_rgb_to_bgr).collect()
    }
}

/// `humanizer.unsharp_mask`: `image + amount * (image - blur(image))`, rounded and clipped.
pub fn unsharp_mask<T: VaePixel>(image: &Mat<T>, amount: f64, sigma: f64) -> Result<Mat<T>> {
    if amount <= 0.0 {
        return Ok(image.clone());
    }
    let float: Mat<f32> = image.convert();
    let blurred = gaussian_blur(&float, (0, 0), sigma, sigma, Border::Reflect101)?;
    let sharpened = add_weighted(&float, 1.0 + amount, &blurred, -amount, 0.0)?;
    let peak = T::PEAK as f32;
    Ok(sharpened.map(|v| T::from_f64(f64::from(v.round_ties_even().clamp(0.0, peak)))))
}

/// `_upscale_clean_guide`: color-anchor at guide resolution, Lanczos-upscale and lightly sharpen.
pub fn upscale_clean_guide<T: VaePixel>(
    native_reference: &Mat<T>,
    small_reference: &Mat<T>,
    guide_small: &Mat<T>,
) -> Result<Mat<T>> {
    let (width, height) = native_reference.size();
    let anchored = anchor_low_freq_color(small_reference, guide_small, None)?;
    if anchored.size() == (width, height) {
        return Ok(anchored);
    }
    let upscaled = resize(&anchored, width, height, Interpolation::Lanczos4)?;
    unsharp_mask(&upscaled, 0.35, 1.2)
}

/// Fidelity accumulators of `regenerate_video_candidate`.
#[derive(Default)]
struct Metrics {
    frames: usize,
    squared_error: f64,
    pixels: usize,
    baseline: f64,
    candidate: f64,
    previous: Option<(Mat<u8>, Mat<f32>, Mat<f32>)>,
}

/// `(frame >> 8).astype(uint8)` for 16-bit frames, then BGR to gray.
fn reference_gray<T: VideoPixel>(frame: &Mat<T>) -> Result<Mat<u8>> {
    let narrow: Mat<u8> = if T::BYTES == 1 {
        frame.map(|v| v.to_f64() as u8)
    } else {
        frame.map(|v| ((v.to_f64() as u16) >> 8) as u8)
    };
    Ok(cvt_color(&narrow, ColorConversion::BgrToGray)?)
}

impl Metrics {
    fn add<T: VideoPixel>(&mut self, reference: &Mat<T>, candidate: &Mat<T>) -> Result<()> {
        let (r32, c32): (Mat<f32>, Mat<f32>) = (reference.convert(), candidate.convert());
        self.squared_error += r32
            .data()
            .iter()
            .zip(c32.data())
            .map(|(a, b)| f64::from((a - b) * (a - b)))
            .sum::<f64>();
        self.pixels += reference.data().len();
        let gray = reference_gray(reference)?;
        if let Some((pg, pr, pc)) = &self.previous {
            let maps = backward_motion_map(&gray, pg)?;
            self.baseline += motion_residual_energy(&r32, pr, &maps)?;
            self.candidate += motion_residual_energy(&c32, pc, &maps)?;
        }
        self.previous = Some((gray, r32, c32));
        self.frames += 1;
        Ok(())
    }

    fn finish(&self, fps: f64, (width, height): (usize, usize), peak: f64) -> RegenerationMetrics {
        let mse = self.squared_error / self.pixels.max(1) as f64;
        RegenerationMetrics {
            frames: self.frames,
            fps,
            width,
            height,
            psnr_db: if mse == 0.0 {
                f64::INFINITY
            } else {
                20.0 * (peak / mse.sqrt()).log10()
            },
            temporal_residual_ratio: self.candidate / self.baseline.max(1e-6),
        }
    }
}

/// Where the frames of one regeneration come from.
pub enum FrameFeed<'a, T> {
    /// Decode and sample the source file.
    Decode,
    /// Every decoded source frame in order (`_frame_stream`), sampled here.
    Stream(&'a mut dyn Iterator<Item = Result<Mat<T>>>),
}

impl<T> fmt::Debug for FrameFeed<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Decode => "FrameFeed::Decode",
            Self::Stream(_) => "FrameFeed::Stream",
        })
    }
}

/// The push side of `regenerate_video_candidate`: batches sampled frames through the VAE,
/// stabilizes, upscales guides, encodes and measures.
struct Regenerator<'a, T: VaePixel> {
    round: LatentRound<'a>,
    batch_size: usize,
    small: (usize, usize),
    native_guide: bool,
    batch: Vec<Mat<T>>,
    natives: VecDeque<Mat<T>>,
    stabilizer: Option<RegeneratedStabilizer<T>>,
    metrics: Metrics,
}

impl<T: VaePixel> Regenerator<'_, T> {
    fn push(&mut self, frame: Mat<T>, encoder: &mut RawVideoEncoder) -> Result<()> {
        self.batch.push(frame);
        if self.batch.len() == self.batch_size {
            self.flush(encoder)?;
        }
        Ok(())
    }

    fn flush(&mut self, encoder: &mut RawVideoEncoder) -> Result<()> {
        if self.batch.is_empty() {
            return Ok(());
        }
        let batch = std::mem::take(&mut self.batch);
        let small: Vec<Mat<T>> = if self.native_guide {
            let (w, h) = self.small;
            let small = batch
                .iter()
                .map(|f| {
                    if f.size() == (w, h) {
                        Ok(f.clone())
                    } else {
                        resize(f, w, h, Interpolation::Area)
                    }
                })
                .collect::<std::result::Result<_, _>>()?;
            self.natives.extend(batch);
            small
        } else {
            batch
        };
        let regenerated = self.round.regenerate(&small)?;
        for (reference, candidate) in small.into_iter().zip(regenerated) {
            let pair = match &mut self.stabilizer {
                Some(s) => s.push(reference, candidate)?,
                None => Some((reference, candidate)),
            };
            if let Some((r, c)) = pair {
                self.emit(r, &c, encoder)?;
            }
        }
        Ok(())
    }

    fn emit(&mut self, small_ref: Mat<T>, small_cand: &Mat<T>, encoder: &mut RawVideoEncoder) -> Result<()> {
        let (reference, candidate) = if self.native_guide {
            let native = self
                .natives
                .pop_front()
                .ok_or_else(|| VideoError::runtime("Guide frame queue ran empty"))?;
            let candidate = upscale_clean_guide(&native, &small_ref, small_cand)?;
            (native, candidate)
        } else {
            let candidate = upscale_clean_guide(&small_ref, &small_ref, small_cand)?;
            (small_ref, candidate)
        };
        encoder.write_frame(&frame_bytes(&candidate))?;
        self.metrics.add(&reference, &candidate)
    }

    fn finish(mut self, encoder: &mut RawVideoEncoder) -> Result<Metrics> {
        self.flush(encoder)?;
        if let Some(s) = self.stabilizer.take()
            && let Some((r, c)) = s.finish()?
        {
            self.emit(r, &c, encoder)?;
        }
        Ok(self.metrics)
    }
}

fn validate_regeneration(
    vae: &dyn VideoVae,
    options: &InvisibleOptions,
    duration: Option<f64>,
) -> Result<()> {
    if !(0.0..=1.0).contains(&options.noise_std) {
        return Err(VideoError::value("noise_std must be between 0 and 1"));
    }
    if options.fps.is_some_and(|f| f < 1.0) {
        return Err(VideoError::value("fps must be at least 1"));
    }
    if options.batch_size < 1 {
        return Err(VideoError::value("batch_size must be at least 1"));
    }
    if duration.is_some_and(|d| d <= 0.0) {
        return Err(VideoError::value("duration must be positive"));
    }
    if !matches!(options.device.as_str(), "auto" | "cuda" | "mps" | "cpu") {
        return Err(VideoError::value("device must be auto, cuda, mps, or cpu"));
    }
    if vae.model() != options.model || vae.requested_device() != options.device {
        return Err(VideoError::value(
            "The supplied video VAE runtime does not match the requested model and device",
        ));
    }
    Ok(())
}

/// `regenerate_video_candidate`: VAE-regenerate sampled frames, encode, and measure.
pub fn regenerate_video_candidate(
    vae: &dyn VideoVae,
    source: &Path,
    output: &Path,
    options: &InvisibleOptions,
    duration: Option<f64>,
    timestamps: Option<Vec<f64>>,
) -> Result<RegenerationMetrics> {
    if probe_video_encode_profile(source)?.high_depth() {
        regenerate_video_frames::<u16>(
            vae,
            source,
            output,
            options,
            duration,
            timestamps,
            FrameFeed::Decode,
        )
    } else {
        regenerate_video_frames::<u8>(
            vae,
            source,
            output,
            options,
            duration,
            timestamps,
            FrameFeed::Decode,
        )
    }
}

/// [`regenerate_video_candidate`] over an explicit frame feed and depth. High-depth sources
/// (10-bit `yuv4xxp10le`) regenerate through `bgr48le`; other HDR or deep streams are refused.
pub fn regenerate_video_frames<T: VaePixel>(
    vae: &dyn VideoVae,
    source: &Path,
    output: &Path,
    options: &InvisibleOptions,
    duration: Option<f64>,
    timestamps: Option<Vec<f64>>,
    feed: FrameFeed<'_, T>,
) -> Result<RegenerationMetrics> {
    validate_regeneration(vae, options, duration)?;
    let geometry = probe_video_geometry(source)?
        .ok_or_else(|| VideoError::value(format!("Could not open video: {}", source.display())))?;
    let (width, height, source_fps) = (geometry.width, geometry.height, geometry.fps);
    if source_fps <= 0.0 {
        return Err(VideoError::value(format!(
            "Video has no usable frame rate: {}",
            source.display()
        )));
    }
    let size = fit_size(width, height, options.long_side)?;
    let (out_w, out_h) = if options.native_guide {
        (width, height)
    } else {
        size
    };
    let effective_fps = options.fps.map_or(source_fps, |f| f.min(source_fps));
    if out_w < width || out_h < height || effective_fps < source_fps - 0.5 {
        tracing::warn!(
            "2D SynthID VAE fallback downscaling {width}x{height}@{source_fps:.2}fps -> {out_w}x{out_h}@{effective_fps:.2}fps; use --native-guide or CUDA Wan 2.1 (wan21-synthid) for native-resolution output"
        );
    }
    vae.set_tiling(size.0.max(size.1) > VAE_TILING_LONG_SIDE);
    let profile = probe_video_encode_profile(source)?;
    let high_depth = profile.high_depth();
    if (profile.component_depth.unwrap_or(0) > 8
        || matches!(
            profile.color_transfer.as_deref(),
            Some("smpte2084" | "arib-std-b67")
        ))
        && !high_depth
    {
        return Err(VideoError::runtime(format!(
            "Invisible video regeneration cannot preserve this high-bit-depth or HDR stream without refusing to silently reduce {} / {} to 8-bit SDR",
            profile
                .source_pixel_format
                .as_deref()
                .unwrap_or("unknown pixel format"),
            profile.color_transfer.as_deref().unwrap_or("unknown transfer")
        )));
    }
    if high_depth != (T::BYTES == 2) {
        return Err(VideoError::value(format!(
            "Regeneration depth {} does not match the source pixel format {}",
            T::RAW_PIX_FMT,
            profile.pixel_format
        )));
    }
    let staged = StagedVideoOutput::new(output)?;
    let encoded = staged.encoded_video.path().to_path_buf();
    let mut raw = RawVideoOptions::new(out_w, out_h, effective_fps, 18, &profile);
    raw.input_pix_fmt = T::RAW_PIX_FMT;
    let mut encoder = RawVideoEncoder::start(command_from(&raw_video_command(&encoded, &raw)?)?)?;
    let sampling = SamplingOptions {
        source_fps,
        duration,
        effective_fps,
        size: (out_w, out_h),
        timestamps,
    };
    let regenerator = Regenerator::<T> {
        round: LatentRound {
            vae,
            noise_std: options.noise_std,
            seed: options.seed,
            noise: None,
        },
        batch_size: options.batch_size,
        small: size,
        native_guide: options.native_guide,
        batch: Vec::with_capacity(options.batch_size),
        natives: VecDeque::new(),
        stabilizer: if options.temporal_stabilize {
            Some(RegeneratedStabilizer::<T>::new(RegenStabilizeOptions::default())?)
        } else {
            None
        },
        metrics: Metrics::default(),
    };
    let outcome = drive(regenerator, source, sampling, feed, &mut encoder);
    let metrics = match outcome {
        Ok(metrics) if metrics.frames >= 2 => metrics,
        Ok(_) => {
            encoder.abort();
            return Err(VideoError::value(
                "The selected clip produced fewer than two frames",
            ));
        }
        Err(error) => {
            encoder.abort();
            return Err(error);
        }
    };
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
    let peak = if high_depth { 65535.0 } else { 255.0 };
    Ok(metrics.finish(effective_fps, (out_w, out_h), peak))
}

fn drive<T: VaePixel>(
    mut regenerator: Regenerator<'_, T>,
    source: &Path,
    sampling: SamplingOptions,
    feed: FrameFeed<'_, T>,
    encoder: &mut RawVideoEncoder,
) -> Result<Metrics> {
    match feed {
        FrameFeed::Decode => {
            for frame in SampledFrames::<T>::open_depth(source, sampling)? {
                regenerator.push(frame?, encoder)?;
            }
        }
        FrameFeed::Stream(frames) => {
            let mut sampler = FrameSampler::new(sampling);
            for frame in frames {
                let frame = frame?;
                match sampler.next_wanted() {
                    None => break,
                    Some(false) => {}
                    Some(true) => regenerator.push(sampler.fit(frame)?, encoder)?,
                }
            }
        }
    }
    regenerator.finish(encoder)
}

/// Profile dispatch when the neural backend is absent.
pub fn require_profile_backend(neural: &NeuralBackends, profile: &str) -> Result<()> {
    match profile {
        p if WAN_EXPERIMENTAL_PROFILES.contains(&p) && neural.wan.is_none() => {
            Err(unavailable("The Wan 2.1 profile"))
        }
        "seedance-experimental" if neural.vae.is_none() => Err(unavailable("The Seedance profile")),
        "synthid" if neural.vae.is_none() => Err(unavailable("Video SynthID regeneration")),
        _ => Ok(()),
    }
}

/// Resize helper shared by the guide paths.
pub fn area_resize<T: Resizable>(frame: &Mat<T>, size: (usize, usize)) -> Result<Mat<T>> {
    Ok(resize(frame, size.0, size.1, Interpolation::Area)?)
}
