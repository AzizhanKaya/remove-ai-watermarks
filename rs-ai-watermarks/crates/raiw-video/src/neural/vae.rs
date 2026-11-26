//! [`SdVaeRuntime`]: `load_video_vae_runtime` over `raiw_gpu::models::video::SdVae`.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use raiw_cv::Mat;
use raiw_gpu::hub::{Fetch, Hub};
use raiw_gpu::models::video::SdVae;
use raiw_gpu::models::video::vae::{CONFIG_FILE, LATENT_CHANNELS, SCALING_FACTOR};

use super::LazyDevice;
use crate::error::{Result, VideoError};
use crate::invisible::{
    DEFAULT_VIDEO_SYNTHID_VAE, Latents, VIDEO_SYNTHID_LATENT_MULTIPLE, VIDEO_SYNTHID_VAE_SCALING_FACTOR,
    VideoVae,
};

/// The pinned `stabilityai/sd-vae-ft-mse` runtime, loaded through the latent-scaling gate on first use.
#[derive(Debug)]
pub struct SdVaeRuntime {
    model: String,
    requested: String,
    device: Arc<LazyDevice>,
    vae: Mutex<Option<SdVae>>,
    tiling: Mutex<bool>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// `AutoencoderKL`'s registered default when a config omits `scaling_factor`.
const AUTOENCODER_KL_DEFAULT_SCALING_FACTOR: f64 = 0.18215;

/// The gate of `load_video_vae_runtime`: the certified profile is defined against 0.18215.
fn gated_scaling_factor(hub: &Hub) -> Result<f64> {
    let config = hub.resolve(&CONFIG_FILE, Fetch::Download)?;
    let parsed: serde_json::Value = serde_json::from_slice(&std::fs::read(&config)?)?;
    let factor = match parsed.get("scaling_factor") {
        None => AUTOENCODER_KL_DEFAULT_SCALING_FACTOR,
        Some(value) => value
            .as_f64()
            .ok_or_else(|| VideoError::runtime("Video VAE config has a non-numeric latent scaling factor"))?,
    };
    tracing::info!("Latent scaling factor {factor:.5}");
    if factor != VIDEO_SYNTHID_VAE_SCALING_FACTOR || SCALING_FACTOR != VIDEO_SYNTHID_VAE_SCALING_FACTOR as f32
    {
        return Err(VideoError::runtime(format!(
            "{DEFAULT_VIDEO_SYNTHID_VAE} loaded with latent scaling factor {factor}, but the certified profile is defined against {VIDEO_SYNTHID_VAE_SCALING_FACTOR}. The perturbation would be rescaled and the output would no longer match any certified row."
        )));
    }
    Ok(factor)
}

impl SdVaeRuntime {
    pub fn new(model: &str, requested: &str, device: Arc<LazyDevice>) -> Result<Self> {
        if model != DEFAULT_VIDEO_SYNTHID_VAE {
            return Err(VideoError::Unavailable(format!(
                "Only the pinned {DEFAULT_VIDEO_SYNTHID_VAE} video VAE is ported; {model} is unavailable"
            )));
        }
        Ok(Self {
            model: model.to_owned(),
            requested: requested.to_owned(),
            device,
            vae: Mutex::new(None),
            tiling: Mutex::new(false),
        })
    }

    /// Run `f` on the loaded model, loading it (and passing the gate) first if needed.
    fn with_vae<R>(&self, f: impl FnOnce(&SdVae) -> raiw_gpu::Result<R>) -> Result<R> {
        let mut slot = lock(&self.vae);
        if slot.is_none() {
            let hub = Hub::from_env();
            gated_scaling_factor(&hub)?;
            let device = self.device.get()?;
            tracing::info!("Loading {} on {device}", self.model);
            *slot = Some(SdVae::load_from_hub(&hub, Fetch::Download, device)?);
        }
        let vae = slot
            .as_mut()
            .ok_or_else(|| VideoError::runtime("Video VAE failed to load"))?;
        vae.set_tiling(*lock(&self.tiling));
        Ok(f(vae)?)
    }

    /// Drop the loaded weights (`release_video_models`).
    pub fn release(&self) {
        lock(&self.vae).take();
    }
}

fn hwc_to_nchw(planes: &[Mat<f32>]) -> Result<(Vec<f32>, usize, usize)> {
    let first = planes
        .first()
        .ok_or_else(|| VideoError::value("Video VAE batches must not be empty"))?;
    let (width, height) = first.size();
    let plane = width * height;
    let mut out = Vec::with_capacity(planes.len() * plane * 3);
    for frame in planes {
        if frame.size() != (width, height) || frame.channels() != 3 {
            return Err(VideoError::value("Video VAE batches need matching RGB frames"));
        }
        let data = frame.data();
        for c in 0..3 {
            out.extend(data.iter().skip(c).step_by(3));
        }
    }
    Ok((out, height, width))
}

fn nchw_to_hwc(samples: &[f32], frames: usize, height: usize, width: usize) -> Result<Vec<Mat<f32>>> {
    let plane = width * height;
    if samples.len() != frames * plane * 3 {
        return Err(VideoError::runtime(
            "Video VAE decoder returned an unexpected sample count",
        ));
    }
    samples
        .chunks_exact(plane * 3)
        .map(|frame| {
            let mut data = Vec::with_capacity(plane * 3);
            for i in 0..plane {
                data.extend([frame[i], frame[plane + i], frame[2 * plane + i]]);
            }
            Ok(Mat::from_vec(width, height, 3, data)?)
        })
        .collect()
}

impl VideoVae for SdVaeRuntime {
    fn model(&self) -> &str {
        &self.model
    }

    fn requested_device(&self) -> &str {
        &self.requested
    }

    fn scaling_factor(&self) -> f64 {
        VIDEO_SYNTHID_VAE_SCALING_FACTOR
    }

    fn encode(&self, rgb: &[Mat<f32>]) -> Result<Latents> {
        let (pixels, height, width) = hwc_to_nchw(rgb)?;
        let data = self.with_vae(|vae| vae.encode(&pixels, rgb.len(), height, width))?;
        let m = VIDEO_SYNTHID_LATENT_MULTIPLE;
        Ok(Latents {
            shape: [rgb.len(), LATENT_CHANNELS, height / m, width / m],
            data,
        })
    }

    fn decode(&self, latents: &Latents) -> Result<Vec<Mat<f32>>> {
        let [n, _, h, w] = latents.shape;
        let samples = self.with_vae(|vae| vae.decode(&latents.data, n, h, w))?;
        let m = VIDEO_SYNTHID_LATENT_MULTIPLE;
        nchw_to_hwc(&samples, n, h * m, w * m)
    }

    fn shared_noise(&self, shape: [usize; 3], seed: u64) -> Result<Vec<f32>> {
        if shape.contains(&0) {
            return Err(VideoError::value("Expected a positive CHW latent shape"));
        }
        raiw_gpu::rng::torch_randn(shape.iter().product(), seed)
            .ok_or_else(|| VideoError::value("Latent noise fields need at least 16 values"))
    }

    fn set_tiling(&self, enabled: bool) {
        *lock(&self.tiling) = enabled;
    }
}
