//! `raiw-gpu` adapters behind the neural seams (the `gpu` feature).
//!
//! Devices and models load on first use, so building a runtime never probes an adapter or
//! reads weights; `auto` falls back to the CPU backend when no GPU adapter works.

mod messages;
mod vae;
mod wan;

use std::sync::{Arc, OnceLock};

use raiw_gpu::{ComputeDevice, DeviceRequest};

pub use messages::{BurnMessageModel, BurnMessageModels};
pub use vae::SdVaeRuntime;
pub use wan::BurnWan;

use crate::error::{Result, VideoError};
use crate::invisible::{DEFAULT_VIDEO_SYNTHID_VAE, NeuralBackends};

/// `pick_device` over burn: the request is validated now and resolved on first use.
#[derive(Debug)]
pub struct LazyDevice {
    request: DeviceRequest,
    resolved: OnceLock<std::result::Result<ComputeDevice, String>>,
}

impl LazyDevice {
    pub fn new(device: &str) -> Result<Self> {
        let request = match device {
            "auto" => DeviceRequest::Auto,
            "cuda" => DeviceRequest::Cuda,
            "mps" => DeviceRequest::Mps,
            "cpu" => DeviceRequest::Cpu,
            _ => return Err(VideoError::value("device must be auto, cuda, mps, or cpu")),
        };
        Ok(Self {
            request,
            resolved: OnceLock::new(),
        })
    }

    pub fn request(&self) -> DeviceRequest {
        self.request
    }

    /// The probed device; `cuda`/`mps` fail when no matching adapter works.
    pub fn get(&self) -> Result<&ComputeDevice> {
        self.resolved
            .get_or_init(|| {
                ComputeDevice::select(self.request)
                    .inspect(|d| tracing::info!("Video neural runtime on {d}"))
                    .map_err(|e| e.to_string())
            })
            .as_ref()
            .map_err(|reason| {
                VideoError::runtime(format!(
                    "{} was requested but is not available: {reason}",
                    self.request.as_str().to_uppercase()
                ))
            })
    }
}

/// The burn-backed neural runtimes for `device` (`auto`, `cuda`, `mps` or `cpu`).
pub fn backends(device: &str) -> Result<NeuralBackends> {
    let device = Arc::new(LazyDevice::new(device)?);
    let requested = device.request().as_str();
    Ok(NeuralBackends {
        vae: Some(Arc::new(SdVaeRuntime::new(
            DEFAULT_VIDEO_SYNTHID_VAE,
            requested,
            Arc::clone(&device),
        )?)),
        wan: Some(Arc::new(BurnWan::new(Arc::clone(&device)))),
        messages: Some(Arc::new(BurnMessageModels::new(device))),
    })
}
