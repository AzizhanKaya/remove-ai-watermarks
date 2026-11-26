//! [`BurnWan`]: `load_video_wan_runtime` over `raiw_gpu::models::video::wan::WanPipeline`.
//!
//! wgpu cannot report adapter memory, so the 8 GiB `auto` rule and the 16 GiB window rule treat
//! the accelerator as small: `auto` never routes to Wan, and explicit Wan profiles use the
//! compact 81-frame window. Wan 2.2 TI2V 5B (the Rust-only `wan22-*` profiles) runs int8
//! blocks swapped with its VAE, which is what fits an 8 GB card.

use std::sync::{Arc, Mutex, PoisonError};

use raiw_cv::Mat;
use raiw_gpu::hub::{Fetch, Hub};
use raiw_gpu::models::video::wan::{self, WanModel};

use super::LazyDevice;
use crate::error::{Result, VideoError};
use crate::wan::{
    WAN21_MODEL_ID, WAN21_MODEL_REVISION, WAN21_SPATIAL_MULTIPLE, WAN22_MODEL_ID, WAN22_MODEL_REVISION,
    WAN22_SPATIAL_MULTIPLE, WanCall, WanPipeline,
};

/// One pinned Wan checkpoint, loaded on first use.
#[derive(Debug)]
pub struct BurnWan {
    device: Arc<LazyDevice>,
    model: &'static WanModel,
    names: (&'static str, &'static str),
    spatial_multiple: usize,
    wan22: bool,
    pipeline: Mutex<Option<wan::WanPipeline>>,
}

impl BurnWan {
    /// Wan 2.1 T2V 1.3B (`wan21-*`).
    pub fn new(device: Arc<LazyDevice>) -> Self {
        Self {
            device,
            model: &wan::WAN21_T2V_1_3B,
            names: (WAN21_MODEL_ID, WAN21_MODEL_REVISION),
            spatial_multiple: WAN21_SPATIAL_MULTIPLE,
            wan22: false,
            pipeline: Mutex::new(None),
        }
    }

    /// Wan 2.2 TI2V 5B (`wan22-*`).
    pub fn wan22(device: Arc<LazyDevice>) -> Self {
        Self {
            device,
            model: &wan::WAN22_TI2V_5B,
            names: (WAN22_MODEL_ID, WAN22_MODEL_REVISION),
            spatial_multiple: WAN22_SPATIAL_MULTIPLE,
            wan22: true,
            pipeline: Mutex::new(None),
        }
    }

    fn label(&self) -> &'static str {
        if self.wan22 { "Wan 2.2" } else { "Wan 2.1" }
    }

    /// Drop the loaded weights (`release_video_models`).
    pub fn release(&self) {
        self.pipeline
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
    }
}

impl WanPipeline for BurnWan {
    fn model(&self) -> &str {
        self.names.0
    }

    fn revision(&self) -> &str {
        self.names.1
    }

    fn spatial_multiple(&self) -> usize {
        self.spatial_multiple
    }

    fn resolve_device(&self, requested: &str) -> Result<String> {
        let label = self.label();
        if !matches!(requested, "auto" | "cuda" | "mps") {
            return Err(VideoError::value(format!(
                "{label} 3D video diffusion requires device='auto', 'cuda', or 'mps'"
            )));
        }
        if !self.device.get()?.is_gpu() {
            return Err(VideoError::runtime(format!(
                "{label} 3D video diffusion requires a GPU adapter (or explicit device='mps')"
            )));
        }
        Ok(if requested == "mps" { "mps" } else { "cuda" }.to_owned())
    }

    fn vram_below(&self, _limit: u64) -> bool {
        true
    }

    fn generate(&self, frames: &[Mat<f32>], call: &WanCall<'_>) -> Result<Vec<Mat<f32>>> {
        let (height, width) = (call.height, call.width);
        let label = self.label();
        let plane = height * width;
        let mut nchw = Vec::with_capacity(frames.len() * plane * 3);
        for frame in frames {
            if frame.size() != (width, height) || frame.channels() != 3 {
                return Err(VideoError::value(format!(
                    "{label} windows need matching RGB frames"
                )));
            }
            for c in 0..3 {
                nchw.extend(frame.data().iter().skip(c).step_by(3).map(|v| 2.0 * v - 1.0));
            }
        }
        let mut slot = self.pipeline.lock().unwrap_or_else(PoisonError::into_inner);
        if slot.is_none() {
            let device = self.device.get()?;
            let residency = self.model.residency_for(device);
            tracing::info!(
                "Loading pinned {label} {}@{} on {device} ({residency:?})",
                self.names.0,
                self.names.1
            );
            *slot = Some(wan::WanPipeline::load_model(
                &Hub::from_env(),
                Fetch::Download,
                self.model,
                device,
                residency,
                None,
            )?);
        }
        let pipeline = slot
            .as_mut()
            .ok_or_else(|| VideoError::runtime(format!("{label} pipeline failed to load")))?;
        let decoded = pipeline.regenerate_chunk(
            &nchw,
            (frames.len(), height, width),
            call.strength,
            call.steps as usize,
            call.seed,
        )?;
        if decoded.len() != frames.len() * plane * 3 {
            return Err(VideoError::runtime(format!(
                "{label} returned an unexpected sample count"
            )));
        }
        decoded
            .chunks_exact(plane * 3)
            .map(|frame| {
                let mut data = Vec::with_capacity(plane * 3);
                for i in 0..plane {
                    data.extend(
                        [frame[i], frame[plane + i], frame[2 * plane + i]]
                            .map(|x| (x / 2.0 + 0.5).clamp(0.0, 1.0)),
                    );
                }
                Ok(Mat::from_vec(width, height, 3, data)?)
            })
            .collect()
    }
}
