//! [`BurnWan`]: `load_video_wan_runtime` over `raiw_gpu::models::video::wan::WanPipeline`.
//!
//! wgpu cannot report adapter memory, so the 8 GiB `auto` rule and the 16 GiB window rule treat
//! the accelerator as small: `auto` never routes to Wan, and explicit Wan profiles use the
//! compact 81-frame window.

use std::sync::{Arc, Mutex, PoisonError};

use raiw_cv::Mat;
use raiw_gpu::hub::{Fetch, Hub};
use raiw_gpu::models::video::wan::{self, BlockResidency};

use super::LazyDevice;
use crate::error::{Result, VideoError};
use crate::wan::{WanCall, WanPipeline};

/// The pinned Wan 2.1 T2V 1.3B pipeline, loaded (blocks streamed) on first use.
#[derive(Debug)]
pub struct BurnWan {
    device: Arc<LazyDevice>,
    pipeline: Mutex<Option<wan::WanPipeline>>,
}

impl BurnWan {
    pub fn new(device: Arc<LazyDevice>) -> Self {
        Self {
            device,
            pipeline: Mutex::new(None),
        }
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
        wan::MODEL_REPOSITORY
    }

    fn revision(&self) -> &str {
        wan::MODEL_REVISION
    }

    fn resolve_device(&self, requested: &str) -> Result<String> {
        if !matches!(requested, "auto" | "cuda" | "mps") {
            return Err(VideoError::value(
                "Wan 2.1 3D video diffusion requires device='auto', 'cuda', or 'mps'",
            ));
        }
        if !self.device.get()?.is_gpu() {
            return Err(VideoError::runtime(
                "Wan 2.1 3D video diffusion requires a GPU adapter (or explicit device='mps')",
            ));
        }
        Ok(if requested == "mps" { "mps" } else { "cuda" }.to_owned())
    }

    fn vram_below(&self, _limit: u64) -> bool {
        true
    }

    fn generate(&self, frames: &[Mat<f32>], call: &WanCall<'_>) -> Result<Vec<Mat<f32>>> {
        let (height, width) = (call.height, call.width);
        let plane = height * width;
        let mut nchw = Vec::with_capacity(frames.len() * plane * 3);
        for frame in frames {
            if frame.size() != (width, height) || frame.channels() != 3 {
                return Err(VideoError::value("Wan 2.1 windows need matching RGB frames"));
            }
            for c in 0..3 {
                nchw.extend(frame.data().iter().skip(c).step_by(3).map(|v| 2.0 * v - 1.0));
            }
        }
        let mut slot = self.pipeline.lock().unwrap_or_else(PoisonError::into_inner);
        if slot.is_none() {
            let device = self.device.get()?;
            tracing::info!(
                "Loading pinned Wan 2.1 {}@{} on {device}",
                wan::MODEL_REPOSITORY,
                wan::MODEL_REVISION
            );
            *slot = Some(wan::WanPipeline::load(
                &Hub::from_env(),
                Fetch::Download,
                device,
                BlockResidency::for_device(device),
                None,
            )?);
        }
        let pipeline = slot
            .as_ref()
            .ok_or_else(|| VideoError::runtime("Wan 2.1 pipeline failed to load"))?;
        let decoded = pipeline.regenerate_chunk(
            &nchw,
            (frames.len(), height, width),
            call.strength,
            call.steps as usize,
            call.seed,
        )?;
        if decoded.len() != frames.len() * plane * 3 {
            return Err(VideoError::runtime("Wan 2.1 returned an unexpected sample count"));
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
