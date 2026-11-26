//! VideoSeal / PixelSeal / ChunkySeal detectors over `raiw_gpu::models::video::VideoMessageDetector`.

use std::sync::{Arc, Mutex, PoisonError};

use raiw_gpu::models::video::{Scheme, VideoMessageDetector};

use super::LazyDevice;
use crate::error::Result;
use crate::messages::{MessageScheme, VideoMessageModel, VideoMessageModels};

/// One loaded detector.
#[derive(Debug)]
pub struct BurnMessageModel {
    scheme: MessageScheme,
    detector: VideoMessageDetector,
}

impl VideoMessageModel for BurnMessageModel {
    fn scheme(&self) -> MessageScheme {
        self.scheme
    }

    fn predict(&self, frames: &[f32], count: usize, height: usize, width: usize) -> Result<Vec<Vec<f32>>> {
        Ok(self.detector.predict(frames, count, height, width)?)
    }
}

/// Verified public assets loaded on demand; only the most recent detector stays resident.
#[derive(Debug)]
pub struct BurnMessageModels {
    device: Arc<LazyDevice>,
    resident: Mutex<Option<Arc<BurnMessageModel>>>,
}

impl BurnMessageModels {
    pub fn new(device: Arc<LazyDevice>) -> Self {
        Self {
            device,
            resident: Mutex::new(None),
        }
    }

    /// Release the resident detector (`clear_model_cache`).
    pub fn release(&self) {
        self.resident
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
    }
}

impl VideoMessageModels for BurnMessageModels {
    fn load(&self, scheme: MessageScheme) -> Result<Arc<dyn VideoMessageModel>> {
        let mut resident = self.resident.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(model) = resident.as_ref().filter(|m| m.scheme == scheme) {
            return Ok(Arc::clone(model) as Arc<dyn VideoMessageModel>);
        }
        resident.take();
        let detector = VideoMessageDetector::load(
            match scheme {
                MessageScheme::VideoSeal => Scheme::VideoSeal,
                MessageScheme::PixelSeal => Scheme::PixelSeal,
                MessageScheme::ChunkySeal => Scheme::ChunkySeal,
            },
            self.device.get()?,
        )?;
        tracing::info!("Loaded pinned public video decoder: scheme={scheme}");
        let model = Arc::new(BurnMessageModel { scheme, detector });
        *resident = Some(Arc::clone(&model));
        Ok(model)
    }
}
