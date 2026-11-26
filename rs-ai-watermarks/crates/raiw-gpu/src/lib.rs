//! Burn-backed neural runtime shared by audio, video, pixel and regeneration models.
//!
//! - [`device`]: Python-style device selection (`auto`/`cpu`/`cuda`/`mps`) over wgpu
//!   (Vulkan, Metal, DX12), native CUDA/ROCm, and the pure-Rust CPU fallback (`burn-flex`).
//! - [`hub`]: Hugging Face hub cache resolution, pinned downloads, and wheel-bundled assets.
//! - [`weights`]: PyTorch/safetensors loading through burn-store with weight-norm folding.
//! - [`layers`]: shared building blocks (convolutions, Snake, LSTM, SEANet, vector quantizers).
//! - [`dsp`]: host STFT/ISTFT and polyphase resampling matching torch/scipy.
//! - [`models`]: the ported architectures behind plain `f32` buffer APIs.
//!
//! ```no_run
//! use raiw_gpu::{ComputeDevice, DeviceRequest};
//! use raiw_gpu::models::audio::AudioSealDetector;
//!
//! let device = ComputeDevice::select(DeviceRequest::Auto)?;
//! let detector = AudioSealDetector::load(&device)?;
//! let reading = detector.detect(&vec![0.0; 16_000], 16_000)?;
//! println!("{} on {device}: {}", reading.label(), reading.detected());
//! # Ok::<(), raiw_gpu::GpuError>(())
//! ```

pub mod device;
pub mod dsp;
pub mod error;
pub mod hub;
pub mod io;
#[cfg(feature = "gpu-runtime")]
pub mod kernels;
pub mod layers;
pub mod models;
pub mod rng;
mod torch_pickle;
pub mod weights;

pub use burn;
pub use device::{Backend, ComputeDevice, DeviceRequest};
pub use error::{GpuError, Result};
pub use hub::{ArchiveMember, Fetch, Hub, HubFile};
pub use weights::{HostTensor, StateDict};
