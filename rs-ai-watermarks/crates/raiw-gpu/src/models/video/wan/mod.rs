//! Wan 2.1 T2V 1.3B video-to-video regeneration (`video_wan.py`, the `wan21-*` profiles), plus
//! the Wan 2.2 TI2V 5B models it generalizes to ([`config`], `docs/wan22-port-plan.md`).
//!
//! - [`vae3d`]: `AutoencoderKLWan` with causal time caching and Wan's own spatial tiling.
//! - [`transformer`]: `WanTransformer3DModel` (patchify, 3-D RoPE self-attention, text
//!   cross-attention, adaLN), with blocks resident or streamed from the checkpoint.
//! - [`scheduler`]: `UniPCMultistepScheduler` with flow sigmas, on host buffers.
//! - [`prompt`]: the default-prompt UMT5 embedding the Python package caches on disk
//!   (UMT5-XXL never loads here; `tools/goldens/gpu_wan_prompt_cache.py` produces the file).
//! - [`pipeline`]: `WanVideoToVideoPipeline` for one window of frames.
//!
//! Everything runs in float32 (Python runs the transformer in bfloat16 on CUDA, float16 on
//! MPS); parity is pinned against the diffusers modules in float32 on CPU.

pub mod attention;
pub mod config;
#[cfg(feature = "gpu-runtime")]
pub mod fused;
pub mod pipeline;
pub mod prompt;
pub mod rope;
pub mod scheduler;
pub mod transformer;
pub mod vae3d;

use crate::hub::HubFile;

pub use config::{TransformerConfig, VaeConfig};
#[cfg(feature = "gpu-runtime")]
pub use fused::Precision;
pub use pipeline::{WAN21_T2V_1_3B, WAN22_TI2V_5B, WanModel, WanPipeline};
pub use prompt::PromptEmbedding;
pub use scheduler::UniPcScheduler;
pub use transformer::{
    BlockResidency, TextContext, TransformerFiles, WAN21_TRANSFORMER, WAN22_TRANSFORMER, WanTransformer,
};
pub use vae3d::{VaeFiles, WAN21_VAE, WAN22_VAE, WanVae};

pub const MODEL_REPOSITORY: &str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers";
pub const MODEL_REVISION: &str = "0fad780a534b6463e45facd96134c9f345acfa5b";

/// `WAN21_FLOW_SHIFT`, `WAN21_INFERENCE_STEPS`, `WAN21_GUIDANCE_SCALE` (CFG off).
pub const FLOW_SHIFT: f64 = 3.0;
pub const INFERENCE_STEPS: usize = 20;
pub const GUIDANCE_SCALE: f64 = 1.0;
/// Calibrated denoising strengths of the `wan21-synthid` and `wan21-seedance` profiles.
pub const SYNTHID_STRENGTH: f64 = 0.38;
pub const SEEDANCE_STRENGTH: f64 = 0.12;
/// Frames are padded to this spatial multiple before encoding (`WAN21_SPATIAL_MULTIPLE`).
pub const SPATIAL_MULTIPLE: usize = 16;
/// Temporal VAE compression: `1 + 4k` frames encode to `1 + k` latent frames.
pub const TEMPORAL_FACTOR: usize = 4;
/// Spatial VAE compression.
pub const SPATIAL_FACTOR: usize = 8;
pub const LATENT_CHANNELS: usize = 16;

const fn pinned(filename: &'static str, sha256: &'static str) -> HubFile<'static> {
    HubFile::new(MODEL_REPOSITORY, MODEL_REVISION, filename).with_sha256(sha256)
}

pub const VAE_CONFIG: HubFile<'static> = pinned(
    "vae/config.json",
    "f0c1cc1d7decb5badc384f54691746a27a9aeff49f7ebca974e583389342d527",
);
pub const VAE_CHECKPOINT: HubFile<'static> = pinned(
    "vae/diffusion_pytorch_model.safetensors",
    "d6e524b3fffede1787a74e81b30976dce5400c4439ba64222168e607ed19e793",
);
pub const TRANSFORMER_CONFIG: HubFile<'static> = pinned(
    "transformer/config.json",
    "0b093fa072e9ff28763febe9b964ee582f566733a6d6709deb9dfba1bde16b81",
);
pub const TRANSFORMER_SHARDS: [HubFile<'static>; 2] = [
    pinned(
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        "6d011927dbd2cc8afe53d57abab04a8fd86f615d83324770d985fb058ece3a24",
    ),
    pinned(
        "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
        "b92ec2309b1f239af6f746431815a881afcc938abb26a4f08d9a2fd6c892f872",
    ),
];
pub const SCHEDULER_CONFIG: HubFile<'static> = pinned(
    "scheduler/scheduler_config.json",
    "3fed2abbd9bbc301a74db01947198057ec5049808910dccab320925bf27bea6e",
);

// Wan 2.2 TI2V 5B (Rust-only experimental; Python has no Wan 2.2 path)
pub const WAN22_REPOSITORY: &str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers";
pub const WAN22_REVISION: &str = "b8fff7315c768468a5333511427288870b2e9635";
/// `scheduler_config.json` default of the 5B checkpoint.
pub const WAN22_FLOW_SHIFT: f64 = 5.0;

const fn pinned22(filename: &'static str, sha256: &'static str) -> HubFile<'static> {
    HubFile::new(WAN22_REPOSITORY, WAN22_REVISION, filename).with_sha256(sha256)
}

pub const WAN22_VAE_CONFIG: HubFile<'static> = pinned22(
    "vae/config.json",
    "d996c340fe9a7df5d7371f76a7d8d6956f6c98256080074d8434fa5eeac11360",
);
pub const WAN22_VAE_CHECKPOINT: HubFile<'static> = pinned22(
    "vae/diffusion_pytorch_model.safetensors",
    "62cd18f19438e35b32ac63020e2852f566e9b02f46b6cdbd87972a356e3c6f4b",
);
pub const WAN22_TRANSFORMER_CONFIG: HubFile<'static> = pinned22(
    "transformer/config.json",
    "dc00d9866e72cf77db6b531aaa33be4dc7148fef9338442bd0ae9181f7075e9b",
);
pub const WAN22_TRANSFORMER_SHARDS: [HubFile<'static>; 5] = [
    pinned22(
        "transformer/diffusion_pytorch_model-00001-of-00005.safetensors",
        "511bec832a201caa410d09c5ce7dbbf8ad2708c345d82038f684fc74cce982be",
    ),
    pinned22(
        "transformer/diffusion_pytorch_model-00002-of-00005.safetensors",
        "7c42724912b1911429125dc50c0e9a49ccbada5a601b657d4ed2e15e7597c193",
    ),
    pinned22(
        "transformer/diffusion_pytorch_model-00003-of-00005.safetensors",
        "e9c3d0c76de786566382f8258101fea973ae37681c6e9fe0e5fe1fb93b806424",
    ),
    pinned22(
        "transformer/diffusion_pytorch_model-00004-of-00005.safetensors",
        "a331121771790939678db6f585553fd5184609f7d02593c699a4d241b0d834c5",
    ),
    pinned22(
        "transformer/diffusion_pytorch_model-00005-of-00005.safetensors",
        "78b655685c47efdb2349f36826bb101264e9f212a16325d584aeb5f53c88e719",
    ),
];
