//! `WanTransformer3DModel` and `AutoencoderKLWan` hyper-parameters, read from a checkpoint's
//! `transformer/config.json` and `vae/config.json`.
//!
//! Wan 2.1 T2V 1.3B and Wan 2.2 TI2V 5B share the transformer and differ only in widths; the
//! 2.2 VAE adds a 2x2 patchify and residual down/up blocks for 16x spatial compression:
//!
//! | | 2.1 1.3B | 2.2 5B |
//! | --- | --- | --- |
//! | heads x head dim | 12 x 128 | 24 x 128 |
//! | feed-forward | 8 960 | 14 336 |
//! | blocks | 30 | 30 |
//! | latent channels | 16 | 48 |
//! | VAE encoder / decoder base | 96 / 96 | 160 / 256 |
//! | VAE spatial factor | 8 | 16 |

use std::path::Path;

use serde::Deserialize;

use crate::error::{GpuError, Result};

/// The fields of `transformer/config.json` the port depends on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct TransformerConfig {
    #[serde(rename = "num_attention_heads")]
    pub heads: usize,
    #[serde(rename = "attention_head_dim")]
    pub head_dim: usize,
    pub ffn_dim: usize,
    #[serde(rename = "num_layers")]
    pub layers: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub text_dim: usize,
    pub freq_dim: usize,
    pub patch_size: [usize; 3],
}

impl TransformerConfig {
    /// `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`.
    pub const WAN21_T2V_1_3B: Self = Self {
        heads: 12,
        head_dim: 128,
        ffn_dim: 8960,
        layers: 30,
        in_channels: 16,
        out_channels: 16,
        text_dim: 4096,
        freq_dim: 256,
        patch_size: [1, 2, 2],
    };

    /// Read and validate a `transformer/config.json`.
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|err| GpuError::io(path, err))?;
        let config: Self = serde_json::from_str(&text)
            .map_err(|err| GpuError::Config(format!("{}: {err}", path.display())))?;
        if config.patch_size != [1, 2, 2]
            || config.head_dim != 128
            || config.in_channels != config.out_channels
        {
            return Err(GpuError::Config(format!(
                "{}: unsupported Wan transformer geometry {config:?}",
                path.display()
            )));
        }
        Ok(config)
    }

    /// Model width, `heads * head_dim`.
    pub const fn dim(&self) -> usize {
        self.heads * self.head_dim
    }

    /// Spatial patch side (the time patch is 1).
    pub const fn patch(&self) -> usize {
        self.patch_size[2]
    }

    /// Features of one `1 x 2 x 2` latent patch.
    pub const fn patch_features(&self) -> usize {
        self.in_channels * self.patch() * self.patch()
    }
}

/// The fields of `vae/config.json` the port depends on.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct VaeConfig {
    pub base_dim: usize,
    pub decoder_base_dim: Option<usize>,
    pub z_dim: usize,
    pub dim_mult: Vec<usize>,
    pub num_res_blocks: usize,
    #[serde(rename = "temperal_downsample")]
    pub time_downsample: Vec<bool>,
    #[serde(default)]
    pub attn_scales: Vec<f64>,
    #[serde(default = "rgb")]
    pub in_channels: usize,
    #[serde(default = "rgb")]
    pub out_channels: usize,
    /// Wan 2.2: residual down/up blocks with `AvgDown3D` / `DupUp3D` shortcuts.
    #[serde(default)]
    pub is_residual: bool,
    pub patch_size: Option<usize>,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
}

const fn rgb() -> usize {
    3
}

impl VaeConfig {
    /// Read and validate a `vae/config.json`.
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|err| GpuError::io(path, err))?;
        let config: Self = serde_json::from_str(&text)
            .map_err(|err| GpuError::Config(format!("{}: {err}", path.display())))?;
        config
            .validate()
            .map_err(|reason| GpuError::Config(format!("{}: {reason}", path.display())))?;
        Ok(config)
    }

    fn validate(&self) -> std::result::Result<(), String> {
        let pixels = 3 * self.patch() * self.patch();
        if !self.attn_scales.is_empty() {
            return Err("encoder attention (attn_scales) is not ported".to_owned());
        }
        if self.dim_mult.len() < 2 || self.time_downsample.len() + 1 != self.dim_mult.len() {
            return Err(format!(
                "dim_mult {:?} does not match temperal_downsample {:?}",
                self.dim_mult, self.time_downsample
            ));
        }
        if (self.in_channels, self.out_channels) != (pixels, pixels) {
            return Err(format!(
                "expected {pixels} patchified RGB channels, got {} in / {} out",
                self.in_channels, self.out_channels
            ));
        }
        if self.latents_mean.len() != self.z_dim || self.latents_std.len() != self.z_dim {
            return Err(format!(
                "latent statistics must hold z_dim = {} values",
                self.z_dim
            ));
        }
        Ok(())
    }

    /// Spatial patchify side (1 for Wan 2.1).
    pub fn patch(&self) -> usize {
        self.patch_size.unwrap_or(1)
    }

    pub fn decoder_dim(&self) -> usize {
        self.decoder_base_dim.unwrap_or(self.base_dim)
    }

    /// Pixels per latent along each side: the patch times one halving per down level.
    pub fn spatial_factor(&self) -> usize {
        self.patch() << (self.dim_mult.len() - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vae_config(json: &serde_json::Value) -> Result<VaeConfig> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        std::fs::write(&path, json.to_string()).unwrap();
        VaeConfig::from_file(&path)
    }

    #[test]
    fn parses_both_vae_configs() {
        let mut wan21 = serde_json::json!({
            "_class_name": "AutoencoderKLWan", "attn_scales": [], "base_dim": 96,
            "dim_mult": [1, 2, 4, 4], "dropout": 0.0, "num_res_blocks": 2,
            "temperal_downsample": [false, true, true], "z_dim": 16,
            "latents_mean": vec![0.0; 16], "latents_std": vec![1.0; 16],
        });
        let config = vae_config(&wan21).unwrap();
        assert_eq!(
            (config.patch(), config.decoder_dim(), config.spatial_factor()),
            (1, 96, 8)
        );
        assert!(!config.is_residual);

        let wan22 = serde_json::json!({
            "base_dim": 160, "clip_output": false, "decoder_base_dim": 256, "dim_mult": [1, 2, 4, 4],
            "in_channels": 12, "is_residual": true, "num_res_blocks": 2, "out_channels": 12,
            "patch_size": 2, "scale_factor_spatial": 16, "temperal_downsample": [false, true, true],
            "z_dim": 48, "latents_mean": vec![0.0; 48], "latents_std": vec![1.0; 48],
        });
        let config = vae_config(&wan22).unwrap();
        assert_eq!(
            (config.patch(), config.decoder_dim(), config.spatial_factor()),
            (2, 256, 16)
        );

        wan21["attn_scales"] = serde_json::json!([1.0]);
        assert!(vae_config(&wan21).is_err());
    }

    #[test]
    fn parses_diffusers_configs() {
        let wan22 = r#"{"_class_name":"WanTransformer3DModel","added_kv_proj_dim":null,"attention_head_dim":128,
            "cross_attn_norm":true,"eps":1e-06,"ffn_dim":14336,"freq_dim":256,"image_dim":null,"in_channels":48,
            "num_attention_heads":24,"num_layers":30,"out_channels":48,"patch_size":[1,2,2],
            "pos_embed_seq_len":null,"qk_norm":"rms_norm_across_heads","rope_max_seq_len":1024,"text_dim":4096}"#;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        std::fs::write(&path, wan22).unwrap();
        let config = TransformerConfig::from_file(&path).unwrap();
        assert_eq!(
            (config.dim(), config.ffn_dim, config.in_channels),
            (3072, 14336, 48)
        );
        assert_eq!(config.patch_features(), 192);
    }
}
