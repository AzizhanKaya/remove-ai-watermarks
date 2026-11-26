//! `WanTransformer3DModel` hyper-parameters, read from a checkpoint's `transformer/config.json`.
//!
//! Wan 2.1 T2V 1.3B and Wan 2.2 TI2V 5B share the architecture and differ only in widths:
//!
//! | | 2.1 1.3B | 2.2 5B |
//! | --- | --- | --- |
//! | heads x head dim | 12 x 128 | 24 x 128 |
//! | feed-forward | 8 960 | 14 336 |
//! | blocks | 30 | 30 |
//! | latent channels | 16 | 48 |

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

#[cfg(test)]
mod tests {
    use super::*;

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
