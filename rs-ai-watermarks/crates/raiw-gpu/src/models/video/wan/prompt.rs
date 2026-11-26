//! The default-prompt UMT5-XXL embedding behind every `wan21-*` run.
//!
//! UMT5-XXL never loads in Rust. `video_wan.load_video_wan_runtime` encodes
//! `WAN21_DEFAULT_PROMPT` on CPU once and `torch.save`s the `[1, 512, 4096]` embedding to
//! `model_cache_dir()/prompt-embeddings/<sha256(key)[:32]>.pt`, keyed by cache version,
//! `model@revision`, `wan21_v2v_prompt_embeds_v1` and the prompt; this module derives the same
//! path, optionally checks a pinned SHA-256 of the file, and reads the tensor.

use std::path::{Path, PathBuf};

use raiw_core::{sha256_bytes, sha256_file};

use crate::error::{GpuError, Result};
use crate::weights::StateDict;

use super::{MODEL_REPOSITORY, MODEL_REVISION};

/// `WAN21_DEFAULT_PROMPT`.
pub const DEFAULT_PROMPT: &str =
    "High fidelity natural video, sharp details, accurate motion, clean cinematic lighting.";
/// `_PROMPT_CACHE_VERSION`, `_PROMPT_CACHE_DIRNAME`, `_WAN21_PROMPT_OUTPUT_PARAMS`.
const CACHE_VERSION: u32 = 1;
const CACHE_DIR: &str = "prompt-embeddings";
const OUTPUT_PARAMS: [&str; 1] = ["wan21_v2v_prompt_embeds_v1"];
/// UMT5 `max_sequence_length` and hidden size.
pub const TEXT_TOKENS: usize = 512;
pub const TEXT_DIM: usize = 4096;

/// A `[512, 4096]` prompt embedding (zero rows past the prompt's tokens, as diffusers pads).
#[derive(Debug, Clone, PartialEq)]
pub struct PromptEmbedding {
    values: Vec<f32>,
}

impl PromptEmbedding {
    /// `_prompt_cache_path(f"{model}@{revision}", output_params, prompt)` file name.
    pub fn cache_file_name(prompt: &str) -> String {
        let model = format!("{MODEL_REPOSITORY}@{MODEL_REVISION}");
        let version = CACHE_VERSION.to_string();
        let mut parts = vec![version.as_str(), model.as_str()];
        parts.extend(OUTPUT_PARAMS);
        parts.push(prompt);
        let digest = sha256_bytes(parts.join("\x1f").as_bytes());
        format!("{}.pt", &digest[..32])
    }

    /// Where the Python package stores the default-prompt embedding.
    pub fn cache_path() -> PathBuf {
        raiw_core::model_cache_dir()
            .join(CACHE_DIR)
            .join(Self::cache_file_name(DEFAULT_PROMPT))
    }

    /// Read the cached default-prompt embedding, failing with instructions when it is absent.
    pub fn load_cached(expected_sha256: Option<&str>) -> Result<Self> {
        let path = Self::cache_path();
        if !path.is_file() {
            return Err(GpuError::MissingPromptCache(path));
        }
        Self::from_file(&path, expected_sha256)
    }

    /// Read a `torch.save`d embedding, verifying its SHA-256 when one is pinned.
    pub fn from_file(path: &Path, expected_sha256: Option<&str>) -> Result<Self> {
        if let Some(expected) = expected_sha256 {
            let actual = sha256_file(path).map_err(|err| GpuError::io(path, err))?;
            if !actual.eq_ignore_ascii_case(expected) {
                return Err(GpuError::DigestMismatch {
                    filename: path.display().to_string(),
                    actual,
                    expected: expected.to_owned(),
                });
            }
        }
        let mut state = StateDict::from_torch_zip(path, None, &[], |_| true)?;
        let tensor = state.take_any("")?;
        state.finish()?;
        if tensor.shape != [1, TEXT_TOKENS, TEXT_DIM] && tensor.shape != [TEXT_TOKENS, TEXT_DIM] {
            return Err(GpuError::ShapeMismatch {
                name: path.display().to_string(),
                actual: tensor.shape,
                expected: vec![1, TEXT_TOKENS, TEXT_DIM],
            });
        }
        Self::from_values(tensor.data)
    }

    /// Wrap a row-major `[512, 4096]` buffer (tests, custom prompts encoded elsewhere).
    pub fn from_values(values: Vec<f32>) -> Result<Self> {
        if values.len() != TEXT_TOKENS * TEXT_DIM {
            return Err(GpuError::Input(format!(
                "prompt embedding holds {} values, expected {TEXT_TOKENS}x{TEXT_DIM}",
                values.len()
            )));
        }
        if !values.iter().all(|v| v.is_finite()) {
            return Err(GpuError::NonFinite);
        }
        Ok(Self { values })
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }
}
