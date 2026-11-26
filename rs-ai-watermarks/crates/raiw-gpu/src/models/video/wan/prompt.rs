//! The default-prompt UMT5-XXL embedding behind every `wan21-*` run.
//!
//! UMT5-XXL never loads in Rust. `video_wan.load_video_wan_runtime` encodes
//! `WAN21_DEFAULT_PROMPT` on CPU once and `torch.save`s the `[1, 226, 4096]` embedding to
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
/// Longest UMT5 sequence (`max_sequence_length` of `WanPipeline.__call__`) and hidden size.
///
/// The cache Python writes holds 226 rows, `encode_prompt`'s own default: those zero padding
/// rows reach cross-attention as real (bias-only) keys, so the row count is kept as stored.
pub const TEXT_TOKENS: usize = 512;
pub const TEXT_DIM: usize = 4096;

/// A `[tokens, 4096]` prompt embedding (zero rows past the prompt's tokens, as diffusers pads).
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
        let tokens = match tensor.shape.as_slice() {
            [1, tokens, TEXT_DIM] | [tokens, TEXT_DIM] if (1..=TEXT_TOKENS).contains(tokens) => *tokens,
            _ => {
                return Err(GpuError::ShapeMismatch {
                    name: path.display().to_string(),
                    actual: tensor.shape,
                    expected: vec![1, TEXT_TOKENS, TEXT_DIM],
                });
            }
        };
        debug_assert_eq!(tensor.data.len(), tokens * TEXT_DIM);
        Self::from_values(tensor.data)
    }

    /// Wrap a row-major `[tokens, 4096]` buffer, `tokens <= 512` (tests, prompts encoded
    /// elsewhere).
    pub fn from_values(values: Vec<f32>) -> Result<Self> {
        let tokens = values.len() / TEXT_DIM;
        if !values.len().is_multiple_of(TEXT_DIM) || !(1..=TEXT_TOKENS).contains(&tokens) {
            return Err(GpuError::Input(format!(
                "prompt embedding holds {} values, expected [1..={TEXT_TOKENS}, {TEXT_DIM}]",
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

    /// Text tokens (rows), padding included: 226 in Python's cache.
    pub fn tokens(&self) -> usize {
        self.values.len() / TEXT_DIM
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_the_stored_token_count() {
        // `encode_prompt` pads to 226 rows; `WanPipeline.__call__` to 512
        for tokens in [1, 226, TEXT_TOKENS] {
            let prompt = PromptEmbedding::from_values(vec![0.0; tokens * TEXT_DIM]).unwrap();
            assert_eq!(prompt.tokens(), tokens);
        }
        assert!(PromptEmbedding::from_values(vec![0.0; (TEXT_TOKENS + 1) * TEXT_DIM]).is_err());
        assert!(PromptEmbedding::from_values(vec![0.0; TEXT_DIM + 1]).is_err());
        assert!(PromptEmbedding::from_values(Vec::new()).is_err());
    }

    /// The cache `video_wan.load_video_wan_runtime` wrote on this machine, when present.
    #[test]
    fn reads_the_python_cache() {
        let path = PromptEmbedding::cache_path();
        if !path.is_file() {
            return;
        }
        let prompt = PromptEmbedding::load_cached(None).unwrap();
        assert_eq!(prompt.tokens(), 226);
        let row = |t: usize| &prompt.values()[t * TEXT_DIM..(t + 1) * TEXT_DIM];
        assert!(
            row(0).iter().any(|v| *v != 0.0),
            "the prompt's first token is encoded"
        );
        assert!(row(225).iter().all(|v| *v == 0.0), "padding rows are zero");
    }
}
