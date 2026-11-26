//! Metadata key and value classifiers (`_is_ai_key`, `_is_ai_value`, standard keys).

use raiw_c2pa::registry::AI_GENERATOR_TOKENS;

use super::app::{any, app_evidence_str};

/// Lower-cased `AI_METADATA_KEYS` of `metadata.py`.
pub const AI_METADATA_KEYS: &[&str] = &[
    "parameters",
    "prompt",
    "negative_prompt",
    "workflow",
    "comfyui",
    "sd-metadata",
    "invokeai_metadata",
    "generation_data",
    "ai_metadata",
    "dream",
    "sd:prompt",
    "sd:negative_prompt",
    "sd:seed",
    "sd:steps",
    "sd:sampler",
    "sd:cfg_scale",
    "sd:model_hash",
    "c2pa",
    "c2pa_chunk",
    "software",
];

pub const AI_KEYWORDS: &[&str] = &[
    "stable_diffusion",
    "comfyui",
    "automatic1111",
    "invokeai",
    "midjourney",
    "dall-e",
    "dalle",
    "imagen",
    "synthid",
    "google_ai",
    "openai",
    "c2pa",
];

pub const STANDARD_METADATA_KEYS: &[&str] = &[
    "Author",
    "Title",
    "Description",
    "Copyright",
    "Creation Time",
    "Software",
    "Comment",
    "Disclaimer",
    "Source",
    "Warning",
];

/// Higgsfield's PNG job-id text key.
pub const HF_JOB_KEY: &str = "hf-job-id";

/// `_is_ai_key`.
pub fn is_ai_key(key: &str) -> bool {
    let lower = key.to_lowercase();
    AI_METADATA_KEYS.contains(&lower.as_str()) || AI_KEYWORDS.iter().any(|kw| lower.contains(kw))
}

/// Whether lower-cased text names a known AI generator token.
pub fn has_generator_token(lower: &str) -> bool {
    AI_GENERATOR_TOKENS.iter().any(|token| lower.contains(token))
}

/// `_is_ai_value`.
pub fn is_ai_value(value: &str) -> bool {
    any(app_evidence_str(value)) || has_generator_token(&value.to_lowercase())
}
