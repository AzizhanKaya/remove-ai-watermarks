//! Metadata-level AI provenance rules shared by detection, removal and the container hooks.

pub mod app;
pub mod generator;
pub mod html;
pub mod keys;
pub mod markers;
pub mod tc260;
pub mod xai;

pub use app::{AppEvidence, app_evidence_bytes, app_evidence_str};
pub use generator::{generator_from_metadata, names_ai_generator, structured_generator};
pub use keys::{STANDARD_METADATA_KEYS, is_ai_key, is_ai_value};
pub use markers::{
    c2pa_cloud_manifest_in, c2pa_marker_in, iptc_ai_system_in, payload_carries_ai_metadata, samsung_genai_in,
};
pub use tc260::{Tc260Label, aigc_label_from_metadata, parse_tc260_aigc_json};
pub use xai::{xai_signature_in_metadata, xai_signature_pair};
