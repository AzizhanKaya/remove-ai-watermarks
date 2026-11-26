//! AI-generation metadata: detection, lossless removal and portable records.

pub mod audio_creation;
pub mod c2pa_store;
pub mod cache;
pub mod deflate;
pub mod detect;
pub mod exif;
pub mod exif_ai;
pub mod forensic;
pub mod hooks;
pub mod pil;
pub mod py;
pub mod pyjson;
pub mod readability;
pub mod record;
pub mod report;
pub mod rules;
pub mod scan;
pub mod strip;
pub mod tc260_signature;
pub mod verify;

pub use detect::{
    AiMetadata, aigc_label, c2pa_cloud_manifest, exif_generator, get_ai_metadata, has_ai_metadata,
    huggingface_job, iptc_ai_system, samsung_genai, synthid_source, xai_signature,
};
pub use hooks::Hooks;
pub use readability::unreadable_reason;
pub use strip::{StripError, remove_ai_metadata};
pub use tc260_signature::{SignatureStatus, Tc260Signature, check_tc260_signature};
pub use verify::{StripOutcome, VerifyError, strip_and_verify, strip_written_output};
