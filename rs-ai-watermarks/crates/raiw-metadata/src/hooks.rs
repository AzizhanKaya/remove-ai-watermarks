//! The metadata-level rules the container walkers call up into.

use raiw_container::MetadataHooks;

use crate::audio_creation::creation_tool_bytes;
use crate::exif_ai::ai_exif_values;
use crate::rules::app::{any, app_evidence_bytes};
use crate::rules::generator::names_ai_generator;
use crate::rules::markers::payload_carries_ai_metadata;
use crate::rules::tc260::parse_tc260_aigc_json;

/// The real [`MetadataHooks`]: every container walk in this crate goes through it.
#[derive(Debug, Clone, Copy, Default)]
pub struct Hooks;

impl MetadataHooks for Hooks {
    fn is_tc260_label(&self, value: &[u8]) -> bool {
        parse_tc260_aigc_json(value).is_some()
    }

    fn names_ai_generator(&self, payload: &[u8]) -> bool {
        names_ai_generator(payload)
    }

    fn is_app_export(&self, value: &[u8]) -> bool {
        any(app_evidence_bytes(value))
    }

    fn is_audio_creation(&self, value: &[u8]) -> bool {
        creation_tool_bytes(value).is_some()
    }

    fn carries_ai_metadata(&self, raw: &[u8]) -> bool {
        payload_carries_ai_metadata(raw)
    }

    fn ai_exif_values(&self, exif: &[u8]) -> Vec<Vec<u8>> {
        ai_exif_values(exif)
    }
}
