//! Metadata-level predicates the walkers call up into, implemented by `raiw-metadata`.

/// The AI-marker rules that live above the container layer.
///
/// Each method mirrors one Python call site; the walkers invoke them in the same
/// order and with the same byte inputs as the reference implementation.
pub trait MetadataHooks {
    /// `metadata.parse_tc260_aigc_json(value) is not None`.
    fn is_tc260_label(&self, value: &[u8]) -> bool;

    /// `metadata.generator_from_metadata((), payload) is not None`.
    fn names_ai_generator(&self, payload: &[u8]) -> bool;

    /// `app_provenance_from_metadata(value) or app_generator_from_metadata(value)`.
    fn is_app_export(&self, value: &[u8]) -> bool;

    /// `audio_creation.creation_tool(value) is not None`.
    fn is_audio_creation(&self, value: &[u8]) -> bool;

    /// `metadata.payload_carries_ai_metadata(raw)`.
    fn carries_ai_metadata(&self, raw: &[u8]) -> bool;

    /// The value bytes of `metadata._ai_exif_targets(piexif.load(exif))` (`{}` when it fails to load).
    fn ai_exif_values(&self, exif: &[u8]) -> Vec<Vec<u8>>;
}

impl<T: MetadataHooks + ?Sized> MetadataHooks for &T {
    fn is_tc260_label(&self, value: &[u8]) -> bool {
        (**self).is_tc260_label(value)
    }

    fn names_ai_generator(&self, payload: &[u8]) -> bool {
        (**self).names_ai_generator(payload)
    }

    fn is_app_export(&self, value: &[u8]) -> bool {
        (**self).is_app_export(value)
    }

    fn is_audio_creation(&self, value: &[u8]) -> bool {
        (**self).is_audio_creation(value)
    }

    fn carries_ai_metadata(&self, raw: &[u8]) -> bool {
        (**self).carries_ai_metadata(raw)
    }

    fn ai_exif_values(&self, exif: &[u8]) -> Vec<Vec<u8>> {
        (**self).ai_exif_values(exif)
    }
}
