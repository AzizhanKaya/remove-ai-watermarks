//! ISO Base Media File Format (AVIF, HEIF, MP4, MOV, M4A, JPEG XL container) walker.
//!
//! Detection seeks over `mdat`; removal never moves a byte: provenance boxes become
//! `free` boxes, and labels, SEI tags and AV1 metadata are blanked in place.

mod boxes;
mod heif;
mod items;
mod labels;
mod region;
mod sei;
mod strip;

pub use boxes::{
    BoxEntry, Boxes, FourCc, Span, boxes, is_isobmff, iter_boxes, read_box_header, top_level_boxes,
};
pub use heif::{heif_transform_applied, source_transform_applied};
pub use items::{MetadataItem, metadata_items, moov_items};
pub use labels::{
    LabelRegion, ai_metadata_regions, ai_metadata_tags, app_regions, audio_creation_regions,
    has_audio_creation_comment, has_xai_signature, is_xai_signature, tc260_aigc_payloads, tc260_regions,
    xai_regions, xai_signature_in_region, xai_signature_payloads,
};
pub use region::{C2PA_BOX_TYPES, DEFAULT_REGION_BYTES, has_c2pa_emsg_marker, scan_c2pa_region, scan_region};
pub use sei::{VideoSeiScan, scan_source_video_sei, scan_video_sei};
pub use strip::{
    ImageStripCounts, MediaStripCounts, MediaStripOptions, ProvenanceBox, STREAM_SCAN_BYTES,
    blank_ai_exif_tokens, blank_ai_xmp_packets, blank_tc260_aigc_tags, has_ai_label_or_generator,
    streaming_provenance_boxes, strip_c2pa_boxes, strip_image_file, strip_media_file,
};
