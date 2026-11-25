//! Structural readability checks behind `metadata.unreadable_reason`.
//!
//! The raster-decode check belongs to the metadata layer; these cover the containers whose
//! top-level elements must tile the file.

use std::io;

use crate::source::ReadAt;
use crate::{ebml, isobmff};

/// The JPEG XL ISOBMFF container signature.
pub const JXL_CONTAINER_MAGIC: &[u8] = b"\x00\x00\x00\x0cJXL \r\n\x87\n";

/// Accepted on their magic alone: JPEG XL codestream, FLV, ID3, FLAC and Ogg.
pub const MAGIC_ONLY: [&[u8]; 5] = [b"\xff\x0a", b"FLV", b"ID3", b"fLaC", b"OggS"];

/// `None` when the last top-level element ends at EOF; a truncation leaves a gap.
pub fn tiling_reason(kind: &str, ends: impl IntoIterator<Item = u64>, size: u64) -> Option<String> {
    let end = ends.into_iter().max().unwrap_or(0);
    (end != size)
        .then(|| format!("the {kind} structure stops at byte {end} of {size} (truncated or corrupt)"))
}

/// Whether ISOBMFF top-level boxes tile the source.
pub fn isobmff_tiling_reason<R: ReadAt + ?Sized>(source: &R) -> io::Result<Option<String>> {
    let ends = isobmff::boxes(source, 0, source.size())?
        .into_iter()
        .map(|entry| entry.end);
    Ok(tiling_reason("ISOBMFF box", ends, source.size()))
}

/// Whether EBML top-level elements tile the source.
pub fn ebml_tiling_reason<R: ReadAt + ?Sized>(source: &R) -> io::Result<Option<String>> {
    Ok(tiling_reason(
        "EBML element",
        ebml::top_level_ends(source)?,
        source.size(),
    ))
}

/// Formats with no local walker that pass on their magic: the list above or MPEG audio sync.
pub fn passes_on_magic(head: &[u8]) -> bool {
    let mpeg_audio_sync = head.len() > 1 && head[0] == 0xFF && head[1] & 0xE0 == 0xE0;
    MAGIC_ONLY.iter().any(|magic| head.starts_with(magic)) || mpeg_audio_sync
}
