//! The official C2PA soft-binding algorithm registry and token-boundary matching.

#[rustfmt::skip]
mod generated;
pub mod sync;

use std::collections::BTreeSet;

use serde::Serialize;

pub use generated::{REGISTRY, SOURCE_LICENSE, SOURCE_REVISION, SOURCE_URL};

/// Whether a soft binding is an embedded watermark or a content fingerprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SoftBindingKind {
    Watermark,
    Fingerprint,
}

impl SoftBindingKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Watermark => "watermark",
            Self::Fingerprint => "fingerprint",
        }
    }
}

/// One normalized entry from the official C2PA soft-binding registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SoftBindingAlgorithm {
    pub identifier: u16,
    pub algorithm: &'static str,
    pub kind: SoftBindingKind,
    pub decoded_media_types: &'static [&'static str],
    pub encoded_media_types: &'static [&'static str],
    pub display_label: &'static str,
    pub date_entered: &'static str,
    pub resolution_apis: &'static [&'static str],
    pub deprecated: bool,
}

/// Compact historical labels, the public display contract, matched by algorithm prefix.
pub const LABEL_OVERRIDES: &[(&str, &str)] = &[
    ("com.adobe.trustmark", "Adobe TrustMark"),
    ("com.adobe.icn", "Adobe Image Comparator Network"),
    ("com.digimarc", "Digimarc Validate"),
    ("com.imatag.lamark", "Imatag (Lamark)"),
    ("ai.steg", "Steg.AI"),
    ("com.microsoft.invismark", "Microsoft InvisMark"),
    ("com.microsoft.wavmark", "Microsoft WavMark"),
    ("com.verimatrix", "Verimatrix"),
    ("com.nagra.nexguard", "NAGRA NexGuard"),
    ("com.aiwatermark.pixelseal", "AIWatermark PixelSeal"),
    ("com.aiwatermark.videoseal", "AIWatermark VideoSeal"),
    ("com.aiwatermark.audioseal", "AIWatermark AudioSeal"),
    ("ai.trufo", "Trufo PawPrint"),
    ("app.overlai", "Overlai"),
    ("com.markany", "MarkAny"),
    ("com.mentaport", "Mentaport"),
    ("es.lumatrace", "LumaTrace"),
    ("ai.verda", "VerdaAI"),
    ("ai.contentlens", "ContentLens"),
    ("io.iscc", "ISCC (content code)"),
];

/// Vendor algorithm families outside the official list, matched with dotted suffixes.
pub const VENDOR_PREFIXES: &[SoftBindingAlgorithm] = &[SoftBindingAlgorithm {
    identifier: 0,
    algorithm: "com.suno.fingerprint",
    kind: SoftBindingKind::Fingerprint,
    decoded_media_types: &["audio/*"],
    encoded_media_types: &["audio/*"],
    display_label: "Suno Audio Fingerprint",
    date_entered: "",
    resolution_apis: &[],
    deprecated: false,
}];

/// `C2PA_SOFT_BINDINGS[algorithm]`: the display label of a registered algorithm.
pub fn registered_label(algorithm: &str) -> Option<&'static str> {
    let entry = REGISTRY.iter().find(|entry| entry.algorithm == algorithm)?;
    Some(
        LABEL_OVERRIDES
            .iter()
            .find(|(prefix, _)| entry.algorithm.starts_with(prefix))
            .map_or(entry.display_label, |(_, label)| label),
    )
}

/// The label shown for `entry`: its registered label, else its own description.
pub fn display_label(entry: &SoftBindingAlgorithm) -> &'static str {
    registered_label(entry.algorithm).unwrap_or(entry.display_label)
}

const fn is_token_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'.' || byte == b'-'
}

/// Every start index of `needle` in `haystack`, overlapping occurrences included.
fn occurrences<'a>(haystack: &'a [u8], needle: &'a [u8]) -> impl Iterator<Item = usize> + 'a {
    let finder = memchr::memmem::Finder::new(needle);
    let mut from = 0;
    std::iter::from_fn(move || {
        let found = from + finder.find(haystack.get(from..)?)?;
        from = found + 1;
        Some(found)
    })
}

fn starts_token(buffer: &[u8], start: usize) -> bool {
    start == 0 || !is_token_byte(buffer[start - 1])
}

/// Whether `algorithm` appears as a whole token: no `[A-Za-z0-9.-]` on either side.
fn names_exact(buffer: &[u8], algorithm: &[u8]) -> bool {
    occurrences(buffer, algorithm).any(|start| {
        let end = start + algorithm.len();
        starts_token(buffer, start) && buffer.get(end).is_none_or(|&b| !is_token_byte(b))
    })
}

/// Whether `prefix` appears followed only by `.segment` groups up to a token boundary.
fn names_prefix_family(buffer: &[u8], prefix: &[u8]) -> bool {
    occurrences(buffer, prefix).any(|start| {
        if !starts_token(buffer, start) {
            return false;
        }
        let tail_start = start + prefix.len();
        let tail_len = buffer[tail_start..]
            .iter()
            .take_while(|&&b| is_token_byte(b))
            .count();
        let tail = &buffer[tail_start..tail_start + tail_len];
        tail.is_empty()
            || (tail[0] == b'.' && tail[1..].split(|&b| b == b'.').all(|segment| !segment.is_empty()))
    })
}

/// Registered and vendor-prefix entries named as whole tokens in manifest bytes.
pub fn registry_entries_in(buffer: &[u8]) -> Vec<&'static SoftBindingAlgorithm> {
    let official = REGISTRY
        .iter()
        .filter(|entry| names_exact(buffer, entry.algorithm.as_bytes()));
    let vendor = VENDOR_PREFIXES
        .iter()
        .filter(|entry| names_prefix_family(buffer, entry.algorithm.as_bytes()));
    official.chain(vendor).collect()
}

/// Sorted, de-duplicated display labels of `entries`, optionally of one kind.
pub fn labels<'a>(
    entries: impl IntoIterator<Item = &'a SoftBindingAlgorithm>,
    kind: Option<SoftBindingKind>,
) -> Vec<String> {
    entries
        .into_iter()
        .filter(|entry| kind.is_none_or(|kind| entry.kind == kind))
        .map(|entry| display_label(entry).to_owned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// `soft_binding_vendors_in`: labels of every soft-binding algorithm named in manifest bytes.
pub fn vendors_in(buffer: &[u8]) -> Vec<String> {
    labels(registry_entries_in(buffer), None)
}
