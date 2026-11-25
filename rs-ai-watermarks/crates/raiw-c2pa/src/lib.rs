//! C2PA Content Credentials: validated reading, raw fallback scanning and vendor attribution.
//!
//! [`extract_c2pa_info`] prefers the validating reader (c2pa-rs) and falls back to the
//! bounded PNG `caBX`/app11 scan, exactly as Python's `_internal/c2pa.py` does.

pub mod attribution;
pub mod extractor;
pub mod fallback;
pub mod info;
pub mod markers;
pub mod png;
#[cfg(feature = "reader")]
pub mod reader;
pub mod registry;
pub mod soft_binding;
pub mod store;
pub mod text;

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::UNIX_EPOCH;

pub use attribution::{
    c2pa_vendor_platform, claim_generator_platform, registry_word_matches, synthid_evidence_vendors_in,
    synthid_verdict,
};
pub use fallback::{cbor_text_after, info_from_raw_app11_profile, parse_c2pa_chunk, raw_app11_jumbf_bytes};
pub use info::{
    AiSourceKind, C2paInfo, Check, CredentialLevel, SignerTrust, SignerValidity, Validation, ValidationSource,
};
pub use markers::{C2PA_STRUCTURAL_MARKERS, markers_claim_ai};
pub use png::{extract_c2pa_chunk, extract_c2pa_info_png, has_c2pa_metadata, inject_c2pa_chunk};
pub use store::{info_from_manifest_json, info_from_manifest_value};

/// Whether the validating reader is compiled in.
pub const fn reader_available() -> bool {
    cfg!(feature = "reader")
}

/// Uncached normalized evidence for `path`.
pub fn extract_c2pa_info_uncached(path: &Path) -> Option<C2paInfo> {
    #[cfg(feature = "reader")]
    if let Ok(Some(store)) = reader::read_manifest_store_json(path) {
        return info_from_manifest_json(&store);
    }
    extract_c2pa_info_png(path)
}

type CacheKey = (PathBuf, i128, u64);

/// The last few results, keyed on content identity (path, mtime, size).
static CACHE: Mutex<Vec<(CacheKey, Option<C2paInfo>)>> = Mutex::new(Vec::new());
const CACHE_ENTRIES: usize = 4;

fn cache_key(path: &Path) -> Option<CacheKey> {
    let metadata = std::fs::metadata(path).ok()?;
    let modified = metadata.modified().ok()?;
    let nanos = match modified.duration_since(UNIX_EPOCH) {
        Ok(after) => after.as_nanos() as i128,
        Err(before) => -(before.duration().as_nanos() as i128),
    };
    Some((path.to_path_buf(), nanos, metadata.len()))
}

/// Normalized C2PA evidence from the validating reader or the PNG fallback; `None` without C2PA.
///
/// Memoized on `(path, mtime, size)`, so one report reaching it twice runs the reader once.
pub fn extract_c2pa_info(path: &Path) -> Option<C2paInfo> {
    let Some(key) = cache_key(path) else {
        return extract_c2pa_info_uncached(path);
    };
    if let Ok(cache) = CACHE.lock()
        && let Some((_, hit)) = cache.iter().find(|(cached, _)| *cached == key)
    {
        return hit.clone();
    }
    let info = extract_c2pa_info_uncached(path);
    if let Ok(mut cache) = CACHE.lock() {
        if cache.len() >= CACHE_ENTRIES {
            cache.remove(0);
        }
        cache.push((key, info.clone()));
    }
    info
}
