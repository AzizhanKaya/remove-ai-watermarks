//! The C2PA manifest store JSON (`read_manifest_store_json`), memoized per file content.

use std::path::Path;
use std::sync::Mutex;

use crate::cache::{LruCache, memoize};

type Read = Result<Option<String>, String>;

static CACHE: Mutex<LruCache<Read>> = Mutex::new(LruCache::new(8));

fn read(path: &Path) -> Read {
    raiw_c2pa::reader::read_manifest_store_json(path).map_err(|err| err.to_string())
}

/// The store, `None` without one or when the reader fails (the tolerant read).
pub fn manifest_store_json(path: &Path) -> Option<String> {
    memoize(&CACHE, path, || read(path)).ok().flatten()
}

/// `read_manifest_store_json(path, strict=True)`: reader failures are errors.
pub fn manifest_store_json_strict(path: &Path) -> Result<Option<String>, String> {
    memoize(&CACHE, path, || read(path))
}
