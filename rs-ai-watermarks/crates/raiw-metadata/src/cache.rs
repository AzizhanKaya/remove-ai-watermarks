//! Per-file memoization keyed on content identity, like Python's `lru_cache` shims.

use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

/// `(path, mtime_ns, size)`: an in-place rewrite changes it even within one mtime tick.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ContentKey {
    path: PathBuf,
    mtime_ns: i128,
    size: u64,
}

impl ContentKey {
    /// `None` when the file cannot be stat'ed (read uncached).
    pub fn of(path: &Path) -> Option<Self> {
        let meta = std::fs::metadata(path).ok()?;
        let modified = meta.modified().ok()?;
        let mtime_ns = match modified.duration_since(UNIX_EPOCH) {
            Ok(after) => after.as_nanos() as i128,
            Err(before) => -(before.duration().as_nanos() as i128),
        };
        Some(Self {
            path: path.to_path_buf(),
            mtime_ns,
            size: meta.len(),
        })
    }

    pub const fn size(&self) -> u64 {
        self.size
    }
}

/// A small most-recently-used list.
#[derive(Debug)]
pub struct LruCache<V> {
    entries: Vec<(ContentKey, V)>,
    capacity: usize,
}

impl<V: Clone> LruCache<V> {
    pub const fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::new(),
            capacity,
        }
    }

    pub fn get(&self, key: &ContentKey) -> Option<V> {
        self.entries
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.clone())
    }

    pub fn put(&mut self, key: ContentKey, value: V) {
        self.entries.retain(|(k, _)| *k != key);
        if self.entries.len() >= self.capacity {
            self.entries.remove(0);
        }
        self.entries.push((key, value));
    }
}

/// Memoize `compute` for `path` in `cache`, computing uncached when the file has no key.
pub fn memoize<V: Clone>(
    cache: &std::sync::Mutex<LruCache<V>>,
    path: &Path,
    compute: impl FnOnce() -> V,
) -> V {
    let Some(key) = ContentKey::of(path) else {
        return compute();
    };
    if let Ok(guard) = cache.lock()
        && let Some(hit) = guard.get(&key)
    {
        return hit;
    }
    let value = compute();
    if let Ok(mut guard) = cache.lock() {
        guard.put(key, value.clone());
    }
    value
}
