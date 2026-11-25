//! Filesystem helpers shared by every writer.

use std::ffi::OsString;
use std::io;
use std::path::{Component, Path, PathBuf};

/// An unused sibling path that is published over its target only on [`StagedSibling::commit`].
///
/// Mirrors Python's `staged_sibling`: the name is never pre-created, so the writer gives
/// the published file default permissions, and a dropped stage removes its leftovers.
#[derive(Debug)]
pub struct StagedSibling {
    staged: PathBuf,
    target: PathBuf,
    committed: bool,
}

impl StagedSibling {
    pub fn new(target: impl Into<PathBuf>) -> io::Result<Self> {
        let target = target.into();
        let mut token = [0u8; 8];
        getrandom::fill(&mut token).map_err(|err| io::Error::other(err.to_string()))?;
        let stem = target.file_stem().map(OsString::from).unwrap_or_default();
        let mut name = OsString::from(".");
        name.push(stem);
        name.push(format!("-{}", hex::encode(token)));
        if let Some(ext) = target.extension() {
            name.push(".");
            name.push(ext);
        }
        Ok(Self {
            staged: target.with_file_name(name),
            target,
            committed: false,
        })
    }

    pub fn path(&self) -> &Path {
        &self.staged
    }

    pub fn target(&self) -> &Path {
        &self.target
    }

    /// Atomically replace the target with the staged file.
    pub fn commit(mut self) -> io::Result<PathBuf> {
        std::fs::rename(&self.staged, &self.target)?;
        self.committed = true;
        Ok(self.target.clone())
    }
}

impl Drop for StagedSibling {
    fn drop(&mut self) {
        if !self.committed {
            let _ = std::fs::remove_file(&self.staged);
        }
    }
}

/// Where downloaded and derived model assets persist (`$HF_HOME` first, then `$XDG_CACHE_HOME`).
pub fn model_cache_dir() -> PathBuf {
    let root = std::env::var_os("HF_HOME")
        .filter(|v| !v.is_empty())
        .or_else(|| std::env::var_os("XDG_CACHE_HOME").filter(|v| !v.is_empty()));
    let base = match root {
        Some(root) => expand_home(PathBuf::from(root)),
        None => home_dir().join(".cache"),
    };
    base.join("remove-ai-watermarks")
}

pub fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map_or_else(|| PathBuf::from("."), PathBuf::from)
}

fn expand_home(path: PathBuf) -> PathBuf {
    match path.strip_prefix("~") {
        Ok(rest) => home_dir().join(rest),
        Err(_) => path,
    }
}

/// Absolute, symlink-resolved form of `path` that tolerates missing tails (Python's `Path.resolve()`).
pub fn resolve_lenient(path: &Path) -> PathBuf {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(path)
    };
    let mut resolved = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                resolved.pop();
            }
            other => {
                resolved.push(other.as_os_str());
                if let Ok(real) = resolved.canonicalize() {
                    resolved = real;
                }
            }
        }
    }
    resolved
}

/// `Path.with_stem(stem + suffix)`: keep the final extension, extend the stem.
pub fn with_stem_suffix(path: &Path, addition: &str) -> PathBuf {
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut name = format!("{stem}{addition}");
    if let Some(ext) = path.extension() {
        name.push('.');
        name.push_str(&ext.to_string_lossy());
    }
    path.with_file_name(name)
}

/// `Path.with_name(name + suffix)`: extend the whole file name.
pub fn with_name_suffix(path: &Path, addition: &str) -> PathBuf {
    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    path.with_file_name(format!("{name}{addition}"))
}

/// `Path.with_name(stem + replacement)`: drop the extension and append `replacement`.
pub fn with_stem_replaced(path: &Path, replacement: &str) -> PathBuf {
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    path.with_file_name(format!("{stem}{replacement}"))
}

/// Lower-cased extension with its leading dot, as Python's `Path.suffix.casefold()`.
pub fn suffix_lower(path: &Path) -> String {
    path.extension()
        .map(|ext| format!(".{}", ext.to_string_lossy().to_lowercase()))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staged_sibling_publishes_only_on_commit() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("out.png");
        {
            let stage = StagedSibling::new(&target).unwrap();
            std::fs::write(stage.path(), b"draft").unwrap();
            assert!(
                stage
                    .path()
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with(".out-")
            );
        }
        assert!(!target.exists());
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
        let stage = StagedSibling::new(&target).unwrap();
        std::fs::write(stage.path(), b"final").unwrap();
        stage.commit().unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"final");
    }

    #[test]
    fn python_path_helpers() {
        assert_eq!(
            with_stem_suffix(Path::new("d/a.tar.gz"), "_clean"),
            Path::new("d/a.tar_clean.gz")
        );
        assert_eq!(
            with_stem_suffix(Path::new("d/.bashrc"), "_clean"),
            Path::new("d/.bashrc_clean")
        );
        assert_eq!(
            with_name_suffix(Path::new("d/dir"), "_clean"),
            Path::new("d/dir_clean")
        );
        assert_eq!(
            with_stem_replaced(Path::new("d/a.mp4"), "_processed.wav"),
            Path::new("d/a_processed.wav")
        );
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().canonicalize().unwrap();
        assert_eq!(
            resolve_lenient(&dir.path().join("x/../y.png")),
            real.join("y.png")
        );
    }

    #[test]
    fn suffix_is_lowercase_with_dot() {
        assert_eq!(suffix_lower(Path::new("a/B.JPEG")), ".jpeg");
        assert_eq!(suffix_lower(Path::new("noext")), "");
    }
}
