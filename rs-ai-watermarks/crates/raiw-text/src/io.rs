//! UTF-8 text input/output without newline translation, published through staged writes.

use std::path::{Component, Path, PathBuf};

use raiw_core::StagedSibling;

use crate::error::{TextError, decode_utf8};

/// A text document given in memory or as a file.
#[derive(Debug, Clone, Copy)]
pub enum TextInput<'a> {
    Text(&'a str),
    File(&'a Path),
}

impl<'a> From<&'a str> for TextInput<'a> {
    fn from(text: &'a str) -> Self {
        Self::Text(text)
    }
}

impl<'a> From<&'a String> for TextInput<'a> {
    fn from(text: &'a String) -> Self {
        Self::Text(text)
    }
}

impl<'a> From<&'a Path> for TextInput<'a> {
    fn from(path: &'a Path) -> Self {
        Self::File(path)
    }
}

impl<'a> From<&'a PathBuf> for TextInput<'a> {
    fn from(path: &'a PathBuf) -> Self {
        Self::File(path)
    }
}

impl TextInput<'_> {
    pub fn source(&self) -> Option<&Path> {
        match self {
            Self::Text(_) => None,
            Self::File(path) => Some(path),
        }
    }

    /// Read the content: in-memory text as is, files as strict UTF-8 bytes.
    pub fn read(&self) -> Result<String, TextError> {
        match self {
            Self::Text(text) => Ok((*text).to_owned()),
            Self::File(path) => decode_utf8(std::fs::read(path).map_err(|err| TextError::io(*path, err))?),
        }
    }
}

/// `Path.resolve()`: canonicalize the longest existing prefix and append the rest.
fn resolve(path: &Path) -> PathBuf {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().map_or_else(|_| path.to_path_buf(), |cwd| cwd.join(path))
    };
    let components: Vec<Component<'_>> = absolute.components().collect();
    for split in (1..=components.len()).rev() {
        let existing: PathBuf = components[..split].iter().collect();
        let Ok(mut resolved) = std::fs::canonicalize(&existing) else {
            continue;
        };
        for component in &components[split..] {
            match component {
                Component::ParentDir => {
                    resolved.pop();
                }
                Component::CurDir => {}
                other => resolved.push(other.as_os_str()),
            }
        }
        return resolved;
    }
    absolute
}

/// `<stem>_clean<suffix>` next to the source.
pub fn default_output_path(source: &Path) -> PathBuf {
    let stem = source
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_default();
    let name = match source.extension() {
        Some(ext) => format!("{stem}_clean.{}", ext.to_string_lossy()),
        None => format!("{stem}_clean"),
    };
    source.with_file_name(name)
}

/// Resolve and validate the destination before any transformation runs.
pub fn resolve_output_path(
    source: Option<&Path>,
    output: Option<&Path>,
) -> Result<Option<PathBuf>, TextError> {
    let resolved = match (output, source) {
        (Some(output), _) => Some(output.to_path_buf()),
        (None, Some(source)) => Some(default_output_path(source)),
        (None, None) => None,
    };
    if let (Some(resolved), Some(source)) = (&resolved, source)
        && resolve(resolved) == resolve(source)
    {
        return Err(TextError::invalid(
            "Output path must differ from source text file",
        ));
    }
    Ok(resolved)
}

/// Validate the destination, then read the text.
pub(crate) fn prepare_text_io(
    input: TextInput<'_>,
    output: Option<&Path>,
) -> Result<(Option<PathBuf>, String), TextError> {
    let resolved = resolve_output_path(input.source(), output)?;
    Ok((resolved, input.read()?))
}

/// Atomically publish `text` as UTF-8 when a destination is active.
pub(crate) fn write_output_text(output: Option<&Path>, text: &str) -> Result<(), TextError> {
    let Some(output) = output else { return Ok(()) };
    if let Some(parent) = output.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|err| TextError::io(parent, err))?;
    }
    let staged = StagedSibling::new(output).map_err(|err| TextError::io(output, err))?;
    std::fs::write(staged.path(), text.as_bytes()).map_err(|err| TextError::io(staged.path(), err))?;
    staged.commit().map_err(|err| TextError::io(output, err))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_output_keeps_the_suffix() {
        assert_eq!(
            default_output_path(Path::new("a/doc.txt")),
            Path::new("a/doc_clean.txt")
        );
        assert_eq!(
            default_output_path(Path::new("a/archive.tar.gz")),
            Path::new("a/archive.tar_clean.gz")
        );
        assert_eq!(
            default_output_path(Path::new(".bashrc")),
            Path::new(".bashrc_clean")
        );
    }

    #[test]
    fn same_output_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("doc.txt");
        std::fs::write(&source, "x").unwrap();
        assert!(resolve_output_path(Some(&source), Some(&dir.path().join("./doc.txt"))).is_err());
        assert!(resolve_output_path(Some(&source), Some(&dir.path().join("new/out.txt"))).is_ok());
    }
}
