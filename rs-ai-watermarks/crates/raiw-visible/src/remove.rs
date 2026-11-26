//! The library-level visible removal (`api.remove_visible_detailed`): detect -> mask -> fill -> write.

use std::path::Path;

use raiw_core::StagedSibling;

use crate::eraser::{Backend, FillBackends};
use crate::error::{Result, VisibleError};
use crate::image_io::{self, LoadedImage, WriteOptions};
use crate::raster::Pixels;
use crate::registry::{Context, Registry, Sensitivity, VisibleRemovalResult};

/// Strips AI metadata from a written output and returns whatever AI metadata remains.
///
/// Owned by raiw-metadata; the CLI supplies the implementation.
pub trait MetadataStripper {
    fn strip(&self, staged: &Path) -> Result<Vec<String>>;
}

impl<F: Fn(&Path) -> Result<Vec<String>>> MetadataStripper for F {
    fn strip(&self, staged: &Path) -> Result<Vec<String>> {
        self(staged)
    }
}

/// The knobs of one visible pass.
#[derive(Clone, Copy, Debug)]
pub struct VisibleOptions<'a> {
    pub sensitivity: Sensitivity,
    pub backend: Backend,
    /// Write a passthrough copy when nothing was removed.
    pub write_noop: bool,
    /// Strip AI metadata from the written output.
    pub strip_metadata: Option<&'a dyn MetadataStripper>,
}

impl Default for VisibleOptions<'_> {
    fn default() -> Self {
        Self { sensitivity: Sensitivity::Auto, backend: Backend::Auto, write_noop: true, strip_metadata: None }
    }
}

impl std::fmt::Debug for dyn MetadataStripper + '_ {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MetadataStripper")
    }
}

/// Remove every detected known mark from `source` and optionally write `output`.
///
/// `provenance` and `vetoed_marks` are the metadata-derived context (TC260 producer marks,
/// non-Google vendor vetoes) the caller collected; the alpha plane and the source's display
/// tags are carried onto the output, and a true no-op copies the original bytes.
pub fn remove_visible(
    source: &Path,
    output: Option<&Path>,
    context: &Context,
    options: &VisibleOptions<'_>,
    backends: &FillBackends,
) -> Result<VisibleRemovalResult> {
    let registry = Registry::standard()?;
    let loaded = image_io::load_image(source)?;
    let bgr = loaded.bgr.as_u8("remove_visible")?;
    let context = Context { sensitivity: options.sensitivity, ..context.clone() };
    let report = registry.remove_auto_marks_detailed(bgr, &context, options.backend, backends)?;
    if let Some(output) = output {
        write_result(source, &loaded, &report, output, options)?;
    }
    Ok(report)
}

fn write_result(
    source: &Path,
    loaded: &LoadedImage,
    report: &VisibleRemovalResult,
    output: &Path,
    options: &VisibleOptions<'_>,
) -> Result<()> {
    let removed = !report.marks.is_empty();
    if !removed && !options.write_noop {
        return Ok(());
    }
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let stage = StagedSibling::new(output)?;
    let same_suffix = raiw_core::fs::suffix_lower(source) == raiw_core::fs::suffix_lower(output);
    if !removed && same_suffix {
        std::fs::copy(source, stage.path())?;
    } else {
        let options = WriteOptions { display_tags_from: Some(source), orientation_applied: Some(false) };
        image_io::save_image(stage.path(), &Pixels::U8(report.image.clone()), loaded.alpha.as_ref(), &options)
            .map_err(|_| VisibleError::WriteFailed(output.to_path_buf()))?;
    }
    if let Some(stripper) = options.strip_metadata {
        let leftover = stripper.strip(stage.path())?;
        if !leftover.is_empty() {
            return Err(VisibleError::MetadataStripIncomplete(leftover));
        }
    }
    stage.commit()?;
    Ok(())
}
