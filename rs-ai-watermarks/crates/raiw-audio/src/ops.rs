//! Metadata, `all` and batch entry points of `audio.py`, over the metadata seam.

use std::path::{Path, PathBuf};

use raiw_core::StagedSibling;
use raiw_core::constants::{AUDIO_EXTENSIONS, AUDIO_OUTPUT_EXTENSIONS};
use raiw_core::fs::suffix_lower;
use serde::Serialize;

use crate::error::{AudioError, Result};
use crate::io::{AudioReader, DEFAULT_CHUNK_FRAMES, probe_audio, write_audio};
use crate::policy::resolve_audio_profiles;
use crate::remove::{AudioRemovalResult, RemovalRequest, ensure_distinct, remove_audio_invisible};
use crate::report::Markers;
use crate::runtime::AudioRuntime;

/// AI metadata found in one supported audio container.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AudioMetadataReport {
    pub source: PathBuf,
    pub has_ai_metadata: bool,
    pub markers: Markers,
}

/// Result of a verified metadata strip that preserves encoded audio.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AudioMetadataResult {
    pub source: PathBuf,
    pub output: PathBuf,
    pub detected: Markers,
    pub remaining: Markers,
}

/// Verified AI metadata removal and optional audio-watermark processing.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AudioAllResult {
    pub source: PathBuf,
    pub output: PathBuf,
    pub detected_metadata: Markers,
    pub remaining_metadata: Markers,
    pub invisible_applied: bool,
    pub invisible_result: Option<AudioRemovalResult>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum BatchMode {
    All,
    Metadata,
}

/// Outcome for one source in an audio batch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AudioBatchItem {
    pub source: PathBuf,
    pub output: Option<PathBuf>,
    pub mode: BatchMode,
    pub changed: bool,
    pub invisible_applied: bool,
    pub error: Option<String>,
}

/// Aggregate outcome for a sequential audio batch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AudioBatchSummary {
    pub directory: PathBuf,
    pub output_directory: PathBuf,
    pub items: Vec<AudioBatchItem>,
}

impl AudioBatchSummary {
    pub fn processed(&self) -> usize {
        self.items.iter().filter(|i| i.error.is_none()).count()
    }

    pub fn failed(&self) -> usize {
        self.items.iter().filter(|i| i.error.is_some()).count()
    }

    pub fn invisible_applied(&self) -> usize {
        self.items.iter().filter(|i| i.invisible_applied).count()
    }
}

/// `_audio_source`: an existing file with a supported audio suffix and a readable container.
pub fn audio_source(runtime: &AudioRuntime, source: &Path) -> Result<PathBuf> {
    if !source.exists() {
        return Err(AudioError::NotFound(format!(
            "Audio file does not exist: {}",
            source.display()
        )));
    }
    if !source.is_file() {
        return Err(AudioError::invalid(format!(
            "Audio source must be a file: {}",
            source.display()
        )));
    }
    let suffix = suffix_lower(source);
    if !AUDIO_EXTENSIONS.contains(&suffix.as_str()) {
        let mut supported: Vec<&str> = AUDIO_EXTENSIONS.to_vec();
        supported.sort_unstable();
        let shown = source
            .extension()
            .map_or_else(|| "<none>".to_owned(), |e| format!(".{}", e.to_string_lossy()));
        return Err(AudioError::invalid(format!(
            "Unsupported audio format {shown}; expected one of: {}",
            supported.join(", ")
        )));
    }
    if let Some(reason) = runtime.metadata.unreadable_reason(source) {
        return Err(AudioError::invalid(format!(
            "Cannot read audio {}: {reason}",
            source.display()
        )));
    }
    Ok(source.to_path_buf())
}

/// `<stem>_clean<suffix>` next to `source`.
pub fn clean_sibling(source: &Path, suffix: Option<&str>) -> PathBuf {
    let stem = source
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let suffix = suffix.map_or_else(
        || {
            source
                .extension()
                .map(|e| format!(".{}", e.to_string_lossy()))
                .unwrap_or_default()
        },
        str::to_owned,
    );
    source.with_file_name(format!("{stem}_clean{suffix}"))
}

fn metadata_output(source: &Path, output: Option<&Path>) -> Result<PathBuf> {
    let path = output.map_or_else(|| clean_sibling(source, None), Path::to_path_buf);
    if suffix_lower(&path) != suffix_lower(source) {
        let dotted = |p: &Path| {
            p.extension()
                .map_or_else(|| "<none>".to_owned(), |e| format!(".{}", e.to_string_lossy()))
        };
        return Err(AudioError::invalid(format!(
            "Audio metadata output container must match the source ({}); metadata removal does not change containers to {}",
            source
                .extension()
                .map(|e| format!(".{}", e.to_string_lossy()))
                .unwrap_or_default(),
            dotted(&path)
        )));
    }
    if ensure_distinct(source, &path).is_err() {
        return Err(AudioError::invalid(
            "Audio metadata removal requires a distinct output path",
        ));
    }
    Ok(path)
}

/// `inspect_audio_metadata`.
pub fn inspect_audio_metadata(runtime: &AudioRuntime, source: &Path) -> Result<AudioMetadataReport> {
    let path = audio_source(runtime, source)?;
    let markers = runtime.metadata.ai_metadata(&path)?;
    Ok(AudioMetadataReport {
        source: path,
        has_ai_metadata: !markers.is_empty(),
        markers,
    })
}

/// `remove_audio_metadata`: strip AI metadata without re-encoding audio samples.
pub fn remove_audio_metadata(
    runtime: &AudioRuntime,
    source: &Path,
    output: Option<&Path>,
    keep_standard: bool,
    detected: Option<Markers>,
) -> Result<AudioMetadataResult> {
    let path = audio_source(runtime, source)?;
    let output = metadata_output(&path, output)?;
    let detected = match detected {
        Some(markers) => markers,
        None => runtime.metadata.ai_metadata(&path)?,
    };
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let (written, remaining) = runtime.metadata.strip_and_verify(&path, &output, keep_standard)?;
    Ok(AudioMetadataResult {
        source: path,
        output: written,
        detected,
        remaining,
    })
}

/// `_transcode_clean_audio`: a metadata-free copy when the output container differs.
pub fn transcode_clean_audio(
    runtime: &AudioRuntime,
    source: &Path,
    target: &Path,
    stream: usize,
) -> Result<()> {
    let tools = runtime
        .tools
        .as_ref()
        .ok_or_else(|| AudioError::unavailable("Audio processing requires ffmpeg"))?;
    let info = probe_audio(tools, source, stream)?;
    if let Some(parent) = target.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let staged = StagedSibling::new(target)?;
    let reader = AudioReader::open(tools, source, &info, DEFAULT_CHUNK_FRAMES)?;
    write_audio(tools, &info, staged.path(), reader, None, false)?;
    staged.commit()?;
    Ok(())
}

/// Options of `remove_audio_all` beyond the removal request.
#[derive(Debug, Clone, Default)]
pub struct AllOptions {
    pub include_invisible: bool,
    pub keep_standard: bool,
    pub removal: RemovalRequest,
}

/// `remove_audio_all`: AI metadata removal plus optional compatible audio-watermark processing.
pub fn remove_audio_all(
    runtime: &AudioRuntime,
    source: &Path,
    output: Option<&Path>,
    options: &AllOptions,
) -> Result<AudioAllResult> {
    let path = audio_source(runtime, source)?;
    let vendor = options.removal.args.vendor;
    if !(options.include_invisible || vendor.is_some()) {
        let meta = remove_audio_metadata(runtime, &path, output, options.keep_standard, None)?;
        return Ok(AudioAllResult {
            source: path,
            output: meta.output,
            detected_metadata: meta.detected,
            remaining_metadata: meta.remaining,
            invisible_applied: false,
            invisible_result: None,
        });
    }
    let mut permissions = options.removal.permissions;
    permissions.allow_unverified |= vendor.is_some();
    resolve_audio_profiles(runtime, &options.removal.args, permissions)?;
    let suffix = suffix_lower(&path);
    let target = match output {
        Some(out) => out.to_path_buf(),
        None if AUDIO_OUTPUT_EXTENSIONS.contains(&suffix.as_str()) => clean_sibling(&path, None),
        None => clean_sibling(&path, Some(".wav")),
    };
    if !AUDIO_OUTPUT_EXTENSIONS.contains(&suffix_lower(&target).as_str()) {
        return Err(AudioError::invalid(format!(
            "Audio output format must be one of: {}",
            AUDIO_OUTPUT_EXTENSIONS.join(", ")
        )));
    }
    ensure_distinct(&path, &target)?;
    let result = remove_audio_invisible(runtime, &path, &target, &options.removal)?;
    let detected = result.before.metadata_markers.clone();
    if !result.skipped
        && let Some(written) = result.output.clone()
    {
        let remaining = match &result.after {
            Some(after) => after.metadata_markers.clone(),
            None => runtime.metadata.ai_metadata(&written)?,
        };
        return Ok(AudioAllResult {
            source: path,
            output: written,
            detected_metadata: detected,
            remaining_metadata: remaining,
            invisible_applied: true,
            invisible_result: Some(result),
        });
    }
    let (delivered, remaining) = if suffix_lower(&target) == suffix {
        let meta = remove_audio_metadata(
            runtime,
            &path,
            Some(&target),
            options.keep_standard,
            Some(detected.clone()),
        )?;
        (meta.output, meta.remaining)
    } else {
        transcode_clean_audio(runtime, &path, &target, options.removal.stream)?;
        (target.clone(), runtime.metadata.ai_metadata(&target)?)
    };
    Ok(AudioAllResult {
        source: path,
        output: delivered,
        detected_metadata: detected,
        remaining_metadata: remaining,
        invisible_applied: false,
        invisible_result: Some(result),
    })
}

/// `remove_audio_batch`: every supported audio file in one directory, sequentially.
pub fn remove_audio_batch(
    runtime: &AudioRuntime,
    directory: &Path,
    output_directory: Option<&Path>,
    mode: BatchMode,
    options: &AllOptions,
) -> Result<AudioBatchSummary> {
    if !directory.exists() {
        return Err(AudioError::NotFound(format!(
            "Audio directory does not exist: {}",
            directory.display()
        )));
    }
    if !directory.is_dir() {
        return Err(AudioError::invalid(format!(
            "Audio batch source must be a directory: {}",
            directory.display()
        )));
    }
    let vendor = options.removal.args.vendor;
    let invisible = options.include_invisible || vendor.is_some();
    if invisible && mode != BatchMode::All {
        return Err(AudioError::invalid(
            "The invisible audio stage is available only in all mode",
        ));
    }
    if invisible {
        let mut permissions = options.removal.permissions;
        permissions.allow_unverified |= vendor.is_some();
        resolve_audio_profiles(runtime, &options.removal.args, permissions)?;
    }
    let out_dir = output_directory.map_or_else(
        || {
            let name = directory
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            directory
                .parent()
                .unwrap_or_else(|| Path::new(""))
                .join(format!("{name}_clean"))
        },
        Path::to_path_buf,
    );
    let canonical = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    if canonical(&out_dir) == canonical(directory) {
        return Err(AudioError::invalid(
            "Audio batch output directory must differ from the source directory",
        ));
    }
    let mut sources: Vec<PathBuf> = std::fs::read_dir(directory)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && AUDIO_EXTENSIONS.contains(&suffix_lower(p).as_str()))
        .collect();
    sources.sort_by_key(|p| {
        p.file_name()
            .map(|n| n.to_string_lossy().to_lowercase())
            .unwrap_or_default()
    });
    if !sources.is_empty() {
        std::fs::create_dir_all(&out_dir)?;
    }
    let mut items = Vec::with_capacity(sources.len());
    for source in sources {
        let original_suffix = source
            .extension()
            .map(|e| format!(".{}", e.to_string_lossy()))
            .unwrap_or_default();
        let suffix = if !invisible || AUDIO_OUTPUT_EXTENSIONS.contains(&suffix_lower(&source).as_str()) {
            original_suffix.clone()
        } else {
            ".wav".to_owned()
        };
        let stem = source
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let item_output = out_dir.join(format!("{stem}{suffix}"));
        let outcome = match mode {
            BatchMode::All => {
                let item_options = AllOptions {
                    include_invisible: invisible,
                    ..options.clone()
                };
                remove_audio_all(runtime, &source, Some(&item_output), &item_options).and_then(|res| {
                    if !res.remaining_metadata.is_empty() {
                        return Err(AudioError::failed(format!(
                            "{} AI metadata marker(s) survived audio cleaning",
                            res.remaining_metadata.len()
                        )));
                    }
                    let changed = !res.detected_metadata.is_empty()
                        || res.invisible_applied
                        || suffix != original_suffix;
                    Ok((res.output, changed, res.invisible_applied))
                })
            }
            BatchMode::Metadata => {
                remove_audio_metadata(runtime, &source, Some(&item_output), options.keep_standard, None)
                    .and_then(|res| {
                        if !res.remaining.is_empty() {
                            return Err(AudioError::failed(format!(
                                "{} AI metadata marker(s) survived audio metadata removal",
                                res.remaining.len()
                            )));
                        }
                        Ok((res.output, !res.detected.is_empty(), false))
                    })
            }
        };
        items.push(match outcome {
            Ok((output, changed, invisible_applied)) => AudioBatchItem {
                source,
                output: Some(output),
                mode,
                changed,
                invisible_applied,
                error: None,
            },
            Err(err) => AudioBatchItem {
                source,
                output: None,
                mode,
                changed: false,
                invisible_applied: false,
                error: Some(err.to_string()),
            },
        });
    }
    Ok(AudioBatchSummary {
        directory: directory.to_path_buf(),
        output_directory: out_dir,
        items,
    })
}
