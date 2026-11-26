//! The text and JSON each `video` command prints, with its exit code (from `cli.py`).
//!
//! The CLI layer prints the banner, renders [`CommandOutput::lines`] verbatim (two-space
//! indents included) and exits with [`CommandOutput::exit_code`].

use std::path::Path;

use raiw_core::{JsonStyle, to_py_json};
use raiw_metadata::AiMetadata;
use serde::Serialize;

use crate::audio::{VideoAudioMultiStreamResult, VideoAudioRemovalResult};
use crate::messages::{VideoMessageRemovalResult, VideoMessageReport, VideoWatermarkPresenceReport};
use crate::report::{
    StreamAction, VideoAllResult, VideoAudioStatus, VideoBatchResult, VideoEraseResult, VideoInvisibleResult,
    VideoMetadataReport, VideoMetadataResult, VideoProvenanceReport, VideoVisibleResult,
};

/// `EXIT_NO_VISIBLE_MARK`, `EXIT_NO_INVISIBLE_SIGNAL` and `EXIT_NOTHING_PROCESSED` (all 2).
pub const EXIT_NO_SIGNAL: i32 = 2;

/// Lines to print and the process exit code.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CommandOutput {
    pub lines: Vec<String>,
    pub exit_code: i32,
}

impl CommandOutput {
    fn push(&mut self, line: impl Into<String>) {
        self.lines.push(line.into());
    }

    fn exit(mut self, code: i32) -> Self {
        self.exit_code = code;
        self
    }
}

/// `json.dumps(asdict(value), indent=2, default=str)`.
pub fn to_json<T: Serialize + ?Sized>(value: &T) -> String {
    to_py_json(value, JsonStyle::PRETTY)
}

fn json_output<T: Serialize + ?Sized>(value: &T, exit_code: i32) -> CommandOutput {
    CommandOutput {
        lines: vec![to_json(value)],
        exit_code,
    }
}

fn name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn sorted_keys(markers: &AiMetadata) -> String {
    let mut keys: Vec<&str> = markers.keys().map(String::as_str).collect();
    keys.sort_unstable();
    keys.join(", ")
}

/// `_print_metadata_not_a_clean_verdict`.
pub fn not_a_clean_verdict() -> String {
    "  This is not the same as 'clean': a pixel watermark such as SynthID cannot be\n  detected here once its metadata proxy is absent.".to_owned()
}

/// `_print_metadata_report`: the warning lines and the `(key, value[:80])` table rows.
pub fn metadata_report(
    source: &Path,
    has_ai: bool,
    markers: &AiMetadata,
) -> (Vec<String>, Vec<(String, String)>) {
    if !has_ai {
        return (
            vec![
                format!("  No AI metadata found in {}", name(source)),
                not_a_clean_verdict(),
            ],
            Vec::new(),
        );
    }
    let mut lines = vec![format!("  Warning: AI metadata detected in {}:", name(source))];
    if let Some(synthid) = markers.get("synthid_watermark").filter(|s| !s.is_empty()) {
        lines.push(format!("  Warning: SynthID watermark {synthid}"));
    }
    let rows = markers
        .iter()
        .map(|(k, v)| (k.clone(), v.chars().take(80).collect()))
        .collect();
    (lines, rows)
}

/// `video identify --json`.
pub fn identify_json(report: &VideoProvenanceReport) -> CommandOutput {
    json_output(report, 0)
}

/// `video identify` text.
pub fn identify_text(report: &VideoProvenanceReport, no_visible: bool, audio: bool) -> CommandOutput {
    let mut out = CommandOutput::default();
    let verdict = if report.is_ai_generated == Some(true) {
        "AI-generated"
    } else {
        "unknown"
    };
    out.push(format!(
        "  Verdict: {verdict}  (confidence: {})",
        report.confidence
    ));
    out.push(format!(
        "  Platform: {}",
        report.platform.as_deref().unwrap_or("undetermined")
    ));
    match &report.visible_mark {
        Some(mark) => {
            let label = if report.visible_marks.len() > 1 {
                report.visible_marks.join(", ")
            } else {
                mark.clone()
            };
            let total = report
                .total_frames
                .map_or_else(|| "None".to_owned(), |t| t.to_string());
            out.push(format!(
                "  Visible mark: {label} ({}/{total} stable frames)",
                report.visible_detected_frames
            ));
        }
        None => out.push(if no_visible {
            "  Visible mark: not checked"
        } else {
            "  Visible mark: none found"
        }),
    }
    if report.metadata_markers.is_empty() {
        out.push("  AI metadata markers: none found");
    } else {
        out.push(format!(
            "  AI metadata markers: {}",
            sorted_keys(&report.metadata_markers)
        ));
    }
    if audio {
        if report.audio_streams.is_empty() {
            out.push("  Audio streams: none found");
        } else {
            let statuses: Vec<String> = report
                .audio_streams
                .iter()
                .enumerate()
                .map(|(i, s)| format!("stream {i}={}", s.watermark_status.as_str()))
                .collect();
            out.push(format!("  Audio streams: {}", statuses.join(", ")));
        }
    }
    if let Some(recipe) = &report.recommended_recipe {
        out.push(format!("  Recommended recipe: {recipe}"));
    }
    if !report.invisible_watermark_hints.is_empty() {
        out.push(format!(
            "  Invisible watermark hints: {}",
            report.invisible_watermark_hints.join(", ")
        ));
    }
    if !report.caveats.is_empty() {
        out.push("  Caveats:");
        for caveat in &report.caveats {
            out.push(format!("  - {caveat}"));
        }
    }
    out
}

/// `video metadata --check --json` (and the default check).
pub fn metadata_check_json(report: &VideoMetadataReport) -> CommandOutput {
    json_output(report, 0)
}

/// `video metadata --remove` (JSON or text).
pub fn metadata_remove_output(result: &VideoMetadataResult, as_json: bool) -> CommandOutput {
    let failed = !result.remaining.is_empty();
    if as_json {
        return json_output(result, i32::from(failed));
    }
    let mut out = CommandOutput::default();
    if failed {
        out.push(format!(
            "  FAILED: {} AI metadata marker(s) survived in {}",
            result.remaining.len(),
            result.output.display()
        ));
        out.push(format!("    still present: {}", sorted_keys(&result.remaining)));
        return out.exit(1);
    }
    out.push(format!("  AI metadata stripped -> {}", result.output.display()));
    out.push(not_a_clean_verdict());
    out
}

/// `_emit_json_result`: JSON with the text path's exit code.
pub fn emit_json_result<T: Serialize>(
    result: &T,
    wrote_output: bool,
    remaining: &AiMetadata,
    no_output_exit: Option<i32>,
) -> CommandOutput {
    let code = match no_output_exit {
        Some(code) if !wrote_output => code,
        _ => i32::from(!remaining.is_empty()),
    };
    json_output(result, code)
}

fn survived(remaining: &AiMetadata, output: Option<&Path>) -> Option<CommandOutput> {
    (!remaining.is_empty()).then(|| CommandOutput {
        lines: vec![format!(
            "  FAILED: {} AI metadata marker(s) survived in {}",
            remaining.len(),
            output.map_or_else(|| "None".to_owned(), |p| p.display().to_string())
        )],
        exit_code: 1,
    })
}

/// `_print_video_audio_unverified`.
pub fn audio_unverified_line() -> String {
    "  Audio watermark: UNVERIFIED; source audio copied unchanged if present".to_owned()
}

/// `_print_video_audio_status`.
pub fn audio_status_line(
    status: &VideoAudioStatus,
    processed: &[usize],
    scope: &[String],
    quality_passed: Option<bool>,
) -> String {
    let scope_suffix = if scope.is_empty() {
        " (proprietary audio schemes such as SynthID are not checked locally)".to_owned()
    } else {
        format!(
            " (public decoders: {}; proprietary audio schemes such as SynthID are not checked locally)",
            scope.join(", ")
        )
    };
    let quality_suffix = if quality_passed == Some(false) {
        "; WARNING: delivered audio failed minimum quality gate"
    } else {
        ""
    };
    match status.stream_action {
        StreamAction::Absent => "  Audio watermark: ABSENT; source has no audio streams".to_owned(),
        StreamAction::Processed => {
            let tracks: Vec<String> = processed.iter().map(ToString::to_string).collect();
            format!(
                "  Audio watermark: {}; processed audio stream(s) {}{scope_suffix}{quality_suffix}",
                status.watermark_status.as_str().to_uppercase(),
                tracks.join(", ")
            )
        }
        _ if status.watermark_status == crate::report::AudioWatermarkState::NotDetected => {
            format!(
                "  Audio watermark: NOT DETECTED under compatible checks{scope_suffix}; source audio copied unchanged if present"
            )
        }
        _ => audio_unverified_line(),
    }
}

/// The scanning banner line of `video visible`.
pub fn visible_start_line(source: &Path, mark: &str) -> String {
    format!(
        "  Scanning {} for a temporally stable {mark} mark...",
        name(source)
    )
}

/// `video visible` (JSON or text).
pub fn visible_output(result: &VideoVisibleResult, requested_mark: &str, as_json: bool) -> CommandOutput {
    if as_json {
        return emit_json_result(
            result,
            result.output.is_some(),
            &result.remaining_metadata,
            Some(EXIT_NO_SIGNAL),
        );
    }
    let Some(output) = &result.output else {
        return CommandOutput {
            lines: vec![format!(
                "  No stable {requested_mark} watermark detected; no output written"
            )],
            exit_code: EXIT_NO_SIGNAL,
        };
    };
    if let Some(failure) = survived(&result.remaining_metadata, Some(output)) {
        return failure;
    }
    let mut out = CommandOutput::default();
    let label = if result.marks.len() > 1 {
        result.marks.join(", ")
    } else {
        result.mark.clone()
    };
    out.push(format!(
        "  Removed {label} watermark from {}/{} frames -> {}",
        result.removed_frames,
        result.total_frames,
        output.display()
    ));
    if result.trimmed_bumper_frames > 0 {
        out.push(format!(
            "  Trimmed {} trailing AI bumper frame(s)",
            result.trimmed_bumper_frames
        ));
    }
    out.push(audio_unverified_line());
    out
}

/// The opening line of `video erase`.
pub fn erase_start_line(source: &Path, boxes: usize) -> String {
    format!("  Erasing {boxes} region(s) from {}...", name(source))
}

/// `video erase` (JSON or text).
pub fn erase_output(result: &VideoEraseResult, as_json: bool) -> CommandOutput {
    if as_json {
        return emit_json_result(result, true, &result.remaining_metadata, None);
    }
    if let Some(failure) = survived(&result.remaining_metadata, Some(&result.output)) {
        return failure;
    }
    CommandOutput {
        lines: vec![
            format!(
                "  Erased {} region(s) across {}/{} frames -> {}",
                result.regions.len(),
                result.erased_frames,
                result.total_frames,
                result.output.display()
            ),
            audio_unverified_line(),
        ],
        exit_code: 0,
    }
}

/// `_video_profile_label`.
pub fn video_profile_label(profile: &str) -> &'static str {
    match profile {
        "seedance" | "seedance-experimental" | "seedance-texture-v1" => "the Seedance profile",
        "wan21-synthid" | "wan21-synthid-experimental" => "the CUDA 3D Wan 2.1 SynthID profile",
        "wan21-seedance" | "wan21-seedance-experimental" => "the CUDA 3D Wan 2.1 Seedance profile",
        "wan22-synthid" | "wan22-synthid-experimental" => "the 3D Wan 2.2 SynthID profile (Rust-only)",
        _ => "the calibrated SynthID-removal profile",
    }
}

/// `video invisible` (JSON or text).
pub fn invisible_output(result: &VideoInvisibleResult, as_json: bool) -> CommandOutput {
    if as_json {
        return emit_json_result(result, true, &result.remaining_metadata, None);
    }
    if let Some(failure) = survived(&result.remaining_metadata, Some(&result.output)) {
        return failure;
    }
    CommandOutput {
        lines: vec![
            format!(
                "  Video pixels: regenerated with {} (not a per-file verification): {}x{}, {} frames at {} fps -> {}",
                video_profile_label(&result.profile),
                result.metrics.width,
                result.metrics.height,
                result.metrics.frames,
                crate::py::format_g(result.metrics.fps, 4),
                result.output.display()
            ),
            audio_unverified_line(),
        ],
        exit_code: 0,
    }
}

/// The opening line of `video all`.
pub fn all_start_line(source: &Path, invisible_profile: Option<&str>, audio_scheme: Option<&str>) -> String {
    let mut stages = vec!["visible marks".to_owned()];
    if let Some(profile) = invisible_profile {
        stages.push(profile.to_owned());
    }
    if let Some(scheme) = audio_scheme {
        stages.push(format!("audio ({scheme})"));
    }
    stages.push("verified AI metadata".to_owned());
    format!("  Cleaning {}: {}...", name(source), stages.join(" + "))
}

/// `video all` (JSON or text).
pub fn all_output(result: &VideoAllResult, requested_profile: &str, as_json: bool) -> CommandOutput {
    if as_json {
        return emit_json_result(result, true, &result.remaining_metadata, None);
    }
    if let Some(failure) = survived(&result.remaining_metadata, Some(&result.output)) {
        return failure;
    }
    let mut out = CommandOutput::default();
    match &result.visible_mark {
        None => out.push(format!(
            "  Visible mark: none found{}",
            if result.invisible_removed {
                ""
            } else {
                "; pixels preserved"
            }
        )),
        Some(mark) => {
            let label = if result.visible_marks.len() > 1 {
                result.visible_marks.join(", ")
            } else {
                mark.clone()
            };
            out.push(format!(
                "  Visible mark: removed {label} from {}/{} frames",
                result.visible_removed_frames, result.total_frames
            ));
        }
    }
    if result.trimmed_bumper_frames > 0 {
        out.push(format!(
            "  Trimmed {} trailing AI bumper frame(s)",
            result.trimmed_bumper_frames
        ));
    }
    out.push(format!("  AI metadata: stripped -> {}", result.output.display()));
    if result.invisible_removed {
        let profile = result.invisible_profile.as_deref().unwrap_or(requested_profile);
        out.push(format!(
            "  Video pixels: regenerated with {} (not a per-file verification)",
            video_profile_label(profile)
        ));
    }
    out.push(audio_status_line(
        &result.audio(),
        &result.audio_processed_streams,
        &result.audio_verification_scope,
        result.audio_quality_passed,
    ));
    out
}

/// The opening line of `video batch`.
pub fn batch_start_line(directory: &Path, mode: &str) -> String {
    format!(
        "  Processing video directory {} in {mode} mode...",
        directory.display()
    )
}

/// `video batch` (JSON or text).
pub fn batch_output(
    result: &VideoBatchResult,
    directory: &Path,
    audio_invisible: bool,
    as_json: bool,
) -> CommandOutput {
    let code = if result.items.is_empty() {
        EXIT_NO_SIGNAL
    } else {
        i32::from(result.failed() > 0)
    };
    if as_json {
        return json_output(result, code);
    }
    let mut out = CommandOutput::default();
    if result.items.is_empty() {
        out.push(format!(
            "  No supported videos found in {}; nothing was processed",
            directory.display()
        ));
        return out.exit(EXIT_NO_SIGNAL);
    }
    for item in &result.items {
        let output = item
            .output
            .as_ref()
            .map_or_else(|| "None".to_owned(), |p| p.display().to_string());
        if let Some(error) = &item.error {
            out.push(format!("  FAILED {}: {error}", name(&item.source)));
        } else if item.changed {
            let detail = item
                .visible_mark
                .as_ref()
                .map_or_else(String::new, |m| format!(" ({m})"));
            out.push(format!("  Processed {}{detail} -> {output}", name(&item.source)));
        } else if item.mode == crate::report::BatchMode::Visible {
            out.push(format!(
                "  Copied {} byte-for-byte -> {output}",
                name(&item.source)
            ));
        } else {
            out.push(format!(
                "  Completed {}; no supported signal found -> {output}",
                name(&item.source)
            ));
        }
    }
    out.push(format!(
        "  Batch complete: {} processed, {} failed -> {}",
        result.processed(),
        result.failed(),
        result.output_directory.display()
    ));
    if result.invisible_removed() > 0 {
        out.push(format!(
            "  Video pixel regeneration: applied to {} file(s)",
            result.invisible_removed()
        ));
    }
    if result.processed() > 0 && !audio_invisible {
        out.push(audio_unverified_line());
    }
    out.exit(code)
}

/// `_audio_check_label`.
pub fn audio_check_label(scheme: &str) -> &'static str {
    match scheme {
        "audioseal" => "open AudioSeal",
        "wavmark" => "public WavMark",
        "perth" => "public Perth v1",
        "silentcipher" => "public SilentCipher-44k",
        "all" => "AudioSeal, WavMark, Perth v1, and SilentCipher-44k",
        _ => "installed compatible public audio watermark",
    }
}

fn verification_status(verified: Option<bool>) -> &'static str {
    match verified {
        Some(true) => "NOT DETECTED in saved output",
        Some(false) => "STILL DETECTED",
        None => "UNVERIFIED",
    }
}

fn audio_exit(skipped: bool, verified: Option<bool>) -> i32 {
    if skipped {
        EXIT_NO_SIGNAL
    } else {
        i32::from(verified == Some(false))
    }
}

/// `video audio` for one stream (JSON or text).
pub fn audio_single_output(
    result: &VideoAudioRemovalResult,
    stream: usize,
    scheme: &str,
    as_json: bool,
) -> CommandOutput {
    let code = audio_exit(result.skipped, result.verified_removed);
    if as_json {
        return json_output(result, code);
    }
    let mut out = CommandOutput::default();
    match (&result.output, result.skipped) {
        (Some(output), false) => out.push(format!("Processed audio track {stream}: {}", output.display())),
        _ => out.push("No compatible source-positive audio signal; no output written."),
    }
    out.push(format!(
        "Compatible {} checks: {}",
        audio_check_label(scheme),
        verification_status(result.verified_removed)
    ));
    out.push(format!("Processing method: {}", result.method));
    let mut caveats: Vec<&String> = Vec::new();
    let extra = result
        .before
        .caveats
        .iter()
        .chain(result.after.iter().flat_map(|a| a.caveats.iter()));
    for caveat in result.caveats.iter().chain(extra) {
        if !caveats.contains(&caveat) {
            caveats.push(caveat);
        }
    }
    for caveat in caveats {
        out.push(format!("  - {caveat}"));
    }
    out.exit(code)
}

/// `video audio --all-streams` (JSON or text).
pub fn audio_multi_output(
    result: &VideoAudioMultiStreamResult,
    scheme: &str,
    as_json: bool,
) -> CommandOutput {
    let code = audio_exit(result.skipped, result.verified_removed);
    if as_json {
        return json_output(result, code);
    }
    let mut out = CommandOutput::default();
    if result.skipped {
        out.push("No compatible source-positive audio signal across streams; no output written.");
    } else {
        let tracks: Vec<String> = result
            .processed_ordinals
            .iter()
            .map(ToString::to_string)
            .collect();
        let output = result
            .output
            .as_ref()
            .map_or_else(|| "None".to_owned(), |p| p.display().to_string());
        out.push(format!(
            "Processed audio track(s) {}: {output}",
            tracks.join(", ")
        ));
        out.push(format!(
            "Compatible {} checks: {}",
            audio_check_label(scheme),
            verification_status(result.verified_removed)
        ));
    }
    for caveat in &result.caveats {
        out.push(format!("  - {caveat}"));
    }
    out.exit(code)
}

/// `_verification_error` for `--json`: the error object printed before exit 1.
pub fn verification_error_json(error: &crate::error::VideoError) -> CommandOutput {
    #[derive(Serialize)]
    #[serde(untagged)]
    enum Evidence<'a> {
        Audio(&'a raiw_audio::AudioProvenanceReport),
        Message(&'a VideoMessageReport),
    }
    #[derive(Serialize)]
    struct ErrorReport<'a> {
        error: String,
        output: Option<&'a str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        before: Option<Evidence<'a>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        after: Option<Evidence<'a>>,
    }
    let (before, after) = match error {
        crate::error::VideoError::Audio(raiw_audio::AudioError::Verification(v)) => {
            (Some(Evidence::Audio(&v.before)), Some(Evidence::Audio(&v.after)))
        }
        crate::error::VideoError::MessageVerification(v) => (
            Some(Evidence::Message(&v.before)),
            Some(Evidence::Message(&v.after)),
        ),
        _ => (None, None),
    };
    json_output(
        &ErrorReport {
            error: error.to_string(),
            output: None,
            before,
            after,
        },
        1,
    )
}

/// Python's `str()` of an optional bool.
fn py_bool(value: Option<bool>) -> &'static str {
    match value {
        Some(true) => "True",
        Some(false) => "False",
        None => "None",
    }
}

/// The default `video message remove` output: `<stem>_message_processed.mp4` beside the source.
pub fn message_default_output(source: &Path) -> std::path::PathBuf {
    let stem = source
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    source.with_file_name(format!("{stem}_message_processed.mp4"))
}

/// `video message identify` (JSON or text).
pub fn message_identify_output(report: &VideoMessageReport, as_json: bool) -> CommandOutput {
    if as_json {
        return json_output(report, 0);
    }
    let mut out = CommandOutput::default();
    out.push(format!(
        "  Public {} known message: {}",
        report.scheme, report.status
    ));
    for caveat in &report.caveats {
        out.push(format!("  - {caveat}"));
    }
    out
}

/// `video message detect` (JSON or text).
pub fn message_detect_output(report: &VideoWatermarkPresenceReport, as_json: bool) -> CommandOutput {
    if as_json {
        return json_output(report, 0);
    }
    let mut out = CommandOutput::default();
    out.push(format!(
        "  Public {} blind presence: {}",
        report.scheme, report.status
    ));
    if let Some(hex) = &report.decoded_hex {
        out.push(format!("  Recovered payload hex: {hex}"));
    }
    for caveat in &report.caveats {
        out.push(format!("  - {caveat}"));
    }
    out
}

/// `video message remove` (JSON or text); exit 2 when skipped, 1 when the target still matches.
pub fn message_remove_output(result: &VideoMessageRemovalResult, as_json: bool) -> CommandOutput {
    let code = if result.skipped {
        EXIT_NO_SIGNAL
    } else {
        i32::from(result.after.as_ref().is_some_and(|a| a.status == "matched"))
    };
    if as_json {
        return json_output(result, code);
    }
    let mut out = CommandOutput::default();
    let processed = if result.skipped {
        "skipped".to_owned()
    } else {
        result
            .output
            .as_ref()
            .map_or_else(|| "None".to_owned(), |p| p.display().to_string())
    };
    out.push(format!("  Known-message processing: {processed}"));
    out.push(format!(
        "  Verified known-target disruption: {}",
        py_bool(result.verified_removed)
    ));
    if let Some(quality) = &result.quality {
        let ssim = quality
            .get("source_ssim")
            .map_or_else(|| "None".to_owned(), raiw_core::pyjson::py_repr);
        out.push(format!(
            "  Delivered SSIM: {ssim}; method: {}",
            result.method.as_deref().unwrap_or("None")
        ));
    }
    out.push("  Scope covers the known public-model message. Other watermarks remain unverified.");
    out.exit(code)
}
