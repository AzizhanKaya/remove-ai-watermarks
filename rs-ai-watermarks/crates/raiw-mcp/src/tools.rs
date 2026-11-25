//! Convenience tools (`identify`, `remove`, `batch`, audio and video-message helpers).

use std::path::Path;

use raiw_core::constants::{
    ALL_PRESETS, AUDIO_BANDWIDTHS_KBPS, AUDIO_EXTENSIONS, AUDIO_METHODS, AUDIO_SCHEMES, AUDIO_VENDORS,
    AUDIO_VOCODER_DEVICES, DEFAULT_VIDEO_MESSAGE_LONG_SIDE, TEXT_EXTENSIONS, VIDEO_EXTENSIONS,
    VIDEO_MESSAGE_DEVICES, VIDEO_MESSAGE_METHODS, VIDEO_MESSAGE_SCHEMES,
};
use raiw_core::fs::{resolve_lenient, suffix_lower, with_name_suffix, with_stem_replaced, with_stem_suffix};
use raiw_core::pyjson::{py_float_repr, py_str};
use serde_json::{Map, Value, json};

use crate::{Arguments, InvalidArguments};

const MODES: [&str; 3] = ["metadata", "visible", "all"];

/// A planned CLI invocation and the output path it targets.
pub type Plan = (Vec<String>, Option<String>);

/// The convenience tool definitions, in protocol order.
pub fn convenience_tools() -> Vec<Value> {
    let mut tools = vec![identify_tool(), remove_tool(), batch_tool()];
    for (name, description) in [
        (
            "audio_identify",
            "Inspect audio metadata and named compatible public-model evidence. Read-only.",
        ),
        (
            "audio_invisible",
            "Process one audio track to WAV, FLAC, MP3, M4A, OGG, OGA, Opus, AAC, or MKA. Default publication requires \
             a compatible source positive under the requested models and a fully negative saved output. \
             Force permits explicitly unverified results.",
        ),
        (
            "video_audio",
            "Process one selected audio track (or all source-positive tracks with all_streams=true) inside a video, \
             copying other encoded streams. Outputs MKV/MOV with float PCM or MP4 with selected-track AAC \
             (target 192 kbps). Visual marks and untouched audio tracks remain unchanged. \
             Default publication requires compatible source-positive and final-mux-negative checks \
             for the requested models.",
        ),
    ] {
        tools.push(audio_tool(name, description));
    }
    for action in ["identify", "remove"] {
        tools.push(video_message_tool(action));
    }
    tools
}

fn identify_tool() -> Value {
    json!({
        "name": "identify",
        "title": "Identify AI provenance",
        "description": "Report which AI provenance signals an image, video, audio, or text file carries: metadata \
            (C2PA, IPTC, EXIF, TC260), visible AI-generator labels, deterministic text marks, and supported \
            local invisible-watermark decoders. Read-only. Media type is detected by file suffix.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Image, video, audio, or text file to inspect."},
                "check_visible": {
                    "type": "boolean",
                    "default": true,
                    "description": "Run pixel, visible-mark, or public audio-model detectors. False checks metadata only.",
                },
                "check_audio": {
                    "type": "boolean",
                    "description": "Inspect audio streams with compatible public decoders (overrides check_visible for audio \
                        files; adds --audio on video identify).",
                },
            },
            "required": ["path"],
            "additionalProperties": false,
        },
        "annotations": {"readOnlyHint": true, "openWorldHint": false},
    })
}

fn remove_tool() -> Value {
    json!({
        "name": "remove",
        "title": "Remove AI watermarks from one file",
        "description": "Write a cleaned copy of an image, video, audio, or text file the user generated or edited. \
            mode=metadata strips AI metadata without touching pixels or re-encoding audio/video; \
            mode=visible also removes a known visible AI label on images or videos; \
            mode=all adds the invisible-watermark scrub for images (when extras and a GPU allow), \
            compatible public-decoder audio (audio all --invisible), and text (text all), while for video \
            it runs visible+metadata cleaning only (use cli_video_all for opt-in video/audio invisible regeneration). \
            The source is never overwritten; the default output is <name>_clean beside it.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Image, video, audio, or text file to clean."},
                "mode": {"type": "string", "enum": MODES, "default": "metadata"},
                "preset": {
                    "type": "string",
                    "enum": ALL_PRESETS,
                    "description": "Optional high-level preset (lossless-fast, balanced, deep-clean) routed through top-level all.",
                },
                "output": {"type": "string", "description": "Output file. Must differ from the source."},
            },
            "required": ["path"],
            "additionalProperties": false,
        },
        "annotations": {"readOnlyHint": false, "destructiveHint": false, "openWorldHint": false},
    })
}

fn batch_tool() -> Value {
    json!({
        "name": "batch",
        "title": "Remove AI watermarks from a directory",
        "description": "Clean every supported image (media=image), video (media=video), or audio file (media=audio) in a \
            directory the user owns, writing results to a separate directory (default <dir>_clean). \
            Modes match the remove tool (for video, use cli_video_batch when opt-in invisible regeneration is needed).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Directory of input files."},
                "mode": {"type": "string", "enum": MODES, "default": "metadata"},
                "media": {"type": "string", "enum": ["image", "video", "audio", "all"], "default": "image"},
                "preset": {
                    "type": "string",
                    "enum": ALL_PRESETS,
                    "description": "Optional high-level preset (lossless-fast, balanced, deep-clean) for multimodal batch.",
                },
                "output_dir": {"type": "string", "description": "Output directory. Must differ from the input."},
            },
            "required": ["directory"],
            "additionalProperties": false,
        },
        "annotations": {"readOnlyHint": false, "destructiveHint": false, "openWorldHint": false},
    })
}

fn audio_tool(name: &str, description: &str) -> Value {
    let mut properties = Map::new();
    properties.insert("path".into(), json!({"type": "string"}));
    properties.insert(
        "stream".into(),
        json!({"type": "integer", "minimum": 0, "default": 0}),
    );
    match name {
        "audio_identify" => {
            properties.insert(
                "check_invisible".into(),
                json!({"type": "boolean", "default": true}),
            );
        }
        "video_audio" => {
            properties.insert("all_streams".into(), json!({"type": "boolean", "default": false}));
        }
        _ => {}
    }
    properties.insert(
        "scheme".into(),
        json!({"type": "string", "enum": AUDIO_SCHEMES, "default": AUDIO_SCHEMES[0]}),
    );
    if name != "audio_identify" {
        let extra = json!({
            "output": {"type": "string"},
            "force": {"type": "boolean", "default": false},
            "allow_unverified": {"type": "boolean", "default": false},
            "allow_positive": {"type": "boolean", "default": false},
            "allow_quality_override": {"type": "boolean", "default": false},
            "verify": {"type": "boolean", "default": true},
            "method": {
                "type": "string",
                "enum": AUDIO_METHODS,
                "default": AUDIO_METHODS[0],
                "description": "DAC and SNAC are explicit 44.1 kHz neural-codec profiles, not SynthID decoders.",
            },
            "vocoder_device": {"type": "string", "enum": AUDIO_VOCODER_DEVICES, "default": AUDIO_VOCODER_DEVICES[0]},
            "vendor": {"type": "string", "enum": AUDIO_VENDORS, "default": AUDIO_VENDORS[0]},
            "bandwidth": {
                "type": "integer",
                "enum": AUDIO_BANDWIDTHS_KBPS,
                "default": AUDIO_BANDWIDTHS_KBPS[AUDIO_BANDWIDTHS_KBPS.len() - 1],
            },
        });
        properties.extend(extra.as_object().cloned().unwrap_or_default());
    }
    json!({
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "required": ["path"],
            "additionalProperties": false,
            "properties": properties,
        },
        "annotations": {"readOnlyHint": name == "audio_identify", "destructiveHint": false, "openWorldHint": false},
    })
}

fn video_message_tool(action: &str) -> Value {
    let mut properties = json!({
        "path": {"type": "string"},
        "reference": {
            "type": "string",
            "description": "Independent encoder message (64 hex digits for videoseal/pixelseal, \
                256 for chunkyseal). Never derive it from this input.",
        },
        "scheme": {"type": "string", "enum": VIDEO_MESSAGE_SCHEMES, "default": VIDEO_MESSAGE_SCHEMES[0]},
    });
    if action == "remove"
        && let Some(map) = properties.as_object_mut()
    {
        let extra = json!({
            "output": {"type": "string"},
            "method": {"type": "string", "enum": VIDEO_MESSAGE_METHODS, "default": VIDEO_MESSAGE_METHODS[0]},
            "noise_std": {"type": "number", "minimum": 0, "maximum": 1},
            "long_side": {"type": "integer", "minimum": 8, "default": DEFAULT_VIDEO_MESSAGE_LONG_SIDE},
            "device": {"type": "string", "enum": VIDEO_MESSAGE_DEVICES, "default": VIDEO_MESSAGE_DEVICES[0]},
            "force": {"type": "boolean", "default": false},
            "verify": {"type": "boolean", "default": true},
        });
        map.extend(extra.as_object().cloned().unwrap_or_default());
    }
    let tail = if action == "identify" {
        "Read-only, checks every decoded frame."
    } else {
        "Writes a processed video after saved-file checks; copies audio and reports delivered quality."
    };
    json!({
        "name": format!("video_message_{action}"),
        "description": format!(
            "Experimental known-message correlation under pinned public VideoSeal, PixelSeal, or ChunkySeal models. \
             Requires an independently known encoder message. Does not identify AI origin or unknown marks. {tail}"
        ),
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": ["path", "reference"],
            "additionalProperties": false,
        },
        "annotations": {"readOnlyHint": action == "identify", "destructiveHint": false, "openWorldHint": false},
    })
}

// Argument validation

/// Check `arguments` against the tool's own input schema and fill its defaults.
pub fn validated(tool: &Value, arguments: &Arguments) -> Result<Arguments, InvalidArguments> {
    let schema = &tool["inputSchema"];
    let properties = schema["properties"].as_object().cloned().unwrap_or_default();
    let required: Vec<&str> = schema["required"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    let mut unknown: Vec<&str> = arguments
        .keys()
        .filter(|k| !properties.contains_key(*k))
        .map(String::as_str)
        .collect();
    if !unknown.is_empty() {
        unknown.sort_unstable();
        return Err(InvalidArguments(format!(
            "unknown argument(s): {}",
            unknown.join(", ")
        )));
    }
    let mut values = Map::new();
    for (key, spec) in &properties {
        let Some(value) = arguments.get(key) else {
            if required.contains(&key.as_str()) {
                return Err(InvalidArguments(format!("'{key}' is required")));
            }
            if let Some(default) = spec.get("default") {
                values.insert(key.clone(), default.clone());
            }
            continue;
        };
        let kind = spec["type"].as_str().unwrap_or("string");
        let minimum = spec.get("minimum").and_then(Value::as_f64);
        let maximum = spec.get("maximum").and_then(Value::as_f64);
        match kind {
            "boolean" if !value.is_boolean() => {
                return Err(InvalidArguments(format!("'{key}' must be a boolean")));
            }
            "integer"
                if !(value.is_i64() || value.is_u64()) || value.as_f64() < Some(minimum.unwrap_or(0.0)) =>
            {
                return Err(InvalidArguments(format!("'{key}' must be a nonnegative integer")));
            }
            "number"
                if !value.as_f64().is_some_and(|n| {
                    n.is_finite() && minimum.is_none_or(|m| n >= m) && maximum.is_none_or(|m| n <= m)
                }) =>
            {
                return Err(InvalidArguments(format!(
                    "'{key}' must be a finite number within its declared bounds"
                )));
            }
            "string" if value.as_str().is_none_or(|s| s.trim().is_empty()) => {
                return Err(InvalidArguments(format!("'{key}' must be a non-empty string")));
            }
            _ => {}
        }
        if let Some(Value::Array(choices)) = spec.get("enum")
            && !choices.contains(value)
        {
            let listed: Vec<String> = choices.iter().map(py_str).collect();
            return Err(InvalidArguments(format!(
                "'{key}' must be one of {}",
                listed.join(", ")
            )));
        }
        values.insert(key.clone(), value.clone());
    }
    Ok(values)
}

// Command builders

fn text(values: &Arguments, key: &str) -> String {
    values.get(key).map(py_str).unwrap_or_default()
}

/// The caller's `key` value, or `default` rendered as a path string.
fn path_or(values: &Arguments, key: &str, default: impl FnOnce() -> std::path::PathBuf) -> String {
    values
        .get(key)
        .map_or_else(|| default().to_string_lossy().into_owned(), py_str)
}

fn flag(values: &Arguments, key: &str) -> bool {
    values.get(key).and_then(Value::as_bool) == Some(true)
}

fn has_suffix(path: &str, set: &[&str]) -> bool {
    set.contains(&suffix_lower(Path::new(path)).as_str())
}

fn is_video(path: &str) -> bool {
    has_suffix(path, VIDEO_EXTENSIONS)
}

fn is_audio(path: &str) -> bool {
    has_suffix(path, AUDIO_EXTENSIONS)
}

fn is_text(path: &str) -> bool {
    has_suffix(path, TEXT_EXTENSIONS)
}

fn strings<const N: usize>(items: [&str; N]) -> Vec<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

/// Route a validated convenience-tool call to its CLI invocation.
pub fn plan(tool: &str, values: &Arguments) -> Result<Plan, InvalidArguments> {
    match tool {
        "identify" => Ok(identify_command(values)),
        "remove" => remove_command(values),
        "batch" => batch_command(values),
        "audio_identify" | "audio_invisible" => audio_command(values, false),
        "video_audio" => audio_command(values, true),
        "video_message_identify" => video_message_command(values, false),
        "video_message_remove" => video_message_command(values, true),
        other => Err(InvalidArguments(format!("Unknown tool: {other}"))),
    }
}

fn identify_command(values: &Arguments) -> Plan {
    let path = text(values, "path");
    if is_text(&path) {
        return (
            vec![
                "text".into(),
                "identify".into(),
                "--json".into(),
                "--".into(),
                path,
            ],
            None,
        );
    }
    if is_audio(&path) {
        let run_audio = values
            .get("check_audio")
            .unwrap_or(&values["check_visible"])
            .as_bool()
            == Some(true);
        let mut args = strings(["audio", "identify", "--json"]);
        if !run_audio {
            args.push("--no-invisible".into());
        }
        args.extend(["--".into(), path]);
        return (args, None);
    }
    let video = is_video(&path);
    let mut args = if video {
        strings(["video", "identify"])
    } else {
        strings(["identify"])
    };
    args.push("--json".into());
    if !flag(values, "check_visible") {
        args.push("--no-visible".into());
    }
    if video && flag(values, "check_audio") {
        args.push("--audio".into());
    }
    args.extend(["--".into(), path]);
    (args, None)
}

fn require_distinct_output(source: &Path, output: &str, message: &str) -> Result<(), InvalidArguments> {
    let out = Path::new(output);
    let same_file = out.exists() && source.exists() && same_file(out, source);
    if resolve_lenient(out) == resolve_lenient(source) || same_file {
        return Err(InvalidArguments::new(message));
    }
    Ok(())
}

#[cfg(unix)]
fn same_file(a: &Path, b: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    match (std::fs::metadata(a), std::fs::metadata(b)) {
        (Ok(a), Ok(b)) => a.dev() == b.dev() && a.ino() == b.ino(),
        _ => false,
    }
}

#[cfg(not(unix))]
fn same_file(a: &Path, b: &Path) -> bool {
    a.canonicalize().ok() == b.canonicalize().ok()
}

fn remove_command(values: &Arguments) -> Result<Plan, InvalidArguments> {
    let path = text(values, "path");
    let mode = text(values, "mode");
    let source = Path::new(&path);
    let output = path_or(values, "output", || with_stem_suffix(source, "_clean"));
    require_distinct_output(
        source,
        &output,
        "'output' must differ from the source; the server never overwrites an input",
    )?;
    if let Some(preset) = values.get("preset").map(py_str) {
        return Ok((
            strings(["all", "--preset", &preset, "-o", &output, "--", &path]),
            Some(output),
        ));
    }
    if is_text(&path) && mode == "visible" {
        return Err(InvalidArguments::new(
            "text files do not have a visible pixel mode; use mode=metadata or mode=all",
        ));
    }
    if is_audio(&path) {
        if mode == "visible" {
            return Err(InvalidArguments::new(
                "audio files do not have a visible pixel mode; use mode=metadata or mode=all",
            ));
        }
        let mut args = if mode == "metadata" {
            strings(["audio", "metadata", "--remove"])
        } else {
            strings(["audio", "all", "--invisible"])
        };
        args.extend(strings(["-o", &output, "--", &path]));
        return Ok((args, Some(output)));
    }
    let mut args = if is_text(&path) {
        strings(["text"])
    } else if is_video(&path) {
        strings(["video"])
    } else {
        Vec::new()
    };
    if mode == "metadata" {
        args.extend(strings(["metadata", "--remove"]));
    } else {
        args.push(mode);
    }
    args.extend(strings(["-o", &output, "--", &path]));
    Ok((args, Some(output)))
}

fn batch_command(values: &Arguments) -> Result<Plan, InvalidArguments> {
    let directory = text(values, "directory");
    let resolved = resolve_lenient(Path::new(&directory));
    let output_dir = path_or(values, "output_dir", || with_name_suffix(&resolved, "_clean"));
    if resolve_lenient(Path::new(&output_dir)) == resolved {
        return Err(InvalidArguments::new(
            "'output_dir' must differ from the input directory",
        ));
    }
    let (mode, media) = (text(values, "mode"), text(values, "media"));
    if let Some(preset) = values.get("preset").map(py_str) {
        if media == "video" || media == "audio" {
            return Err(InvalidArguments::new(
                "preset is supported for media=image or media=all multimodal batch",
            ));
        }
        let args = strings([
            "batch",
            "--mode",
            &mode,
            "--preset",
            &preset,
            "-o",
            &output_dir,
            "--",
            &directory,
        ]);
        return Ok((args, Some(output_dir)));
    }
    if media == "audio" {
        if mode == "visible" {
            return Err(InvalidArguments::new(
                "audio batch does not have a visible pixel mode; use mode=metadata or mode=all",
            ));
        }
        let mut args = strings(["audio", "batch", "--mode", &mode]);
        if mode == "all" {
            args.push("--invisible".into());
        }
        args.extend(strings(["-o", &output_dir, "--", &directory]));
        return Ok((args, Some(output_dir)));
    }
    let mut args = if media == "video" {
        strings(["video", "batch"])
    } else {
        strings(["batch"])
    };
    args.extend(strings(["--mode", &mode, "-o", &output_dir, "--", &directory]));
    Ok((args, Some(output_dir)))
}

fn output_and_verify_flags(
    source: &Path,
    output: &str,
    values: &Arguments,
    audio_permissions: bool,
) -> Result<Vec<String>, InvalidArguments> {
    require_distinct_output(source, output, "'output' must differ from the source")?;
    let explicit_vendor = audio_permissions && values.get("vendor").is_some_and(|v| v != AUDIO_VENDORS[0]);
    let allow_unverified = audio_permissions && (flag(values, "allow_unverified") || explicit_vendor);
    let (verify, force) = (flag(values, "verify"), flag(values, "force"));
    if !(verify || force || allow_unverified) {
        return Err(InvalidArguments::new(if audio_permissions {
            "Disabling verification requires force=true or allow_unverified=true"
        } else {
            "Disabling verification requires force=true"
        }));
    }
    let mut flags = strings(["-o", output]);
    if force {
        flags.push("--force".into());
    }
    if audio_permissions {
        if allow_unverified {
            flags.push("--allow-unverified".into());
        }
        if flag(values, "allow_positive") {
            flags.push("--allow-positive".into());
        }
        if flag(values, "allow_quality_override") {
            flags.push("--allow-quality-override".into());
        }
    }
    if !verify {
        flags.push("--no-verify".into());
    }
    Ok(flags)
}

fn audio_command(values: &Arguments, video: bool) -> Result<Plan, InvalidArguments> {
    let path = text(values, "path");
    let source = Path::new(&path);
    let identify = values.contains_key("check_invisible");
    let mut command = if video {
        strings(["video", "audio"])
    } else {
        strings(["audio", if identify { "identify" } else { "invisible" }])
    };
    command.extend(["--json".into(), "--stream".into(), text(values, "stream")]);
    let mut output = None;
    if identify {
        if !flag(values, "check_invisible") {
            command.push("--no-invisible".into());
        }
    } else {
        let suffix = if video {
            "_audio_processed.mkv"
        } else {
            "_processed.wav"
        };
        let out = path_or(values, "output", || with_stem_replaced(source, suffix));
        command.extend(output_and_verify_flags(source, &out, values, true)?);
        let method = text(values, "method");
        if method != AUDIO_METHODS[0] {
            command.extend(["--method".into(), method]);
        }
        let default_bandwidth = AUDIO_BANDWIDTHS_KBPS[AUDIO_BANDWIDTHS_KBPS.len() - 1].to_string();
        let bandwidth = text(values, "bandwidth");
        if bandwidth != default_bandwidth {
            command.extend(["--bandwidth".into(), bandwidth]);
        }
        let device = text(values, "vocoder_device");
        if device != AUDIO_VOCODER_DEVICES[0] {
            command.extend(["--vocoder-device".into(), device]);
        }
        let vendor = values
            .get("vendor")
            .map_or_else(|| AUDIO_VENDORS[0].to_owned(), py_str);
        if vendor != AUDIO_VENDORS[0] {
            command.extend(["--vendor".into(), vendor]);
        }
        output = Some(out);
    }
    if flag(values, "all_streams") {
        command.push("--all-streams".into());
    }
    let scheme = text(values, "scheme");
    if scheme != AUDIO_SCHEMES[0] {
        command.extend(["--scheme".into(), scheme]);
    }
    command.extend(["--".into(), path]);
    Ok((command, output))
}

fn video_message_command(values: &Arguments, remove: bool) -> Result<Plan, InvalidArguments> {
    let path = text(values, "path");
    let mut command = strings([
        "video",
        "message",
        if remove { "remove" } else { "identify" },
        "--json",
    ]);
    command.extend([
        "--reference".into(),
        text(values, "reference"),
        "--scheme".into(),
        text(values, "scheme"),
    ]);
    let mut output = None;
    if remove {
        let source = Path::new(&path);
        let out = path_or(values, "output", || {
            with_stem_replaced(source, "_message_processed.mp4")
        });
        if values.contains_key("noise_std") && text(values, "method") != "vae" {
            return Err(InvalidArguments::new("noise_std requires method=vae"));
        }
        command.extend(output_and_verify_flags(source, &out, values, false)?);
        command.extend([
            "--method".into(),
            text(values, "method"),
            "--long-side".into(),
            text(values, "long_side"),
            "--device".into(),
            text(values, "device"),
        ]);
        if let Some(noise) = values.get("noise_std") {
            let noise = noise.as_f64().map_or_else(
                || py_str(noise),
                |n| {
                    if noise.is_f64() {
                        py_float_repr(n)
                    } else {
                        py_str(noise)
                    }
                },
            );
            command.extend(["--noise-std".into(), noise]);
        }
        output = Some(out);
    }
    command.extend(["--".into(), path]);
    Ok((command, output))
}
