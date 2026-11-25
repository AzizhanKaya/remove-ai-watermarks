//! `identify_audio`: hash-bound inspection of one audio track with the installed decoders.

use std::io::Read;
use std::path::{Path, PathBuf};

use raiw_core::constants::AUDIO_OUTPUT_EXTENSIONS;
use raiw_core::fs::suffix_lower;
use raiw_core::sha256_file;

use crate::codecs::surround::is_mono_or_stereo;
use crate::decoders::Decoder;
use crate::error::{AudioError, Result};
use crate::io::{AudioStreamInfo, probe_audio};
use crate::options::{MetadataVendor, Scheme};
use crate::report::{AudioDetection, AudioProvenanceReport, Markers, WatermarkStatus, aggregate};
use crate::runtime::AudioRuntime;
use crate::scan::scan_audio_multi;

/// Containers with structural walkers in the metadata subsystem.
const STRUCTURED_FORMATS: [&str; 11] = [
    "wav", "mov", "mp4", "m4a", "3gp", "3g2", "mj2", "matroska", "webm", "flv", "avi",
];
/// `SYNTHID_EDIT_SIGNERS`: re-signers that keep Google identity without asserting a SynthID mark.
const SYNTHID_EDIT_SIGNERS: [&str; 1] = ["YouTube"];

/// Options of one `identify_audio` call.
#[derive(Debug, Clone, PartialEq)]
pub struct IdentifyOptions {
    pub check_invisible: bool,
    pub stream: usize,
    pub scheme: Scheme,
    /// Pre-read metadata markers (`_metadata_markers`), skipping a second metadata read.
    pub metadata_markers: Option<Markers>,
}

impl Default for IdentifyOptions {
    fn default() -> Self {
        Self {
            check_invisible: true,
            stream: 0,
            scheme: Scheme::AudioSeal,
            metadata_markers: None,
        }
    }
}

/// `_source(source, stream)`: resolve, probe and reject structurally unreadable containers.
pub fn resolve_source(
    runtime: &AudioRuntime,
    source: &Path,
    stream: usize,
) -> Result<(PathBuf, AudioStreamInfo)> {
    let path = std::fs::canonicalize(source).unwrap_or_else(|_| source.to_path_buf());
    let tools = runtime
        .tools
        .as_ref()
        .ok_or_else(|| AudioError::failed("ffprobe is required for audio processing"))?;
    let info = probe_audio(tools, &path, stream)?;
    let mut head = Vec::with_capacity(12);
    std::fs::File::open(&path)?.take(12).read_to_end(&mut head)?;
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let structured = info
        .format_name
        .split(',')
        .any(|f| STRUCTURED_FORMATS.contains(&f));
    if structured && !head.starts_with(b"RF64") {
        if let Some(reason) = runtime.metadata.unreadable_reason(&path) {
            return Err(AudioError::invalid(format!("cannot read {name}: {reason}")));
        }
    } else if head.starts_with(b"FORM") && head.len() == 12 {
        let declared = u64::from(u32::from_be_bytes([head[4], head[5], head[6], head[7]])) + 8;
        if declared > std::fs::metadata(&path)?.len() {
            return Err(AudioError::invalid(format!(
                "cannot read {name}: truncated AIFF container"
            )));
        }
    }
    Ok((path, info))
}

fn marker<'a>(markers: &'a Markers, key: &str) -> &'a str {
    markers.get(key).map_or("", String::as_str)
}

/// `_audio_metadata_vendor`: proprietary AI-audio waveform vendors from pre-strip metadata.
pub fn metadata_vendor(runtime: &AudioRuntime, markers: &Markers) -> Option<MetadataVendor> {
    if markers.is_empty() {
        return None;
    }
    let identity = ["claim_generator", "issuer", "ai_tool", "audio_generator"]
        .iter()
        .map(|k| marker(markers, k))
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    if identity.contains("elevenlabs") || identity.contains("eleven labs") {
        return Some(MetadataVendor::ElevenLabs);
    }
    if identity.contains("cartesia") {
        return Some(MetadataVendor::Cartesia);
    }
    if identity.contains("suno") {
        return Some(MetadataVendor::Suno);
    }
    if identity.contains("openai") && runtime.metadata.markers_claim_ai(markers) {
        return Some(MetadataVendor::OpenAi);
    }
    let synthid = marker(markers, "synthid_watermark").to_lowercase();
    let binding = format!(
        "{} {}",
        marker(markers, "soft_binding_algorithm"),
        marker(markers, "soft_binding")
    )
    .to_lowercase();
    if synthid.contains("google") || (binding.contains("google") && binding.contains("synthid")) {
        return Some(MetadataVendor::GoogleSynthId);
    }
    let google = identity.contains("google")
        && !SYNTHID_EDIT_SIGNERS
            .iter()
            .any(|signer| identity.contains(&signer.to_lowercase()));
    if google && (markers.contains_key("synthid_watermark") || runtime.metadata.markers_claim_ai(markers)) {
        return Some(MetadataVendor::GoogleSynthId);
    }
    None
}

/// `_recommend_audio_recipe`: an explicit cleaning command without claiming hidden marks exist.
pub fn recommend_recipe(
    runtime: &AudioRuntime,
    source: &Path,
    info: &AudioStreamInfo,
    markers: &Markers,
    status: WatermarkStatus,
) -> Option<String> {
    let suffix = suffix_lower(source);
    let in_ext = if suffix.is_empty() {
        ".wav".to_owned()
    } else {
        suffix
    };
    let out_ext = if AUDIO_OUTPUT_EXTENSIONS.contains(&in_ext.as_str()) {
        in_ext.as_str()
    } else {
        ".wav"
    };
    let vendor = metadata_vendor(runtime, markers);
    if vendor == Some(MetadataVendor::ElevenLabs)
        || (vendor == Some(MetadataVendor::GoogleSynthId) && is_mono_or_stereo(info))
    {
        let method = if vendor == Some(MetadataVendor::ElevenLabs) {
            "auto"
        } else {
            "dac"
        };
        let permission = if status == WatermarkStatus::Detected {
            ""
        } else {
            " --allow-unverified"
        };
        return Some(format!(
            "remove-ai-watermarks audio all input{in_ext} -o clean{out_ext} --invisible --method {method}{permission}"
        ));
    }
    if status == WatermarkStatus::Detected {
        return Some(format!(
            "remove-ai-watermarks audio all input{in_ext} -o clean{out_ext} --invisible"
        ));
    }
    runtime
        .metadata
        .markers_claim_ai(markers)
        .then(|| format!("remove-ai-watermarks audio metadata input{in_ext} --remove -o clean{in_ext}"))
}

/// `identify_audio(source, check_invisible=..., stream=..., scheme=...)`.
///
/// Channel checks and, for multichannel sources, a mono downmix are separate. Whole-clip decoding
/// is bounded; skipping it never becomes a complete negative. `scheme=auto` scans every installed
/// compatible decoder without requiring absent optional extras.
pub fn identify_audio(
    runtime: &AudioRuntime,
    source: &Path,
    options: &IdentifyOptions,
) -> Result<AudioProvenanceReport> {
    let (path, info) = resolve_source(runtime, source, options.stream)?;
    let digest = sha256_file(&path)?;
    let markers = match &options.metadata_markers {
        Some(markers) => markers.clone(),
        None => runtime.metadata.ai_metadata(&path)?,
    };
    let mut detections: Vec<AudioDetection> = Vec::new();
    let mut caveats: Vec<String> = Vec::new();
    let scheme = options.scheme;
    if !options.check_invisible {
        caveats.push("Audio watermark decoding was skipped.".into());
    } else if let Some(tools) = runtime.tools.as_ref() {
        let requested = scheme.requested();
        let mut skipped_auto: Vec<Decoder> = Vec::new();
        let mut active: Vec<Decoder> = Vec::new();
        let mut by_scheme: Vec<(Decoder, Vec<AudioDetection>)> = Vec::new();
        for &name in &requested {
            if runtime.decoders.is_available(name) {
                active.push(name);
            } else if scheme == Scheme::Auto {
                skipped_auto.push(name);
            } else {
                caveats.push(format!(
                    "{name} decoding is unavailable; install remove-ai-watermarks[{}].",
                    name.extra()
                ));
                if requested.len() > 1 {
                    by_scheme.push((name, vec![AudioDetection::missing(name)]));
                }
            }
        }
        let (checks, notes) = scan_audio_multi(tools, &runtime.decoders, &path, &info, &active)?;
        for det in checks {
            match by_scheme.iter_mut().find(|(s, _)| *s == det.scheme) {
                Some((_, list)) => list.push(det),
                None => by_scheme.push((det.scheme, vec![det])),
            }
        }
        caveats.extend(notes);
        let mut found: Vec<AudioDetection> = requested
            .iter()
            .flat_map(|name| {
                by_scheme
                    .iter()
                    .filter(move |(s, _)| s == name)
                    .flat_map(|(_, list)| list.clone())
            })
            .collect();
        if scheme == Scheme::Auto {
            if skipped_auto.len() == requested.len() {
                caveats.push(
                    "audioseal decoding is unavailable; install remove-ai-watermarks[audio-detect].".into(),
                );
            } else if !skipped_auto.is_empty() {
                let names: Vec<&str> = skipped_auto.iter().map(|d| d.as_str()).collect();
                caveats.push(format!(
                    "Auto audio inspection skipped uninstalled optional decoders: {}.",
                    names.join(", ")
                ));
            }
            if found.iter().any(|d| !d.inapplicable) {
                if found.iter().any(|d| d.inapplicable) {
                    caveats.push(
                        "Auto audio inspection skipped checks whose minimum window (16000 samples / one second) \
                         exceeds the clip duration."
                            .into(),
                    );
                }
                found.retain(|d| !d.inapplicable);
            }
        }
        detections = found;
    } else {
        caveats.push(
            "Audio sample processing is unavailable; install remove-ai-watermarks[audio] and ffmpeg.".into(),
        );
    }
    if sha256_file(&path)? != digest {
        return Err(AudioError::failed("Audio source changed during inspection"));
    }
    let status = aggregate(&detections);
    let recipe = recommend_recipe(runtime, &path, &info, &markers, status);
    Ok(AudioProvenanceReport {
        source: path,
        sha256: digest,
        stream: info,
        metadata_markers: markers,
        watermark_status: status,
        detections,
        caveats,
        recommended_recipe: recipe,
    })
}
