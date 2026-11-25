//! `remove_audio_invisible` / `remove_audio_profile`: process, verify the saved candidate, publish atomically.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use raiw_core::fs::suffix_lower;
use raiw_core::{StagedSibling, sha256_file};
use serde::Serialize;

use crate::codecs::raw::DEFAULT_PRESERVE_HF_HZ;
use crate::codecs::stream::stream_neural_track;
use crate::codecs::surround::{is_tagged_surround, transform_surround_stems};
use crate::codecs::{CodecKind, CodecRequest, NeuralCodec, ResolvedDevice, StereoMode, encodec, raw};
use crate::decoders::Decoder;
use crate::error::{AudioError, Result};
use crate::frames::Frames;
use crate::identify::{IdentifyOptions, identify_audio};
use crate::io::{AudioReader, write_audio};
use crate::options::{Device, Method, Profile, valid_bandwidth};
use crate::policy::{
    Permissions, ProcessingArgs, adaptive_auto_profiles, enforce_publication_gates, require_profile_runtime,
    resolve_audio_profiles, should_auto_scrub, validate_stream_for_profile,
};
use crate::quality::{AudioFidelity, audio_quality_passed, is_lossy_output, measure_fidelity};
use crate::report::{
    AudioProvenanceReport, Markers, WatermarkStatus, missing_decoder_reason, verification_outcome,
};
use crate::runtime::AudioRuntime;
use crate::spectral;

/// Progress reports `(stage, fraction)`.
pub type Progress = Arc<dyn Fn(&str, f64) + Send + Sync>;

/// A completed output or a no-target skip, with its verification scope.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AudioRemovalResult {
    pub source: PathBuf,
    pub output: Option<PathBuf>,
    pub frames_processed: usize,
    pub before: AudioProvenanceReport,
    pub after: Option<AudioProvenanceReport>,
    pub verified_removed: Option<bool>,
    pub skipped: bool,
    pub removed_markers: Markers,
    pub method: Profile,
    pub fidelity: Option<AudioFidelity>,
    pub verification_scope: Vec<Decoder>,
    pub bandwidth_kbps: Option<f64>,
    pub vocoder_device: Option<Device>,
    pub quality_passed: Option<bool>,
}

/// Every knob of `remove_audio_invisible`.
#[derive(Clone, Default)]
pub struct RemovalRequest {
    pub args: ProcessingArgs,
    pub permissions: Permissions,
    pub stream: usize,
    pub preserve_hf_hz: Option<f64>,
    pub max_duration_seconds: Option<f64>,
    pub progress: Option<Progress>,
}

impl std::fmt::Debug for RemovalRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemovalRequest")
            .field("args", &self.args)
            .field("permissions", &self.permissions)
            .field("stream", &self.stream)
            .field("preserve_hf_hz", &self.preserve_hf_hz)
            .field("max_duration_seconds", &self.max_duration_seconds)
            .finish_non_exhaustive()
    }
}

impl RemovalRequest {
    fn report(&self, stage: &str, fraction: f64) {
        if let Some(progress) = &self.progress {
            progress(stage, fraction);
        }
    }
}

const fn device_label(device: ResolvedDevice) -> Device {
    match device {
        ResolvedDevice::Cpu => Device::Cpu,
        ResolvedDevice::Mps => Device::Mps,
        ResolvedDevice::Cuda => Device::Cuda,
    }
}

/// Reject an output that is (or links to) the source.
pub fn ensure_distinct(source: &Path, output: &Path) -> Result<()> {
    let original = std::fs::canonicalize(source).unwrap_or_else(|_| source.to_path_buf());
    let target = std::fs::canonicalize(output).unwrap_or_else(|_| absolute(output));
    if target == original {
        return Err(AudioError::invalid("Audio output must differ from the source"));
    }
    Ok(())
}

fn absolute(path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    std::env::current_dir().map_or_else(|_| path.to_path_buf(), |cwd| cwd.join(path))
}

fn skip_result(
    before: AudioProvenanceReport,
    profile: Profile,
    bandwidth: f64,
    device: Option<Device>,
) -> AudioRemovalResult {
    AudioRemovalResult {
        source: before.source.clone(),
        output: None,
        frames_processed: 0,
        before,
        after: None,
        verified_removed: None,
        skipped: true,
        removed_markers: Markers::new(),
        method: profile,
        fidelity: None,
        verification_scope: Vec::new(),
        bandwidth_kbps: profile.is_neural().then_some(bandwidth),
        vocoder_device: device,
        quality_passed: None,
    }
}

/// `remove_audio_invisible`: publish only a fully checked negative by default.
///
/// `--method auto` picks a model-specific first profile and falls back (to `combined`, or per
/// vendor cohort) only after local verification fails; every attempt reuses one hash-bound
/// source report. Failed intermediate candidates are discarded before the next profile runs.
pub fn remove_audio_invisible(
    runtime: &AudioRuntime,
    source: &Path,
    output: &Path,
    request: &RemovalRequest,
) -> Result<AudioRemovalResult> {
    let args = request.args;
    let vendor = args.vendor;
    let mut permissions = request.permissions;
    permissions.allow_unverified |= vendor.is_some();
    let mut profiles = resolve_audio_profiles(runtime, &args, permissions)?;
    ensure_distinct(source, output)?;
    if request.max_duration_seconds.is_some_and(|m| m <= 0.0) {
        return Err(AudioError::invalid("max_duration_seconds must be positive"));
    }
    request.report("identify", 0.1);
    let identify = IdentifyOptions {
        check_invisible: args.verify,
        stream: request.stream,
        scheme: args.scheme,
        metadata_markers: None,
    };
    let before = identify_audio(runtime, source, &identify)?;
    if let (Some(max), Some(duration)) = (request.max_duration_seconds, before.stream.duration) {
        let seconds = duration.to_f64();
        if seconds > max {
            return Err(AudioError::invalid(format!(
                "Audio duration {seconds:.2}s exceeds max_duration_seconds={max:.2}s"
            )));
        }
    }
    if args.method == Method::Auto {
        profiles = adaptive_auto_profiles(runtime, args.scheme, &before, vendor);
    }
    if should_auto_scrub(runtime, &before, args.method, vendor) {
        permissions.allow_unverified = true;
    }
    if !(permissions.unverified() || permissions.allow_positive)
        && before.watermark_status != WatermarkStatus::Detected
    {
        request.report("complete", 1.0);
        let profile = profiles[0];
        let device = (profile != Profile::Spectral).then_some(args.vocoder_device);
        return Ok(skip_result(before, profile, args.bandwidth_kbps, device));
    }
    if !args.verify
        || before.watermark_status != WatermarkStatus::Detected
        || (!permissions.unverified() && missing_decoder_reason(&before).is_some())
    {
        profiles.truncate(1);
    }
    let mut last: Option<AudioError> = None;
    let count = profiles.len();
    for (index, &profile) in profiles.iter().enumerate() {
        let is_last = index == count - 1;
        request.report("resynthesize", 0.35 + 0.45 * (index as f64 / count.max(1) as f64));
        let attempt = Permissions {
            force: permissions.force && is_last,
            allow_unverified: permissions.allow_unverified && is_last,
            allow_positive: permissions.allow_positive && is_last,
            allow_quality_override: permissions.allow_quality_override && is_last,
        };
        let profile_request = ProfileRequest {
            profile,
            verify: args.verify,
            stream: request.stream,
            bandwidth_kbps: args.bandwidth_kbps,
            scheme: args.scheme,
            vocoder_device: args.vocoder_device,
            permissions: attempt,
            preserve_hf_hz: request.preserve_hf_hz,
        };
        match remove_audio_profile(runtime, source, output, &profile_request, Some(&before)) {
            Ok(result) => {
                request.report("complete", 1.0);
                return Ok(result);
            }
            Err(err @ AudioError::Verification(_)) => last = Some(err),
            Err(AudioError::Unavailable(message)) => {
                if let Some(AudioError::Verification(previous)) = last {
                    let mut previous = *previous;
                    previous.reason = Some(format!("Automatic fallback unavailable: {message}"));
                    return Err(AudioError::Verification(Box::new(previous)));
                }
                return Err(AudioError::Unavailable(message));
            }
            Err(err) => return Err(err),
        }
    }
    Err(last.unwrap_or_else(|| AudioError::failed("No audio processing profile was attempted")))
}

/// One concrete profile attempt (`_remove_audio_profile`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProfileRequest {
    pub profile: Profile,
    pub verify: bool,
    pub stream: usize,
    pub bandwidth_kbps: f64,
    pub scheme: crate::options::Scheme,
    pub vocoder_device: Device,
    pub permissions: Permissions,
    pub preserve_hf_hz: Option<f64>,
}

type Stream<'a> = Box<dyn Iterator<Item = Result<Frames>> + 'a>;

/// Source frames with a running count and the sticky full-scale flag `_guard_full_scale` reads.
struct CountingReader {
    reader: AudioReader,
    count: Arc<AtomicUsize>,
    exceeded: Arc<AtomicBool>,
}

impl Iterator for CountingReader {
    type Item = Result<Frames>;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.reader.next()?;
        if let Ok(frames) = &item {
            self.count.fetch_add(frames.frames(), Ordering::Relaxed);
            if !self.exceeded.load(Ordering::Relaxed) && !frames.is_empty() && frames.peak() > 1.0 {
                self.exceeded.store(true, Ordering::Relaxed);
            }
        }
        Some(item)
    }
}

/// `_guard_full_scale`: pull marginal codec overshoot back under full scale when the source never exceeded it.
fn guard_full_scale(frames: Frames, exceeded: &AtomicBool) -> Frames {
    if exceeded.load(Ordering::Relaxed) || frames.is_empty() {
        return frames;
    }
    let peak = frames.peak();
    if peak.is_finite() && peak > 0.9999 && peak <= 1.35 {
        frames.scale(0.9999 / peak)
    } else {
        frames
    }
}

fn codec(runtime: &AudioRuntime, kind: CodecKind) -> Result<&dyn NeuralCodec> {
    runtime.codecs.get(kind).map(AsRef::as_ref).ok_or_else(|| {
        AudioError::unavailable(format!(
            "{} processing requires remove-ai-watermarks[audio-codec]",
            kind.label()
        ))
    })
}

/// `_surround_codec_full_track`: DAC/SNAC across mono, stereo, 5.1 or 7.1 with the default HF crossover.
fn raw_full_track<'a>(
    input: Stream<'a>,
    codec: &'a dyn NeuralCodec,
    sample_rate: u32,
    preserve_hf_hz: Option<f64>,
    request: CodecRequest,
) -> Stream<'a> {
    let crossover = preserve_hf_hz.or((sample_rate >= 44_100).then_some(DEFAULT_PRESERVE_HF_HZ));
    let transform = move |samples: &Frames| {
        let mut stem = |part: &Frames| {
            raw::resynthesize(codec, part, sample_rate, StereoMode::MidSide, crossover, &request)
        };
        transform_surround_stems(samples, sample_rate, &mut stem)
    };
    Box::new(stream_neural_track(
        input,
        sample_rate,
        codec.kind().label(),
        transform,
        true,
    ))
}

/// `_encodec_surround_track`: mid/side EnCodec across tagged 5.1 or 7.1 stems.
fn encodec_surround_track<'a>(
    input: Stream<'a>,
    codec: &'a dyn NeuralCodec,
    sample_rate: u32,
    request: CodecRequest,
) -> Stream<'a> {
    let transform = move |samples: &Frames| {
        let mut stem = |part: &Frames| {
            let mode = if part.channels() == 2 {
                StereoMode::MidSide
            } else {
                StereoMode::Independent
            };
            encodec::resynthesize(codec, part, sample_rate, mode, &request)
        };
        transform_surround_stems(samples, sample_rate, &mut stem)
    };
    Box::new(stream_neural_track(
        input,
        sample_rate,
        "EnCodec",
        transform,
        true,
    ))
}

/// `_remove_audio_profile`: apply one profile and verify the saved candidate before publishing.
pub fn remove_audio_profile(
    runtime: &AudioRuntime,
    source: &Path,
    output: &Path,
    request: &ProfileRequest,
    before: Option<&AudioProvenanceReport>,
) -> Result<AudioRemovalResult> {
    let permissions = request.permissions;
    let profile = request.profile;
    if !request.verify && !permissions.unverified() {
        return Err(AudioError::invalid(
            "Disabling verification requires force=True or allow_unverified=True",
        ));
    }
    if profile.is_neural() && !valid_bandwidth(request.bandwidth_kbps) {
        return Err(AudioError::invalid("EnCodec bandwidth must be 6, 12, or 24 kbps"));
    }
    let device = runtime.resolve_device(request.vocoder_device, profile);
    require_profile_runtime(runtime, profile)?;
    let tools = runtime
        .tools
        .clone()
        .ok_or_else(|| AudioError::unavailable("Audio processing requires ffmpeg"))?;
    ensure_distinct(source, output)?;
    let before = match before {
        Some(report) => report.clone(),
        None => identify_audio(
            runtime,
            source,
            &IdentifyOptions {
                check_invisible: request.verify,
                stream: request.stream,
                scheme: request.scheme,
                metadata_markers: None,
            },
        )?,
    };
    validate_stream_for_profile(&before.stream, profile)?;
    let reported_device = profile.is_raw_codec().then_some(device_label(device));
    let bandwidth = profile.is_neural().then_some(request.bandwidth_kbps);
    if !(permissions.unverified() || permissions.positive())
        && before.watermark_status != WatermarkStatus::Detected
    {
        return Ok(AudioRemovalResult {
            vocoder_device: reported_device,
            ..skip_result(before, profile, request.bandwidth_kbps, None)
        });
    }
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let info = before.stream.clone();
    let rate = info.sample_rate;
    let count = Arc::new(AtomicUsize::new(0));
    let exceeded = Arc::new(AtomicBool::new(false));
    let codec_request = CodecRequest {
        device,
        bandwidth_kbps: request.bandwidth_kbps,
    };
    let staged = StagedSibling::new(output)?;
    let reader = AudioReader::open(&tools, &before.source, &info, rate as usize * 8)?;
    let mut stream: Stream<'_> = Box::new(CountingReader {
        reader,
        count: Arc::clone(&count),
        exceeded: Arc::clone(&exceeded),
    });
    if profile.uses_spectral() {
        stream = Box::new(spectral::filter_chunks(stream, rate)?);
    }
    if profile.is_neural() {
        let encodec_codec = codec(runtime, CodecKind::Encodec)?;
        stream = if is_tagged_surround(&info) {
            encodec_surround_track(stream, encodec_codec, rate, codec_request)
        } else {
            let mode = if info.channels == 2 {
                StereoMode::MidSide
            } else {
                StereoMode::Independent
            };
            Box::new(encodec::resynthesize_chunks(
                encodec_codec,
                stream,
                rate,
                mode,
                codec_request,
            )?)
        };
    }
    if profile.is_raw_codec() {
        let kind = if profile == Profile::Dac {
            CodecKind::Dac
        } else {
            CodecKind::Snac
        };
        stream = raw_full_track(
            stream,
            codec(runtime, kind)?,
            rate,
            request.preserve_hf_hz,
            codec_request,
        );
    }
    let guard = Arc::clone(&exceeded);
    let guarded = stream.map(move |item| item.map(|frames| guard_full_scale(frames, &guard)));
    let bitexact = profile.is_raw_codec() && suffix_lower(staged.path()) == ".wav";
    let written = write_audio(&tools, &info, staged.path(), guarded, None, bitexact)?;
    if written != count.load(Ordering::Relaxed) {
        return Err(AudioError::failed(
            "Audio processing changed the decoded sample-frame count",
        ));
    }
    let fidelity = measure_fidelity(&tools, &before.source, &info, staged.path(), 0)?;
    let quality_ok = audio_quality_passed(&fidelity, is_lossy_output(staged.path()));
    let mut after = identify_audio(
        runtime,
        staged.path(),
        &IdentifyOptions {
            check_invisible: request.verify,
            stream: 0,
            scheme: request.scheme,
            metadata_markers: None,
        },
    )?;
    if !after.metadata_markers.is_empty() {
        let mut keys: Vec<&str> = after.metadata_markers.keys().map(String::as_str).collect();
        keys.sort_unstable();
        let listed: Vec<String> = keys.iter().map(|k| format!("'{k}'")).collect();
        return Err(AudioError::failed(format!(
            "AI metadata remains in the processed audio: [{}]",
            listed.join(", ")
        )));
    }
    let verified = verification_outcome(&before, &after);
    enforce_publication_gates(&before, &after, verified, quality_ok, permissions)?;
    if sha256_file(&before.source)? != before.sha256 {
        return Err(AudioError::failed("Audio source changed during processing"));
    }
    staged.commit()?;
    after.source = output.to_path_buf();
    let mut scope: Vec<Decoder> = before
        .detections
        .iter()
        .chain(&after.detections)
        .map(|d| d.scheme)
        .collect();
    scope.sort_by_key(|d| d.as_str());
    scope.dedup();
    Ok(AudioRemovalResult {
        source: before.source.clone(),
        output: Some(output.to_path_buf()),
        frames_processed: written,
        removed_markers: before.metadata_markers.clone(),
        before,
        after: Some(after),
        verified_removed: verified,
        skipped: false,
        method: profile,
        fidelity: Some(fidelity),
        verification_scope: scope,
        bandwidth_kbps: bandwidth,
        vocoder_device: reported_device,
        quality_passed: Some(quality_ok),
    })
}
