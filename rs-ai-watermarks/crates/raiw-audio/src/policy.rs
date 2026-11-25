//! Profile sequencing, argument validation and publication gates of `audio.py`.

use crate::codecs::CodecKind;
use crate::codecs::surround::{is_mono_or_stereo, is_tagged_surround};
use crate::decoders::Decoder;
use crate::error::{AudioError, Result, VerificationIncomplete};
use crate::identify::metadata_vendor;
use crate::io::AudioStreamInfo;
use crate::options::{Device, MetadataVendor, Method, Profile, Scheme, Vendor, valid_bandwidth};
use crate::report::{AudioProvenanceReport, WatermarkStatus, verification_failure_reason};
use crate::runtime::AudioRuntime;

/// Verification and permission knobs shared by every removal entry point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Permissions {
    /// Alias enabling the three split permissions on the final attempt.
    pub force: bool,
    /// Process without a local source positive and publish an unverified output.
    pub allow_unverified: bool,
    /// Publish a still-positive output.
    pub allow_positive: bool,
    /// Publish an output that failed the integrity gate.
    pub allow_quality_override: bool,
}

impl Permissions {
    pub const fn unverified(self) -> bool {
        self.force || self.allow_unverified
    }

    pub const fn positive(self) -> bool {
        self.force || self.allow_positive
    }

    pub const fn quality_override(self) -> bool {
        self.force || self.allow_quality_override
    }
}

/// The processing arguments `validate_audio_arguments` checks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProcessingArgs {
    pub method: Method,
    pub scheme: Scheme,
    pub verify: bool,
    pub bandwidth_kbps: f64,
    pub vocoder_device: Device,
    pub vendor: Option<Vendor>,
}

impl Default for ProcessingArgs {
    fn default() -> Self {
        Self {
            method: Method::Auto,
            scheme: Scheme::AudioSeal,
            verify: true,
            bandwidth_kbps: 24.0,
            vocoder_device: Device::Auto,
            vendor: None,
        }
    }
}

/// `validate_audio_arguments`: candidate profiles, without touching runtime backends.
pub fn validate_audio_arguments(args: &ProcessingArgs, permissions: Permissions) -> Result<Vec<Profile>> {
    if !args.verify && !permissions.unverified() && args.vendor.is_none() {
        return Err(AudioError::invalid(
            "Disabling verification requires force=True or allow_unverified=True",
        ));
    }
    if !matches!(args.vocoder_device, Device::Auto | Device::Cpu) && args.method == Method::Spectral {
        return Err(AudioError::invalid(
            "A non-CPU vocoder device requires method=dac, snac, encodec, combined, or auto",
        ));
    }
    if args.method.uses_bandwidth() && !valid_bandwidth(args.bandwidth_kbps) {
        return Err(AudioError::invalid("EnCodec bandwidth must be 6, 12, or 24 kbps"));
    }
    Ok(match args.method.profile() {
        Some(profile) => vec![profile],
        None => args.scheme.auto_profiles().to_vec(),
    })
}

/// `_require_base_audio_runtime`.
pub fn require_base_runtime(runtime: &AudioRuntime, method: Method) -> Result<()> {
    if runtime.tools.is_none() {
        let extra = if matches!(method, Method::Dac | Method::Snac) {
            "audio-codec"
        } else {
            "audio"
        };
        return Err(AudioError::unavailable(format!(
            "Audio processing requires 'remove-ai-watermarks[{extra}]' and ffmpeg"
        )));
    }
    Ok(())
}

/// `resolve_audio_profiles`: validated candidates plus the base runtime check.
pub fn resolve_audio_profiles(
    runtime: &AudioRuntime,
    args: &ProcessingArgs,
    permissions: Permissions,
) -> Result<Vec<Profile>> {
    let profiles = validate_audio_arguments(args, permissions)?;
    require_base_runtime(runtime, args.method)?;
    Ok(profiles)
}

/// `require_audio_profile_runtime`.
pub fn require_profile_runtime(runtime: &AudioRuntime, profile: Profile) -> Result<()> {
    let missing = match profile {
        Profile::Encodec | Profile::Combined if !runtime.codec_available(CodecKind::Encodec) => {
            Some("Neural audio processing requires remove-ai-watermarks[audio-codec]")
        }
        Profile::Dac if !runtime.codec_available(CodecKind::Dac) => {
            Some("DAC processing requires remove-ai-watermarks[audio-codec]")
        }
        Profile::Snac if !runtime.codec_available(CodecKind::Snac) => {
            Some("SNAC processing requires remove-ai-watermarks[audio-codec]")
        }
        _ => None,
    };
    if let Some(message) = missing {
        return Err(AudioError::unavailable(message));
    }
    require_base_runtime(runtime, profile.into())
}

/// `validate_audio_stream_for_profile`: DAC/SNAC need mono, stereo or a tagged 5.1/7.1 layout.
pub fn validate_stream_for_profile(info: &AudioStreamInfo, profile: Profile) -> Result<()> {
    if profile.is_raw_codec() && !(is_mono_or_stereo(info) || is_tagged_surround(info)) {
        let label = if profile == Profile::Dac { "DAC" } else { "SNAC" };
        return Err(AudioError::invalid(format!(
            "{label} requires a mono or stereo (or tagged 5.1/7.1) channel layout"
        )));
    }
    Ok(())
}

fn dac_layout(runtime: &AudioRuntime, info: &AudioStreamInfo) -> bool {
    (is_mono_or_stereo(info) || is_tagged_surround(info)) && runtime.codec_available(CodecKind::Dac)
}

/// `should_auto_scrub_audio`: an oracle-certified proprietary vendor in pre-strip metadata.
pub fn should_auto_scrub(
    runtime: &AudioRuntime,
    before: &AudioProvenanceReport,
    method: Method,
    vendor: Option<Vendor>,
) -> bool {
    method == Method::Auto
        && vendor.is_none()
        && matches!(
            metadata_vendor(runtime, &before.metadata_markers),
            Some(MetadataVendor::GoogleSynthId | MetadataVendor::ElevenLabs)
        )
        && dac_layout(runtime, &before.stream)
}

/// `adaptive_auto_profiles`: minimal-distortion candidates per detected scheme and vendor cohort.
pub fn adaptive_auto_profiles(
    runtime: &AudioRuntime,
    scheme: Scheme,
    before: &AudioProvenanceReport,
    vendor: Option<Vendor>,
) -> Vec<Profile> {
    use Profile::{Combined, Dac, Encodec, Spectral};
    enum Cohort {
        Google,
        Wideband,
        Meta,
        Resemble,
    }
    let cohort = match vendor {
        Some(Vendor::Google) => Some(Cohort::Google),
        Some(
            Vendor::ElevenLabs
            | Vendor::Suno
            | Vendor::Udio
            | Vendor::OpenAi
            | Vendor::Cartesia
            | Vendor::Microsoft,
        ) => Some(Cohort::Wideband),
        Some(Vendor::Meta) => Some(Cohort::Meta),
        Some(Vendor::Resemble) => Some(Cohort::Resemble),
        None => metadata_vendor(runtime, &before.metadata_markers).map(|v| match v {
            MetadataVendor::GoogleSynthId => Cohort::Google,
            _ => Cohort::Wideband,
        }),
    };
    if let Some(cohort) = cohort {
        let dac = dac_layout(runtime, &before.stream);
        return match (cohort, dac) {
            (Cohort::Google, true) => vec![Dac, Combined],
            (Cohort::Google, false) => vec![Dac],
            (Cohort::Wideband, true) => vec![Dac, Encodec, Combined],
            (Cohort::Wideband, false) => vec![Encodec, Combined],
            (Cohort::Meta, true) => vec![Spectral, Dac, Combined],
            (Cohort::Meta | Cohort::Resemble, false) => vec![Spectral, Combined],
            (Cohort::Resemble, true) => vec![Dac, Spectral, Combined],
        };
    }
    if !matches!(scheme, Scheme::All | Scheme::Auto) || before.watermark_status != WatermarkStatus::Detected {
        return scheme.auto_profiles().to_vec();
    }
    let detected: Vec<Decoder> = before
        .detections
        .iter()
        .filter(|d| d.status == WatermarkStatus::Detected)
        .map(|d| d.scheme)
        .collect();
    if !detected.is_empty()
        && detected
            .iter()
            .all(|d| matches!(d, Decoder::AudioSeal | Decoder::Perth))
    {
        return vec![Spectral, Combined];
    }
    if !detected.is_empty()
        && detected
            .iter()
            .all(|d| matches!(d, Decoder::WavMark | Decoder::SilentCipher))
    {
        return vec![Encodec, Combined];
    }
    vec![Combined]
}

/// `enforce_audio_publication_gates`: decoder verification first, then the integrity gate.
pub fn enforce_publication_gates(
    before: &AudioProvenanceReport,
    after: &AudioProvenanceReport,
    verified: Option<bool>,
    quality_ok: bool,
    permissions: Permissions,
) -> Result<()> {
    let blocked = match verified {
        Some(false) => !permissions.positive(),
        None => !permissions.unverified(),
        Some(true) => false,
    };
    let incomplete = |reason: Option<String>| {
        AudioError::Verification(Box::new(VerificationIncomplete {
            before: before.clone(),
            after: after.clone(),
            reason,
        }))
    };
    if blocked {
        return Err(incomplete(verification_failure_reason(after)));
    }
    if !permissions.quality_override() && !quality_ok {
        return Err(incomplete(Some(
            "Delivered audio failed the minimum quality gate".into(),
        )));
    }
    Ok(())
}
