//! The installed backends one audio operation runs against.

use std::sync::Arc;

use crate::codecs::{CodecKind, CodecSet, ResolvedDevice};
use crate::decoders::DecoderSet;
use crate::io::FfmpegTools;
use crate::metadata::{MetadataSeam, NoMetadata};
use crate::options::{Device, Profile};

/// Accelerators the neural backends can use.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Accelerators {
    pub cuda: bool,
    pub mps: bool,
}

/// ffmpeg, decoders, codecs and the metadata stage (Python's module-level `is_available()` state).
#[derive(Clone)]
pub struct AudioRuntime {
    /// `None` when ffmpeg/ffprobe are missing (sample processing unavailable).
    pub tools: Option<FfmpegTools>,
    pub decoders: DecoderSet,
    pub codecs: CodecSet,
    pub metadata: Arc<dyn MetadataSeam>,
    pub accelerators: Accelerators,
}

impl std::fmt::Debug for AudioRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioRuntime")
            .field("tools", &self.tools)
            .field("decoders", &self.decoders)
            .field("codecs", &self.codecs)
            .field("accelerators", &self.accelerators)
            .finish_non_exhaustive()
    }
}

impl Default for AudioRuntime {
    fn default() -> Self {
        Self::discover()
    }
}

impl AudioRuntime {
    /// System ffmpeg, no neural backends, no metadata stage.
    pub fn discover() -> Self {
        Self {
            tools: FfmpegTools::discover().ok(),
            decoders: DecoderSet::default(),
            codecs: CodecSet::default(),
            metadata: Arc::new(NoMetadata),
            accelerators: Accelerators::default(),
        }
    }

    #[must_use]
    pub fn with_decoders(mut self, decoders: DecoderSet) -> Self {
        self.decoders = decoders;
        self
    }

    #[must_use]
    pub fn with_codecs(mut self, codecs: CodecSet) -> Self {
        self.codecs = codecs;
        self
    }

    #[must_use]
    pub fn with_metadata(mut self, metadata: Arc<dyn MetadataSeam>) -> Self {
        self.metadata = metadata;
        self
    }

    #[must_use]
    pub const fn with_accelerators(mut self, accelerators: Accelerators) -> Self {
        self.accelerators = accelerators;
        self
    }

    pub fn codec_available(&self, kind: CodecKind) -> bool {
        self.codecs.get(kind).is_some()
    }

    /// `resolve_audio_device`: `auto` picks the fastest verified accelerator for neural codecs.
    pub const fn resolve_device(&self, requested: Device, profile: Profile) -> ResolvedDevice {
        match requested {
            Device::Cpu => ResolvedDevice::Cpu,
            Device::Mps => ResolvedDevice::Mps,
            Device::Cuda => ResolvedDevice::Cuda,
            Device::Auto => {
                if matches!(profile, Profile::Spectral) {
                    ResolvedDevice::Cpu
                } else if self.accelerators.cuda {
                    ResolvedDevice::Cuda
                } else if profile.is_raw_codec() && self.accelerators.mps {
                    ResolvedDevice::Mps
                } else {
                    ResolvedDevice::Cpu
                }
            }
        }
    }
}
