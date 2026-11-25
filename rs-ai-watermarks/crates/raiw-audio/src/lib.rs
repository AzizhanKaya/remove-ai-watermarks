//! Audio watermark inspection and processing (port of `audio.py` and `_internal/audio_*.py`).
//!
//! Neural forward passes sit behind [`decoders`] and [`codecs`] traits (implemented by `raiw-gpu`
//! under the `gpu` feature); everything around them — ffmpeg IO, SciPy-exact resampling and
//! filtering, quality gates, scanning, verification and publication — lives here.

pub mod chunks;
pub mod clock;
pub mod codecs;
pub mod decoders;
pub mod dsp;
pub mod error;
pub mod frames;
pub mod identify;
pub mod io;
pub mod metadata;
pub mod ops;
pub mod options;
pub mod perceptual;
pub mod policy;
pub mod quality;
pub mod remove;
pub mod report;
pub mod runtime;
pub mod scan;
pub mod spectral;

pub use clock::{ClockError, ClockRecord, RationalClock};
pub use error::{AudioError, Result, VerificationIncomplete};
pub use frames::{Frames, FramesError};
pub use identify::{IdentifyOptions, identify_audio};
pub use ops::{
    AllOptions, AudioAllResult, AudioBatchItem, AudioBatchSummary, AudioMetadataReport, AudioMetadataResult,
    BatchMode, inspect_audio_metadata, remove_audio_all, remove_audio_batch, remove_audio_metadata,
};
pub use options::{Device, Method, Profile, Scheme, Vendor};
pub use policy::{Permissions, ProcessingArgs};
pub use remove::{
    AudioRemovalResult, ProfileRequest, RemovalRequest, remove_audio_invisible, remove_audio_profile,
};
pub use report::{AudioDetection, AudioProvenanceReport, AudioWindowReading, Channel, WatermarkStatus};
pub use runtime::{Accelerators, AudioRuntime};
