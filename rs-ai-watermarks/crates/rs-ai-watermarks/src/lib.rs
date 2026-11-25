//! `rs-ai-watermarks`: remove visible, invisible, and provenance AI marks from images, video, audio and text.
//!
//! The facade re-exports each subsystem crate under a stable module name; cargo features
//! mirror the Python extras.

#[cfg(feature = "audio")]
pub use raiw_audio as audio;
#[cfg(feature = "metadata")]
pub use raiw_c2pa as c2pa;
#[cfg(feature = "classify")]
pub use raiw_classify as classify;
#[cfg(feature = "metadata")]
pub use raiw_container as container;
pub use raiw_core as core;
#[cfg(any(feature = "image", feature = "classify"))]
pub use raiw_cv as cv;
#[cfg(feature = "gpu")]
pub use raiw_gpu as gpu;
#[cfg(feature = "image")]
pub use raiw_identify as identify;
pub use raiw_mcp as mcp;
#[cfg(feature = "metadata")]
pub use raiw_metadata as metadata;
#[cfg(feature = "image")]
pub use raiw_pixel as pixel;
#[cfg(feature = "gpu")]
pub use raiw_regen as regen;
#[cfg(feature = "text")]
pub use raiw_text as text;
#[cfg(feature = "video")]
pub use raiw_video as video;
#[cfg(feature = "image")]
pub use raiw_visible as visible;

pub mod cli;

/// Package version, kept in lockstep with the Python distribution.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
