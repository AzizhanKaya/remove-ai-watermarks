//! Video architectures: the 2-D frame VAE shared by the regeneration profiles and the
//! VideoSeal-family message detectors.

pub mod tiling;
pub mod vae;
pub mod videoseal;
pub mod wan;

pub use vae::SdVae;
pub use videoseal::{Scheme, VideoMessageDetector};
