//! Invisible pixel watermark decoders and pixel forensics.
//!
//! - [`dwt_dct`] and [`invisible_watermark`]: the open DWT-DCT marks of Stable Diffusion, SDXL and
//!   FLUX.2, bit-exact with PyWavelets and OpenCV.
//! - [`invismark`]: the local Microsoft Paint InvisMark payload.
//! - [`trustmark`]: Adobe TrustMark preprocessing, BCH payload decoding and the durability verdict;
//!   the neural decoder is a [`trustmark::TrustMarkModel`] supplied by the caller.
//! - [`evidence`]: the pixel-forensics statistics of `pixel_evidence.py`.
//!
//! Inputs are decoded pixel buffers ([`raiw_cv::Mat`] / [`raiw_cv::MatView`]); file decoding with
//! OpenCV or Pillow semantics belongs to the image IO layer.

pub mod dwt_dct;
pub mod error;
pub mod invisible_watermark;
pub mod invismark;
mod math;
pub mod uuid;

pub use dwt_dct::{DecodedBits, block_bits, decode_dwt_dct, decode_dwt_dct_lengths};
pub use error::{DwtDctError, InvisMarkError, TrustMarkError};
pub use invisible_watermark::{OpenWatermark, detect_invisible_watermark};
pub use invismark::{
    LocalInvisMarkDetection, LocalInvisMarkRemoval, detect_local_invismark, disrupt_local_invismark,
    invismark_id_from_c2pa_info,
};
pub use uuid::Uuid;
