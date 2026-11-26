//! Motion-compensated helpers shared by video pipelines (port of `video_temporal.py`).

mod affine;
pub mod color;
pub mod flow;
pub mod matte;
mod motion;
pub mod pixel;
pub mod recover;
pub mod regen;
pub mod stabilize;
pub mod track;

use raiw_cv::{Border, Mat, magnitude, sobel};

pub use flow::{Maps, backward_motion_map, backward_warp, motion_residual_energy, scaled_farneback};
pub use matte::{ClipAlphaMatte, estimate_clip_alpha_matte, unblend_ycbcr_roi};
pub use pixel::VideoPixel;
pub use recover::{RecoverOptions, recover_unobscured_background};
pub use stabilize::{StabilizeOptions, stabilize_filled_frame_bidirectional};
pub use track::{detect_ai_bumper_start, track_video_regions};

use crate::error::Result;

/// `sobel_magnitude`: 3x3 float32 Sobel gradient magnitude of a single-channel image.
pub fn sobel_magnitude(gray: &Mat<u8>) -> Result<Mat<f32>> {
    let f32_gray: Mat<f32> = gray.convert();
    let gx: Mat<f32> = sobel(&f32_gray, 1, 0, 3, 1.0, 0.0, Border::default())?;
    let gy: Mat<f32> = sobel(&f32_gray, 0, 1, 3, 1.0, 0.0, Border::default())?;
    Ok(magnitude(&gx, &gy)?)
}
