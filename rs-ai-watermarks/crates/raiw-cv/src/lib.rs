//! Pure-Rust image kernels that reproduce the OpenCV 4.10 operations the Python package calls.
//!
//! Images are [`Mat<T>`]: row-major, channel-interleaved buffers. Kernels borrow their inputs as
//! [`MatView`] (a `&Mat` or a zero-copy ROI) and return owned results. Channel order is whatever the
//! caller stores; OpenCV-style call sites keep BGR, `image` crate interop keeps RGB.
// Runtime SIMD dispatch (see `simd.rs`) only works when hot loops are force-inlined.
#![allow(clippy::inline_always)]

pub mod arith;
pub mod border;
pub mod canny;
pub mod channels;
pub mod color;
pub mod components;
pub mod distance;
pub mod draw;
pub mod element;
pub mod error;
pub mod features;
pub mod fft;
pub mod filter;
pub mod flow;
pub mod image_alignment;
pub mod image_math;
pub mod image_quality;
pub mod inpaint;
pub mod interop;
pub mod mat;
pub mod morph;
pub mod resize;
mod simd;
pub mod stats;
pub mod template;
pub mod tiling;
pub mod types;
pub mod warp;

pub use arith::{abs_diff, add_weighted, bitwise_and, bitwise_not, bitwise_or, convert_scale_abs, magnitude};
pub use border::{Border, copy_make_border};
pub use canny::canny;
pub use channels::{extract_channel, flip, insert_channel, merge, split, transpose};
pub use color::{ColorConversion, cvt_color};
pub use components::{
    ComponentStats, Components, Connectivity, connected_components, connected_components_with_stats,
};
pub use distance::distance_transform_l2;
pub use draw::{LineType, bounding_rect, box_points, ellipse2poly, fill_convex_poly, fill_ellipse};
pub use element::{Depth, Element, Float, cv_round};
pub use error::{CvError, Result};
pub use features::{corner_min_eigen_val, good_features_to_track};
pub use fft::{optimal_dft_size, phase_correlate};
pub use filter::{filter_2d, gaussian_blur, laplacian, sep_filter_2d, sobel};
pub use flow::{
    FarnebackParams, LkParams, LkTrack, calc_optical_flow_farneback, calc_optical_flow_pyr_lk, pyr_down,
};
pub use inpaint::{InpaintMethod, inpaint};
pub use interop::{Decoded, Encoding, ReadMode, imdecode, imencode};
pub use mat::{Mat, MatView, MatViewMut};
pub use morph::{MorphOp, MorphShape, StructuringElement, dilate, erode, morphology_ex};
pub use resize::{resize, resize_by};
pub use stats::{
    MinMaxLoc, calc_hist, count_non_zero, mean_std_dev, min_max_loc, normalize_l2, normalize_min_max,
};
pub use template::{TemplateSearch, match_template_ccoeff_normed};
pub use types::{Interpolation, Point, Point2d, Point2f, Rect, RotatedRect, Scalar};
pub use warp::{Affine, get_rotation_matrix_2d, invert_affine, remap, warp_affine};
