//! Bounded-resolution Farneback flow and backward warps shared by every temporal stage.

use raiw_cv::{
    Border, ColorConversion, FarnebackParams, Interpolation, Mat, calc_optical_flow_farneback, cvt_color,
    remap, resize, warp::WarpElement,
};

use crate::error::Result;

const FLOW_MAX_LONG_SIDE: usize = 512;

/// A backward remap: source x and y coordinates per destination pixel.
pub type Maps = (Mat<f32>, Mat<f32>);

/// `_scaled_farneback`: flow at bounded resolution, scaled back to native geometry (2 channels).
pub fn scaled_farneback(from_gray: &Mat<u8>, to_gray: &Mat<u8>) -> Result<Mat<f32>> {
    let (width, height) = (from_gray.width(), from_gray.height());
    let long_side = width.max(height);
    if long_side <= FLOW_MAX_LONG_SIDE {
        return Ok(calc_optical_flow_farneback(
            from_gray,
            to_gray,
            FarnebackParams::default(),
        )?);
    }
    let scale = FLOW_MAX_LONG_SIDE as f64 / long_side as f64;
    let work_w = crate::py::round_int(width as f64 * scale).max(16) as usize;
    let work_h = crate::py::round_int(height as f64 * scale).max(16) as usize;
    let small_from = resize(from_gray, work_w, work_h, Interpolation::Area)?;
    let small_to = resize(to_gray, work_w, work_h, Interpolation::Area)?;
    let small = calc_optical_flow_farneback(&small_from, &small_to, FarnebackParams::default())?;
    let mut full = resize(&small, width, height, Interpolation::Linear)?;
    let sx = (width as f64 / work_w as f64) as f32;
    let sy = (height as f64 / work_h as f64) as f32;
    for pair in full.data_mut().chunks_exact_mut(2) {
        pair[0] *= sx;
        pair[1] *= sy;
    }
    Ok(full)
}

/// `_flow_field`.
pub fn flow_field(from_gray: &Mat<u8>, to_gray: &Mat<u8>) -> Result<Mat<f32>> {
    scaled_farneback(from_gray, to_gray)
}

/// `_displacement_to_maps`: add the pixel grid to a 2-channel displacement field.
pub fn displacement_to_maps(flow: &Mat<f32>) -> Maps {
    let (width, height) = (flow.width(), flow.height());
    let data = flow.data();
    let mut map_x = Vec::with_capacity(width * height);
    let mut map_y = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) * 2;
            map_x.push(data[i] + x as f32);
            map_y.push(data[i + 1] + y as f32);
        }
    }
    (
        Mat::from_vec(width, height, 1, map_x).unwrap_or_else(|_| Mat::zeros(width, height, 1)),
        { Mat::from_vec(width, height, 1, map_y).unwrap_or_else(|_| Mat::zeros(width, height, 1)) },
    )
}

/// `_backward_map` / `backward_motion_map`: a remap from the previous frame into current coordinates.
pub fn backward_motion_map(current_gray: &Mat<u8>, previous_gray: &Mat<u8>) -> Result<Maps> {
    Ok(displacement_to_maps(&scaled_farneback(
        current_gray,
        previous_gray,
    )?))
}

/// `_backward_warp`: apply a precomputed backward map with reflected borders.
pub fn backward_warp<T: WarpElement>(
    image: &Mat<T>,
    maps: &Maps,
    interpolation: Interpolation,
) -> Result<Mat<T>> {
    Ok(remap(image, &maps.0, &maps.1, interpolation, Border::Reflect)?)
}

/// `_motion_residual` / `motion_residual_energy`: mean absolute residual after warping `previous`.
pub fn motion_residual_energy(current: &Mat<f32>, previous: &Mat<f32>, maps: &Maps) -> Result<f64> {
    let warped = backward_warp(previous, maps, Interpolation::Linear)?;
    let total: f64 = current
        .data()
        .iter()
        .zip(warped.data())
        .map(|(a, b)| f64::from((a - b).abs()))
        .sum();
    Ok(total / current.data().len().max(1) as f64)
}

/// `_to_gray_u8` for 8-bit frames.
pub fn gray_u8(frame: &Mat<u8>) -> Result<Mat<u8>> {
    if frame.channels() == 1 {
        return Ok(frame.clone());
    }
    Ok(cvt_color(frame, ColorConversion::BgrToGray)?)
}

/// `_to_gray_u8` for 16-bit frames: gray, then the high byte.
pub fn gray_u16_as_u8(frame: &Mat<u16>) -> Result<Mat<u8>> {
    let gray = if frame.channels() == 1 {
        frame.clone()
    } else {
        cvt_color(frame, ColorConversion::BgrToGray)?
    };
    Ok(gray.map(|v| (v >> 8) as u8))
}
