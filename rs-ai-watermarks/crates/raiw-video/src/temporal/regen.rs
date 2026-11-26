//! Motion-gated 3-frame stabilization of regenerated frames (`iter_stabilized_regenerated_frames`).
//!
//! Source frames only estimate flow and gate occlusions; source pixels never enter the output.

use raiw_cv::{Interpolation, Mat};

use super::flow::{backward_warp, displacement_to_maps, flow_field};
use super::pixel::{VideoPixel, depth_bounds};
use crate::error::{Result, VideoError};

/// Stabilization controls (`neighbor_weight=0.25`, `fb_tolerance_px=1.5`, ...).
#[derive(Clone, Copy, Debug)]
pub struct RegenStabilizeOptions {
    pub neighbor_weight: f32,
    pub fb_tolerance: f32,
    pub photometric_tolerance: f64,
    pub min_valid_fraction: f64,
}

impl Default for RegenStabilizeOptions {
    fn default() -> Self {
        Self {
            neighbor_weight: 0.25,
            fb_tolerance: 1.5,
            photometric_tolerance: 14.0,
            min_valid_fraction: 0.35,
        }
    }
}

impl RegenStabilizeOptions {
    fn validate(&self) -> Result<()> {
        if !(0.0..=0.35).contains(&self.neighbor_weight) {
            return Err(VideoError::value("Neighbor weight must be between 0 and 0.35"));
        }
        if self.fb_tolerance <= 0.0 {
            return Err(VideoError::value(
                "Forward-backward flow tolerance must be positive",
            ));
        }
        if self.photometric_tolerance <= 0.0 {
            return Err(VideoError::value("Photometric tolerance must be positive"));
        }
        if !(0.0..=1.0).contains(&self.min_valid_fraction) {
            return Err(VideoError::value(
                "Minimum valid pixel fraction must be between 0 and 1",
            ));
        }
        Ok(())
    }
}

struct Entry<T> {
    raw_source: Mat<T>,
    raw_regenerated: Mat<T>,
    gray: Mat<u8>,
    source: Mat<f32>,
    regenerated: Mat<f32>,
}

fn prepare<T: VideoPixel>(
    source: Mat<T>,
    regenerated: Mat<T>,
    shape: Option<(usize, usize)>,
) -> Result<Entry<T>> {
    if source.channels() != 3 || regenerated.channels() != 3 {
        return Err(VideoError::value(
            "Stabilization expects 3-channel uint8 or uint16 video frames",
        ));
    }
    let expected = shape.unwrap_or_else(|| regenerated.size());
    if source.size() != expected || regenerated.size() != expected {
        return Err(VideoError::value(
            "All source and regenerated frames must share identical geometry",
        ));
    }
    Ok(Entry {
        gray: T::gray_u8(&source)?,
        source: source.convert(),
        regenerated: regenerated.convert(),
        raw_source: source,
        raw_regenerated: regenerated,
    })
}

/// `_neighbor_blend_term`: a per-pixel weight and the warped neighbor regeneration.
fn neighbor_term(
    current_source: &Mat<f32>,
    neighbor_source: &Mat<f32>,
    neighbor_regenerated: &Mat<f32>,
    flow_to: &Mat<f32>,
    flow_from: &Mat<f32>,
    options: &RegenStabilizeOptions,
    photometric_tolerance: f32,
) -> Result<Option<(Vec<f32>, Mat<f32>)>> {
    let maps = displacement_to_maps(flow_to);
    let back = backward_warp(flow_from, &maps, Interpolation::Linear)?;
    let warped_source = backward_warp(neighbor_source, &maps, Interpolation::Linear)?;
    let pixels = flow_to.width() * flow_to.height();
    let (fd, bd, cs, ws) = (
        flow_to.data(),
        back.data(),
        current_source.data(),
        warped_source.data(),
    );
    let mut weights = Vec::with_capacity(pixels);
    let mut valid = 0usize;
    for i in 0..pixels {
        let (u, v) = (fd[i * 2] + bd[i * 2], fd[i * 2 + 1] + bd[i * 2 + 1]);
        let fb = (u * u + v * v).sqrt();
        let photo = ((cs[i * 3] - ws[i * 3]).abs()
            + (cs[i * 3 + 1] - ws[i * 3 + 1]).abs()
            + (cs[i * 3 + 2] - ws[i * 3 + 2]).abs())
            / 3.0;
        if fb <= options.fb_tolerance && photo <= photometric_tolerance {
            valid += 1;
        }
        let fb_conf = (1.0 - fb / options.fb_tolerance).clamp(0.0, 1.0);
        let photo_conf = (1.0 - photo / photometric_tolerance).clamp(0.0, 1.0);
        weights.push(options.neighbor_weight * fb_conf * photo_conf);
    }
    if (valid as f64 / pixels.max(1) as f64) < options.min_valid_fraction {
        return Ok(None);
    }
    Ok(Some((
        weights,
        backward_warp(neighbor_regenerated, &maps, Interpolation::Linear)?,
    )))
}

type Flows = (Mat<f32>, Mat<f32>);

#[allow(clippy::too_many_arguments)]
fn blend_triplet<T: VideoPixel>(
    previous: Option<(&Mat<f32>, &Mat<f32>)>,
    previous_flows: Option<&Flows>,
    current_source: &Mat<f32>,
    current_regenerated: &Mat<f32>,
    next: Option<(&Mat<f32>, &Mat<f32>)>,
    next_flows: Option<&Flows>,
    options: &RegenStabilizeOptions,
) -> Result<Mat<T>> {
    let (tolerance, peak) = depth_bounds::<T>(options.photometric_tolerance);
    let mut accum = current_regenerated.clone();
    let mut add = |term: Option<(Vec<f32>, Mat<f32>)>| {
        if let Some((weights, warped)) = term {
            let base = current_regenerated.data();
            for (i, value) in accum.data_mut().iter_mut().enumerate() {
                *value += weights[i / 3] * (warped.data()[i] - base[i]);
            }
        }
    };
    if let (Some((src, reg)), Some((forward_prev_to_curr, backward_curr_to_prev))) =
        (previous, previous_flows)
    {
        add(neighbor_term(
            current_source,
            src,
            reg,
            backward_curr_to_prev,
            forward_prev_to_curr,
            options,
            tolerance as f32,
        )?);
    }
    if let (Some((src, reg)), Some((forward_curr_to_next, backward_next_to_curr))) = (next, next_flows) {
        add(neighbor_term(
            current_source,
            src,
            reg,
            forward_curr_to_next,
            backward_next_to_curr,
            options,
            tolerance as f32,
        )?);
    }
    let data: Vec<T> = accum
        .data()
        .iter()
        .map(|v| T::from_f64(f64::from(v.round_ties_even()).clamp(0.0, peak)))
        .collect();
    Ok(Mat::from_vec(accum.width(), accum.height(), 3, data)?)
}

/// Push-style `iter_stabilized_regenerated_frames`: feed `(source, regenerated)` pairs and
/// receive `(source, stabilized)` with one frame of latency.
pub struct RegeneratedStabilizer<T> {
    options: RegenStabilizeOptions,
    shape: Option<(usize, usize)>,
    current: Option<Entry<T>>,
    previous: Option<(Mat<f32>, Mat<f32>)>,
    previous_flows: Option<Flows>,
}

impl<T> std::fmt::Debug for RegeneratedStabilizer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegeneratedStabilizer")
            .field("shape", &self.shape)
            .finish_non_exhaustive()
    }
}

impl<T: VideoPixel> RegeneratedStabilizer<T> {
    pub fn new(options: RegenStabilizeOptions) -> Result<Self> {
        options.validate()?;
        Ok(Self {
            options,
            shape: None,
            current: None,
            previous: None,
            previous_flows: None,
        })
    }

    pub fn push(&mut self, source: Mat<T>, regenerated: Mat<T>) -> Result<Option<(Mat<T>, Mat<T>)>> {
        let next = prepare(source, regenerated, self.shape)?;
        self.shape = Some(next.regenerated.size());
        let Some(current) = self.current.take() else {
            self.current = Some(next);
            return Ok(None);
        };
        if self.options.neighbor_weight == 0.0 {
            self.current = Some(next);
            return Ok(Some((current.raw_source, current.raw_regenerated)));
        }
        let next_flows = (
            flow_field(&current.gray, &next.gray)?,
            flow_field(&next.gray, &current.gray)?,
        );
        let out = blend_triplet::<T>(
            self.previous.as_ref().map(|(s, r)| (s, r)),
            self.previous_flows.as_ref(),
            &current.source,
            &current.regenerated,
            Some((&next.source, &next.regenerated)),
            Some(&next_flows),
            &self.options,
        )?;
        self.previous = Some((current.source, current.regenerated));
        self.previous_flows = Some(next_flows);
        self.current = Some(next);
        Ok(Some((current.raw_source, out)))
    }

    pub fn finish(mut self) -> Result<Option<(Mat<T>, Mat<T>)>> {
        let Some(current) = self.current.take() else {
            return Ok(None);
        };
        if self.previous.is_none() || self.options.neighbor_weight == 0.0 {
            return Ok(Some((current.raw_source, current.raw_regenerated)));
        }
        let out = blend_triplet::<T>(
            self.previous.as_ref().map(|(s, r)| (s, r)),
            self.previous_flows.as_ref(),
            &current.source,
            &current.regenerated,
            None,
            None,
            &self.options,
        )?;
        Ok(Some((current.raw_source, out)))
    }
}
