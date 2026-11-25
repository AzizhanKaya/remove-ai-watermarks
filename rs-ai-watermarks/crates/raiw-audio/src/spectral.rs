//! The calibrated research band-stop, streamed with context at block boundaries (`audio_spectral.py`).

use rayon::prelude::*;

use crate::chunks::{ContextTransform, transform_chunks};
use crate::dsp::{Band, Section, butter_sos, sosfiltfilt};
use crate::error::{AudioError, Result};
use crate::frames::Frames;

pub const BAND_HZ: (f64, f64) = (1000.0, 1500.0);
pub const CONTEXT_SECONDS: f64 = 0.1;

/// Zero-phase filtering of every channel with SciPy's odd extension, shortened for tiny inputs.
#[derive(Debug, Clone)]
pub struct BandStop {
    sos: Vec<Section>,
    context: usize,
}

impl BandStop {
    pub fn new(sample_rate: u32) -> Result<Self> {
        if f64::from(sample_rate) <= 2.0 * BAND_HZ.1 {
            return Err(AudioError::invalid(
                "Audio sample rate is too low for the research band-stop profile",
            ));
        }
        let sos = butter_sos(4, Band::Bandstop(BAND_HZ.0, BAND_HZ.1), f64::from(sample_rate))?;
        let context = ((f64::from(sample_rate) * CONTEXT_SECONDS).ceil() as usize).max(1);
        Ok(Self { sos, context })
    }

    pub const fn context(&self) -> usize {
        self.context
    }

    /// `sosfiltfilt(sos, padded, axis=0, padlen=min(default, len - 1))`, narrowed to float32.
    pub fn filter_block(&self, padded: &Frames) -> Result<Frames> {
        let default_pad = 3 * (2 * self.sos.len() + 1);
        let padlen = default_pad.min(padded.frames().saturating_sub(1));
        let channels: Vec<Vec<f32>> = padded
            .to_channels()
            .par_iter()
            .map(|channel| {
                sosfiltfilt(&self.sos, channel, Some(padlen))
                    .map(|y| y.into_iter().map(|v| v as f32).collect())
            })
            .collect::<std::result::Result<_, _>>()?;
        Ok(Frames::from_channels(&channels))
    }
}

/// A boxed per-block transform.
pub type BlockTransform = Box<dyn FnMut(&Frames) -> Result<Frames> + Send>;
pub type FilteredChunks<I> = ContextTransform<I, BlockTransform>;

/// `filter_audio_chunks(chunks, sample_rate)`: nonfinal blocks need at least 0.1 s of samples.
pub fn filter_chunks<I>(chunks: I, sample_rate: u32) -> Result<FilteredChunks<I>>
where
    I: Iterator<Item = Result<Frames>>,
{
    let filter = BandStop::new(sample_rate)?;
    let context = filter.context();
    let transform: BlockTransform = Box::new(move |p| filter.filter_block(p));
    transform_chunks(chunks, transform, context)
}
