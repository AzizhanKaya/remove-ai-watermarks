//! raiw-gpu burn models behind the neural traits (`gpu` feature).
//!
//! Models load lazily on first use and stay in a shared [`ModelCache`]: decoders (small) may be
//! resident together, a codec is resident alone, and a model on another device evicts the rest.

use std::any::Any;
use std::sync::{Arc, Mutex, OnceLock};

use raiw_gpu::models::audio::{
    AudioSealDetector, DacCodec, EncodecCodec, PerthDetector, SilentCipherDecoder, SnacCodec, WavMarkDecoder,
};
use raiw_gpu::{ComputeDevice, DeviceRequest, GpuError};

use crate::codecs::{CodecError, CodecKind, CodecRequest, CodecSet, NeuralCodec, ResolvedDevice};
use crate::decoders::silentcipher::SymbolScores;
use crate::decoders::{
    AudioSealNetwork, AudioSealOutput, DecoderError, DecoderSet, PerthNetwork, SilentCipherNetwork,
    WavMarkNetwork,
};
use crate::runtime::{Accelerators, ModelResidency};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelKey {
    AudioSeal,
    WavMark,
    Perth,
    SilentCipher,
    Codec(CodecKind),
}

impl ModelKey {
    const fn is_codec(self) -> bool {
        matches!(self, Self::Codec(_))
    }
}

type Resident = (ModelKey, DeviceRequest, Arc<dyn Any + Send + Sync>);

/// Lazily loaded models with bounded residency.
#[derive(Default)]
pub struct ModelCache {
    resident: Mutex<Vec<Resident>>,
}

impl std::fmt::Debug for ModelCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count = self.resident.lock().map_or(0, |r| r.len());
        f.debug_struct("ModelCache").field("resident", &count).finish()
    }
}

impl ModelCache {
    /// The cached model, loading it (and evicting incompatible residents) when absent.
    fn get<T, F>(&self, key: ModelKey, device: DeviceRequest, load: F) -> Result<Arc<Mutex<T>>, GpuError>
    where
        T: Send + 'static,
        F: FnOnce(&ComputeDevice) -> Result<T, GpuError>,
    {
        let mut resident = self
            .resident
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some((_, _, model)) = resident.iter().find(|(k, d, _)| *k == key && *d == device)
            && let Ok(model) = Arc::clone(model).downcast::<Mutex<T>>()
        {
            return Ok(model);
        }
        resident.retain(|(k, d, _)| !key.is_codec() && !k.is_codec() && *d == device && *k != key);
        let compute = ComputeDevice::select(device)?;
        let model = Arc::new(Mutex::new(load(&compute)?));
        resident.push((key, device, Arc::clone(&model) as Arc<dyn Any + Send + Sync>));
        Ok(model)
    }
}

impl ModelResidency for ModelCache {
    fn release(&self) {
        self.resident
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
    }
}

fn failed(err: impl std::fmt::Display) -> DecoderError {
    DecoderError::Failed(err.to_string())
}

fn lock<T>(model: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    model.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// A lazily loaded burn decoder or codec on one device request.
#[derive(Debug, Clone)]
pub struct GpuModel {
    cache: Arc<ModelCache>,
    device: DeviceRequest,
    label: String,
}

impl GpuModel {
    fn new(cache: &Arc<ModelCache>, device: DeviceRequest) -> Self {
        Self {
            cache: Arc::clone(cache),
            device,
            label: device.as_str().to_owned(),
        }
    }
}

impl AudioSealNetwork for GpuModel {
    fn detect(&self, samples: &[f32]) -> Result<AudioSealOutput, DecoderError> {
        let model = self
            .cache
            .get(ModelKey::AudioSeal, self.device, AudioSealDetector::load)
            .map_err(failed)?;
        let detection = lock(&model).detect(samples, 16_000).map_err(failed)?;
        Ok(AudioSealOutput {
            probabilities: detection.probabilities,
            message: detection.message.to_vec(),
        })
    }
}

impl WavMarkNetwork for GpuModel {
    fn decode(&self, windows: &[&[f32]]) -> Result<Vec<Vec<f32>>, DecoderError> {
        let model = self
            .cache
            .get(ModelKey::WavMark, self.device, WavMarkDecoder::load)
            .map_err(failed)?;
        let raw = lock(&model).decode(windows).map_err(failed)?;
        Ok(raw.into_iter().map(|row| row.to_vec()).collect())
    }
}

impl PerthNetwork for GpuModel {
    fn presence(&self, samples: &[f32]) -> Result<Vec<f32>, DecoderError> {
        let model = self
            .cache
            .get(ModelKey::Perth, self.device, PerthDetector::load)
            .map_err(failed)?;
        Ok(vec![lock(&model).read(samples, 16_000).map_err(failed)?.raw])
    }
}

impl SilentCipherNetwork for GpuModel {
    fn device(&self) -> &str {
        &self.label
    }

    /// librosa's default `soxr_hq` resampler is not ported; non-44.1 kHz checks stay unverified.
    fn resample(&self, _samples: &[f32], sample_rate: u32) -> Result<Vec<f32>, DecoderError> {
        Err(DecoderError::Failed(format!(
            "SilentCipher resampling from {sample_rate} Hz to 44100 Hz (librosa soxr_hq) is unavailable"
        )))
    }

    fn scores(&self, normalized: &[f32], shift: usize) -> Result<SymbolScores, DecoderError> {
        let model = self
            .cache
            .get(ModelKey::SilentCipher, self.device, SilentCipherDecoder::load)
            .map_err(failed)?;
        let tail = &normalized[shift.min(normalized.len())..];
        let scores = lock(&model).scores(tail).map_err(failed)?;
        Ok(SymbolScores {
            rows: 5,
            frames: scores.frames,
            values: scores.values,
        })
    }

    fn cpu_fallback(&self) -> Option<Arc<dyn SilentCipherNetwork>> {
        (self.device != DeviceRequest::Cpu).then(|| Arc::new(Self::new(&self.cache, DeviceRequest::Cpu)) as _)
    }
}

/// A burn codec; the device comes from each request (`--vocoder-device`).
#[derive(Debug, Clone)]
pub struct GpuCodec {
    cache: Arc<ModelCache>,
    kind: CodecKind,
}

const fn device_request(device: ResolvedDevice) -> DeviceRequest {
    match device {
        ResolvedDevice::Cpu => DeviceRequest::Cpu,
        ResolvedDevice::Mps => DeviceRequest::Mps,
        ResolvedDevice::Cuda => DeviceRequest::Cuda,
    }
}

impl NeuralCodec for GpuCodec {
    fn kind(&self) -> CodecKind {
        self.kind
    }

    fn reconstruct(&self, stem: &[f32], request: &CodecRequest) -> Result<Vec<f32>, CodecError> {
        let device = device_request(request.device);
        let key = ModelKey::Codec(self.kind);
        let err = |e: GpuError| CodecError(e.to_string());
        match self.kind {
            CodecKind::Dac => {
                let model = self.cache.get(key, device, DacCodec::load).map_err(err)?;
                lock(&model).reconstruct(stem).map_err(err)
            }
            CodecKind::Snac => {
                let model = self.cache.get(key, device, SnacCodec::load).map_err(err)?;
                lock(&model)
                    .reconstruct(stem, crate::codecs::raw::FIXED_QUANTIZERS)
                    .map_err(err)
            }
            CodecKind::Encodec => {
                let model = self.cache.get(key, device, EncodecCodec::load).map_err(err)?;
                lock(&model)
                    .reconstruct(stem, request.bandwidth_kbps)
                    .map_err(err)
            }
        }
    }
}

/// Decoders on `RAIW_AUDIO_DECODER_DEVICE` (default `cpu`, as the Python loaders; many short
/// windows of varying shape are faster there) and codecs on the requested device.
pub fn backends() -> (DecoderSet, CodecSet, Arc<ModelCache>) {
    let cache = Arc::new(ModelCache::default());
    let device = std::env::var("RAIW_AUDIO_DECODER_DEVICE")
        .ok()
        .and_then(|v| v.parse::<DeviceRequest>().ok())
        .unwrap_or(DeviceRequest::Cpu);
    let model = Arc::new(GpuModel::new(&cache, device));
    let decoders = DecoderSet {
        audioseal: Some(Arc::clone(&model) as _),
        wavmark: Some(Arc::clone(&model) as _),
        perth: Some(Arc::clone(&model) as _),
        silentcipher: Some(model as _),
    };
    let codec = |kind| {
        Some(Arc::new(GpuCodec {
            cache: Arc::clone(&cache),
            kind,
        }) as Arc<dyn NeuralCodec>)
    };
    let codecs = CodecSet {
        encodec: codec(CodecKind::Encodec),
        dac: codec(CodecKind::Dac),
        snac: codec(CodecKind::Snac),
    };
    (decoders, codecs, cache)
}

/// Discrete and Apple GPU availability, probed once per process.
pub fn accelerators() -> Accelerators {
    static PROBED: OnceLock<Accelerators> = OnceLock::new();
    *PROBED.get_or_init(|| Accelerators {
        cuda: ComputeDevice::select(DeviceRequest::Cuda).is_ok_and(|d| d.is_gpu()),
        mps: ComputeDevice::select(DeviceRequest::Mps).is_ok_and(|d| d.is_gpu()),
    })
}
