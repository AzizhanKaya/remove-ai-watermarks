//! Descript Audio Codec 44.1 kHz (`descript/dac_44khz`, Transformers `DacModel` layout).
//!
//! Mirrors `_internal/audio_dac.py`: `model.encode(x, n_quantizers=3)` followed by
//! `model.decode(audio_codes=...)` on one mono stem already on the 44.1 kHz model grid.

use std::path::Path;

use burn::tensor::activation::tanh;
use burn::tensor::{Int, Tensor};

use crate::device::ComputeDevice;
use crate::error::{GpuError, Result};
use crate::hub::{Fetch, Hub, HubFile};
use crate::io::{check_mono, mono_tensor, read_finite, read_i64};
use crate::layers::residual::{ResidualUnit, UnitNaming};
use crate::layers::vq::{CodeScore, VectorQuantizer, VqSpec, residual_quantize};
use crate::layers::{Conv1d, Conv1dSpec, ConvTranspose1d, ConvTranspose1dSpec, Snake1d};
use crate::weights::{StateDict, int_tensor};

pub const MODEL_REPOSITORY: &str = "descript/dac_44khz";
pub const MODEL_REVISION: &str = "c1bc521685adf9cfe247bc39a5ca58917eda1ac4";
pub const CHECKPOINT_FILENAME: &str = "model.safetensors";
pub const CHECKPOINT_SHA256: &str = "6128ebff483a41422b0164d079a3773b0d8d82e64c4293d775994cbf8baf913a";
pub const CONFIG_SHA256: &str = "4eb55fb9af1990b8d608184ad29b70e358589719af7ea8d3c06998f7c2264a64";
pub const SAMPLE_RATE: u32 = 44_100;
pub const HOP_LENGTH: usize = 512;
pub const FIXED_QUANTIZERS: usize = 3;

pub const CHECKPOINT_FILE: HubFile<'static> =
    HubFile::new(MODEL_REPOSITORY, MODEL_REVISION, CHECKPOINT_FILENAME).with_sha256(CHECKPOINT_SHA256);
pub const CONFIG_FILE: HubFile<'static> =
    HubFile::new(MODEL_REPOSITORY, MODEL_REVISION, "config.json").with_sha256(CONFIG_SHA256);

const ENCODER_DIM: usize = 64;
const LATENT_DIM: usize = 1024;
const DECODER_DIM: usize = 1536;
const DOWNSAMPLING: [usize; 4] = [2, 4, 8, 8];
const CODEBOOKS: usize = 9;
const CODEBOOK_SIZE: usize = 1024;
const CODEBOOK_DIM: usize = 8;

/// Codebook indices, `codes[q][frame]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Codes {
    pub codebooks: Vec<Vec<u32>>,
}

impl Codes {
    pub fn frames(&self) -> usize {
        self.codebooks.first().map_or(0, Vec::len)
    }
}

#[derive(Debug, Clone)]
struct EncoderBlock {
    units: [ResidualUnit; 3],
    snake: Snake1d,
    conv: Conv1d,
}

#[derive(Debug, Clone)]
struct DecoderBlock {
    snake: Snake1d,
    conv_t: ConvTranspose1d,
    units: [ResidualUnit; 3],
}

#[derive(Debug, Clone)]
pub struct DacCodec {
    encoder_in: Conv1d,
    encoder_blocks: Vec<EncoderBlock>,
    encoder_snake: Snake1d,
    encoder_out: Conv1d,
    quantizers: Vec<VectorQuantizer>,
    decoder_in: Conv1d,
    decoder_blocks: Vec<DecoderBlock>,
    decoder_snake: Snake1d,
    decoder_out: Conv1d,
    device: ComputeDevice,
}

impl DacCodec {
    /// Load the pinned checkpoint from the hub cache only (Python's `local_files_only=True`).
    pub fn load(device: &ComputeDevice) -> Result<Self> {
        Self::load_from_hub(&Hub::from_env(), Fetch::CacheOnly, device)
    }

    pub fn load_from_hub(hub: &Hub, fetch: Fetch, device: &ComputeDevice) -> Result<Self> {
        let config = hub.resolve(&CONFIG_FILE, fetch)?;
        let checkpoint = hub.resolve(&CHECKPOINT_FILE, fetch)?;
        if config.parent() != checkpoint.parent() {
            return Err(GpuError::Config(
                "pinned DAC artifacts must share one snapshot directory".to_owned(),
            ));
        }
        Self::from_checkpoint(&checkpoint, device)
    }

    /// Load from a local `model.safetensors` without digest checks.
    pub fn from_checkpoint(path: &Path, device: &ComputeDevice) -> Result<Self> {
        let mut state = StateDict::from_safetensors(path, &[])?;
        state.fold_weight_norm()?;
        let burn = device.burn();
        let unit = |state: &mut StateDict, prefix: &str, dim: usize| -> Result<[ResidualUnit; 3]> {
            Ok([
                ResidualUnit::load(
                    state,
                    &format!("{prefix}.res_unit1"),
                    UnitNaming::Named,
                    dim,
                    1,
                    1,
                    burn,
                )?,
                ResidualUnit::load(
                    state,
                    &format!("{prefix}.res_unit2"),
                    UnitNaming::Named,
                    dim,
                    3,
                    1,
                    burn,
                )?,
                ResidualUnit::load(
                    state,
                    &format!("{prefix}.res_unit3"),
                    UnitNaming::Named,
                    dim,
                    9,
                    1,
                    burn,
                )?,
            ])
        };

        let encoder_in = Conv1d::load(
            &mut state,
            "encoder.conv1",
            Conv1dSpec::new(1, ENCODER_DIM, 7).padding(3),
            burn,
        )?;
        let mut encoder_blocks = Vec::with_capacity(DOWNSAMPLING.len());
        for (index, &stride) in DOWNSAMPLING.iter().enumerate() {
            let prefix = format!("encoder.block.{index}");
            let dim = ENCODER_DIM << (index + 1);
            let down = Conv1dSpec::new(dim / 2, dim, 2 * stride)
                .stride(stride)
                .padding(stride.div_ceil(2));
            encoder_blocks.push(EncoderBlock {
                units: unit(&mut state, &prefix, dim / 2)?,
                snake: Snake1d::load(&mut state, &format!("{prefix}.snake1"), dim / 2, burn)?,
                conv: Conv1d::load(&mut state, &format!("{prefix}.conv1"), down, burn)?,
            });
        }
        let encoder_snake = Snake1d::load(&mut state, "encoder.snake1", LATENT_DIM, burn)?;
        let encoder_out = Conv1d::load(
            &mut state,
            "encoder.conv2",
            Conv1dSpec::new(LATENT_DIM, LATENT_DIM, 3).padding(1),
            burn,
        )?;

        let spec = VqSpec {
            input_dim: LATENT_DIM,
            codebook_size: CODEBOOK_SIZE,
            codebook_dim: CODEBOOK_DIM,
            stride: 1,
            score: CodeScore::Distance,
        };
        let quantizers = (0..CODEBOOKS)
            .map(|index| {
                VectorQuantizer::load(&mut state, &format!("quantizer.quantizers.{index}"), spec, burn)
            })
            .collect::<Result<Vec<_>>>()?;

        let decoder_in = Conv1d::load(
            &mut state,
            "decoder.conv1",
            Conv1dSpec::new(LATENT_DIM, DECODER_DIM, 7).padding(3),
            burn,
        )?;
        let mut decoder_blocks = Vec::with_capacity(DOWNSAMPLING.len());
        for (index, &stride) in DOWNSAMPLING.iter().rev().enumerate() {
            let prefix = format!("decoder.block.{index}");
            let (input, output) = (DECODER_DIM >> index, DECODER_DIM >> (index + 1));
            let up = ConvTranspose1dSpec::new(input, output, 2 * stride, stride).padding(stride.div_ceil(2));
            decoder_blocks.push(DecoderBlock {
                snake: Snake1d::load(&mut state, &format!("{prefix}.snake1"), input, burn)?,
                conv_t: ConvTranspose1d::load(&mut state, &format!("{prefix}.conv_t1"), up, burn)?,
                units: unit(&mut state, &prefix, output)?,
            });
        }
        let last = DECODER_DIM >> DOWNSAMPLING.len();
        let decoder_snake = Snake1d::load(&mut state, "decoder.snake1", last, burn)?;
        let decoder_out = Conv1d::load(
            &mut state,
            "decoder.conv2",
            Conv1dSpec::new(last, 1, 7).padding(3),
            burn,
        )?;
        state.finish()?;
        Ok(Self {
            encoder_in,
            encoder_blocks,
            encoder_snake,
            encoder_out,
            quantizers,
            decoder_in,
            decoder_blocks,
            decoder_snake,
            decoder_out,
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &ComputeDevice {
        &self.device
    }

    /// Codebook indices of mono 44.1 kHz samples using the first `n_quantizers` codebooks.
    pub fn encode(&self, samples: &[f32], n_quantizers: usize) -> Result<Codes> {
        let (codes, _) = self.quantize(samples, n_quantizers)?;
        let codebooks = codes
            .into_iter()
            .map(|code| Ok(read_i64(code)?.into_iter().map(|c| c as u32).collect()))
            .collect::<Result<Vec<Vec<u32>>>>()?;
        Ok(Codes { codebooks })
    }

    /// Decode codebook indices back to audio (`frames * 512` samples).
    pub fn decode(&self, codes: &Codes) -> Result<Vec<f32>> {
        let frames = codes.frames();
        if codes.codebooks.is_empty() || codes.codebooks.len() > CODEBOOKS || frames == 0 {
            return Err(GpuError::Input(format!(
                "DAC codes must hold 1..={CODEBOOKS} nonempty codebooks"
            )));
        }
        let burn = self.device.burn();
        let mut quantized: Option<Tensor<3>> = None;
        for (quantizer, row) in self.quantizers.iter().zip(&codes.codebooks) {
            if row.len() != frames || row.iter().any(|&c| c as usize >= CODEBOOK_SIZE) {
                return Err(GpuError::Input(
                    "DAC codebooks must share one length and index < 1024".to_owned(),
                ));
            }
            let indices: Tensor<2, Int> =
                int_tensor(row.iter().map(|&c| i64::from(c)).collect(), [1, frames], burn);
            let part = quantizer.decode(indices);
            quantized = Some(match quantized {
                Some(sum) => sum + part,
                None => part,
            });
        }
        let quantized = quantized.ok_or_else(|| GpuError::Input("empty DAC codes".to_owned()))?;
        read_finite(self.decode_latent(quantized))
    }

    /// Fixed-profile resynthesis: encode with three codebooks and decode, all on device.
    pub fn reconstruct(&self, samples: &[f32]) -> Result<Vec<f32>> {
        let (_, quantized) = self.quantize(samples, FIXED_QUANTIZERS)?;
        read_finite(self.decode_latent(quantized))
    }

    fn quantize(&self, samples: &[f32], n_quantizers: usize) -> Result<(Vec<Tensor<2, Int>>, Tensor<3>)> {
        check_mono(samples, "DAC")?;
        if n_quantizers == 0 || n_quantizers > CODEBOOKS {
            return Err(GpuError::Input(format!(
                "DAC n_quantizers must be between 1 and {CODEBOOKS}"
            )));
        }
        if samples.len() < HOP_LENGTH {
            return Err(GpuError::Input(format!(
                "DAC needs at least {HOP_LENGTH} samples"
            )));
        }
        let latent = self.encode_latent(mono_tensor(samples, self.device.burn()));
        Ok(residual_quantize(&self.quantizers[..n_quantizers], &latent))
    }

    fn encode_latent(&self, x: Tensor<3>) -> Tensor<3> {
        let mut h = self.encoder_in.forward(x);
        for block in &self.encoder_blocks {
            h = block.units.iter().fold(h, |h, unit| unit.forward(h));
            h = block.conv.forward(block.snake.forward(h));
        }
        self.encoder_out.forward(self.encoder_snake.forward(h))
    }

    fn decode_latent(&self, z: Tensor<3>) -> Tensor<3> {
        let mut h = self.decoder_in.forward(z);
        for block in &self.decoder_blocks {
            h = block.conv_t.forward(block.snake.forward(h));
            h = block.units.iter().fold(h, |h, unit| unit.forward(h));
        }
        tanh(self.decoder_out.forward(self.decoder_snake.forward(h)))
    }
}
