//! `WanTransformer3DModel` (Wan 2.1 T2V 1.3B): 1x2x2 patches, 30 blocks of RoPE self-attention,
//! text cross-attention and a GELU feed-forward, each modulated by the timestep (adaLN).
//!
//! Two block engines share the embeddings and the output head:
//!
//! - burn ops in float32 ([`BlockResidency::Streamed`] / [`BlockResidency::Resident`]), the
//!   CPU and parity path. The 5.7 GB float32 checkpoint does not fit next to activations on an
//!   8 GB display GPU, so streamed blocks are memory-mapped and uploaded one at a time;
//!   token-wise work runs in bounded chunks and attention through [`scaled_dot_product`].
//! - [`BlockResidency::Fused`] on a GPU: every block resident at f16 (or f32) and run by the
//!   CubeCL kernels ([`super::fused`]).

use std::path::PathBuf;
use std::sync::mpsc::sync_channel;

use burn::tensor::activation::{gelu_approximate, silu};
use burn::tensor::{Device, Tensor, TensorData, s};

use crate::device::ComputeDevice;
use crate::error::{GpuError, Result};
use crate::hub::{Fetch, Hub, HubFile};
use crate::io::read_finite;
use crate::layers::Linear;
use crate::weights::StateDict;

use super::attention::{LayerNorm, RmsNorm, scaled_dot_product};
use super::config::TransformerConfig;
#[cfg(feature = "gpu-runtime")]
use super::fused::{CrossKv, FusedBlock, Precision, RopeTables};
use super::prompt::{PromptEmbedding, TEXT_DIM};
use super::rope::Rope;
use super::{TRANSFORMER_CONFIG, TRANSFORMER_SHARDS};

const EPS: f64 = 1e-6;
/// Tokens per feed-forward chunk, bounding the `[tokens, ffn_dim]` intermediate.
const FFN_TOKENS: usize = 4096;

/// Where transformer block weights live between forwards, and which ops run them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlockResidency {
    /// burn ops in float32; memory-map the shards and upload one block at a time on every
    /// forward (~190 MB peak).
    #[default]
    Streamed,
    /// burn ops in float32 with all 30 blocks on the device (5.5 GB).
    Resident,
    /// GPU only: all blocks resident at `Precision` (2.7 GB at f16) and run by the CubeCL
    /// kernels (flash attention, fused GEMM epilogues and norms).
    #[cfg(feature = "gpu-runtime")]
    Fused(Precision),
}

impl BlockResidency {
    /// The fastest residency `device` runs: fused f16 kernels on a GPU, streamed burn ops on CPU.
    pub fn for_device(device: &ComputeDevice) -> Self {
        #[cfg(feature = "gpu-runtime")]
        if device.is_gpu() {
            return Self::Fused(Precision::F16);
        }
        let _ = device;
        Self::Streamed
    }
}

/// The prompt embedding projected by `condition_embedder.text_embedder`, `[tokens, 1536]`, plus
/// each fused block's cross-attention keys and values.
#[derive(Debug, Clone)]
pub struct TextContext {
    hidden: Tensor<2>,
    #[cfg(feature = "gpu-runtime")]
    cross: Vec<CrossKv>,
}

#[derive(Debug, Clone)]
struct Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(
        state: &mut StateDict,
        prefix: &str,
        config: &TransformerConfig,
        device: &Device,
    ) -> Result<Self> {
        let dim = config.dim();
        let linear = |state: &mut StateDict, name: &str| {
            Linear::load(state, &format!("{prefix}.{name}"), dim, dim, device)
        };
        Ok(Self {
            to_q: linear(state, "to_q")?,
            to_k: linear(state, "to_k")?,
            to_v: linear(state, "to_v")?,
            to_out: linear(state, "to_out.0")?,
            norm_q: RmsNorm::load(state, &format!("{prefix}.norm_q"), dim, EPS, device)?,
            norm_k: RmsNorm::load(state, &format!("{prefix}.norm_k"), dim, EPS, device)?,
            heads: config.heads,
            head_dim: config.head_dim,
        })
    }

    fn split_heads(&self, x: Tensor<2>) -> Tensor<3> {
        let tokens = x.dims()[0];
        x.reshape([tokens, self.heads, self.head_dim])
    }

    /// `[tokens, heads, head_dim]` -> `[1, heads, tokens, head_dim]`.
    fn batched(x: Tensor<3>) -> Tensor<4> {
        x.swap_dims(0, 1).unsqueeze_dim(0)
    }

    fn project_out(&self, out: Tensor<4>) -> Tensor<2> {
        let tokens = out.dims()[2];
        let merged = out.squeeze_dim::<3>(0).swap_dims(0, 1);
        self.to_out
            .forward(merged.reshape([tokens, self.heads * self.head_dim]))
    }

    /// Self-attention with RoPE on queries and keys.
    fn self_attention(&self, x: Tensor<2>, rope: &Rope) -> Tensor<2> {
        let q = rope.apply(self.split_heads(self.norm_q.forward(self.to_q.forward(x.clone()))));
        let k = rope.apply(self.split_heads(self.norm_k.forward(self.to_k.forward(x.clone()))));
        let v = self.split_heads(self.to_v.forward(x));
        self.project_out(scaled_dot_product(
            Self::batched(q),
            Self::batched(k),
            Self::batched(v),
        ))
    }

    /// Cross-attention from the tokens to the projected text context.
    fn cross_attention(&self, x: Tensor<2>, context: &Tensor<2>) -> Tensor<2> {
        let q = self.split_heads(self.norm_q.forward(self.to_q.forward(x)));
        let k = self.split_heads(self.norm_k.forward(self.to_k.forward(context.clone())));
        let v = self.split_heads(self.to_v.forward(context.clone()));
        self.project_out(scaled_dot_product(
            Self::batched(q),
            Self::batched(k),
            Self::batched(v),
        ))
    }
}

/// `WanTransformerBlock`.
#[derive(Debug, Clone)]
struct Block {
    /// `scale_shift_table` `[6, dim]`: shift, scale, gate (attention) then (feed-forward).
    table: Tensor<2>,
    norm: LayerNorm,
    attn1: Attention,
    norm2: LayerNorm,
    attn2: Attention,
    ffn_in: Linear,
    ffn_out: Linear,
}

impl Block {
    fn load(
        state: &mut StateDict,
        prefix: &str,
        config: &TransformerConfig,
        device: &Device,
    ) -> Result<Self> {
        let (dim, ffn) = (config.dim(), config.ffn_dim);
        let table: Tensor<3> = state.tensor(&format!("{prefix}.scale_shift_table"), [1, 6, dim], device)?;
        Ok(Self {
            table: table.reshape([6, dim]),
            norm: LayerNorm::plain(dim, EPS, device),
            attn1: Attention::load(state, &format!("{prefix}.attn1"), config, device)?,
            norm2: LayerNorm::load(state, &format!("{prefix}.norm2"), dim, EPS, device)?,
            attn2: Attention::load(state, &format!("{prefix}.attn2"), config, device)?,
            ffn_in: Linear::load(state, &format!("{prefix}.ffn.net.0.proj"), dim, ffn, device)?,
            ffn_out: Linear::load(state, &format!("{prefix}.ffn.net.2"), ffn, dim, device)?,
        })
    }

    fn forward(&self, h: Tensor<2>, context: &Tensor<2>, timestep: &Tensor<2>, rope: &Rope) -> Tensor<2> {
        let modulation = self.table.clone() + timestep.clone();
        let row = |i: usize| modulation.clone().slice(s![i..i + 1]);
        let x = self.norm.forward(h.clone()) * row(1).add_scalar(1.0) + row(0);
        let h = h + self.attn1.self_attention(x, rope) * row(2);
        let h = h.clone() + self.attn2.cross_attention(self.norm2.forward(h), context);
        let x = self.norm.forward(h.clone()) * row(4).add_scalar(1.0) + row(3);
        h + self.feed_forward(&x) * row(5)
    }

    fn feed_forward(&self, x: &Tensor<2>) -> Tensor<2> {
        let tokens = x.dims()[0];
        let chunks: Vec<Tensor<2>> = (0..tokens)
            .step_by(FFN_TOKENS)
            .map(|t0| {
                let t1 = (t0 + FFN_TOKENS).min(tokens);
                let chunk = x.clone().slice(s![t0..t1]);
                self.ffn_out.forward(gelu_approximate(self.ffn_in.forward(chunk)))
            })
            .collect();
        Tensor::cat(chunks, 0)
    }
}

#[derive(Debug, Clone)]
enum Blocks {
    Resident(Vec<Block>),
    Streamed(Vec<PathBuf>),
    #[cfg(feature = "gpu-runtime")]
    Fused(Vec<FusedBlock>),
}

/// The pinned `transformer/` files of one Wan checkpoint.
#[derive(Debug, Clone, Copy)]
pub struct TransformerFiles {
    pub config: HubFile<'static>,
    pub shards: &'static [HubFile<'static>],
}

/// `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`.
pub const WAN21_TRANSFORMER: TransformerFiles = TransformerFiles {
    config: TRANSFORMER_CONFIG,
    shards: &TRANSFORMER_SHARDS,
};

/// `WanTransformer3DModel` (`transformer/` of a Wan 2.1 or 2.2 Diffusers checkpoint).
#[derive(Debug, Clone)]
pub struct WanTransformer {
    config: TransformerConfig,
    patch: Linear,
    time_1: Linear,
    time_2: Linear,
    time_proj: Linear,
    text_1: Linear,
    text_2: Linear,
    out_table: Tensor<2>,
    norm_out: LayerNorm,
    proj_out: Linear,
    blocks: Blocks,
    /// Blocks run per forward (30 unless truncated for engine comparisons).
    layers: usize,
    device: ComputeDevice,
}

impl WanTransformer {
    /// Resolve the pinned shards through the hub cache (downloading if allowed).
    pub fn load(device: &ComputeDevice, residency: BlockResidency) -> Result<Self> {
        Self::load_from_hub(&Hub::from_env(), Fetch::Download, device, residency)
    }

    /// The pinned Wan 2.1 1.3B transformer.
    pub fn load_from_hub(
        hub: &Hub,
        fetch: Fetch,
        device: &ComputeDevice,
        residency: BlockResidency,
    ) -> Result<Self> {
        Self::load_files(hub, fetch, &WAN21_TRANSFORMER, device, residency)
    }

    /// The pinned transformer `files` of any supported checkpoint.
    pub fn load_files(
        hub: &Hub,
        fetch: Fetch,
        files: &TransformerFiles,
        device: &ComputeDevice,
        residency: BlockResidency,
    ) -> Result<Self> {
        let config = hub.resolve(&files.config, fetch)?;
        let shards = files
            .shards
            .iter()
            .map(|file| hub.resolve(file, fetch))
            .collect::<Result<Vec<_>>>()?;
        let snapshot = config.parent().and_then(|dir| dir.parent());
        if shards
            .iter()
            .any(|shard| shard.parent().and_then(|dir| dir.parent()) != snapshot)
        {
            return Err(GpuError::Config(
                "pinned Wan transformer artifacts must share one snapshot directory".to_owned(),
            ));
        }
        Self::from_shards(TransformerConfig::from_file(&config)?, shards, device, residency)
    }

    /// Load local `transformer/*.safetensors` shards without digest checks.
    pub fn from_shards(
        config: TransformerConfig,
        shards: Vec<PathBuf>,
        device: &ComputeDevice,
        residency: BlockResidency,
    ) -> Result<Self> {
        Self::from_shards_truncated(config, shards, device, residency, config.layers)
    }

    /// [`Self::from_shards`] keeping only the first `layers` blocks: a smaller model for
    /// comparing block engines against each other (its output is not Wan's).
    pub fn from_shards_truncated(
        config: TransformerConfig,
        shards: Vec<PathBuf>,
        device: &ComputeDevice,
        residency: BlockResidency,
        layers: usize,
    ) -> Result<Self> {
        let layers = layers.min(config.layers);
        let burn = device.burn();
        let (dim, features, patch) = (config.dim(), config.patch_features(), config.patch());
        let mut state = StateDict::from_safetensors_shards(&shards, |name| !name.starts_with("blocks."))?;
        let patch_weight = state.take(
            "patch_embedding.weight",
            &[dim, config.in_channels, 1, patch, patch],
        )?;
        let patch_bias = state.tensor("patch_embedding.bias", [dim], burn)?;
        let patch = Linear::from_host(patch_weight, features, dim, Some(patch_bias), burn);
        let linear = |state: &mut StateDict, name: &str, input: usize, output: usize| {
            Linear::load(state, &format!("condition_embedder.{name}"), input, output, burn)
        };
        let time_1 = linear(&mut state, "time_embedder.linear_1", config.freq_dim, dim)?;
        let time_2 = linear(&mut state, "time_embedder.linear_2", dim, dim)?;
        let time_proj = linear(&mut state, "time_proj", dim, 6 * dim)?;
        let text_1 = linear(&mut state, "text_embedder.linear_1", config.text_dim, dim)?;
        let text_2 = linear(&mut state, "text_embedder.linear_2", dim, dim)?;
        let out_table: Tensor<3> = state.tensor("scale_shift_table", [1, 2, dim], burn)?;
        let proj_out = Linear::load(&mut state, "proj_out", dim, config.out_channels * 4, burn)?;
        state.finish()?;
        let blocks = match residency {
            BlockResidency::Streamed => Blocks::Streamed(shards),
            BlockResidency::Resident => Blocks::Resident(
                (0..layers)
                    .map(|i| load_block(&shards, i, &config, burn))
                    .collect::<Result<Vec<_>>>()?,
            ),
            #[cfg(feature = "gpu-runtime")]
            BlockResidency::Fused(precision) => {
                if !device.is_gpu() {
                    return Err(GpuError::Config(
                        "fused Wan blocks run on the GPU kernels; use Streamed on the CPU".to_owned(),
                    ));
                }
                Blocks::Fused(
                    (0..layers)
                        .map(|i| {
                            let mut state = read_block(&shards, i)?;
                            let block = FusedBlock::load(
                                &mut state,
                                &format!("blocks.{i}"),
                                &config,
                                precision,
                                burn,
                            )?;
                            state.finish()?;
                            Ok(block)
                        })
                        .collect::<Result<Vec<_>>>()?,
                )
            }
        };
        Ok(Self {
            config,
            patch,
            time_1,
            time_2,
            time_proj,
            text_1,
            text_2,
            out_table: out_table.reshape([2, dim]),
            norm_out: LayerNorm::plain(dim, EPS, burn),
            proj_out,
            blocks,
            layers,
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &ComputeDevice {
        &self.device
    }

    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// `condition_embedder.text_embedder(prompt_embeds)`, reused by every denoising step.
    pub fn encode_text(&self, prompt: &PromptEmbedding) -> TextContext {
        let embeds: Tensor<2> = Tensor::from_data(
            TensorData::new(prompt.values().to_vec(), [prompt.tokens(), TEXT_DIM]),
            self.device.burn(),
        );
        let hidden = self.text_2.forward(gelu_approximate(self.text_1.forward(embeds)));
        TextContext {
            #[cfg(feature = "gpu-runtime")]
            cross: match &self.blocks {
                Blocks::Fused(blocks) => blocks.iter().map(|block| block.cross_kv(&hidden)).collect(),
                _ => Vec::new(),
            },
            hidden,
        }
    }

    /// True when the blocks run on the fused GPU kernels.
    pub fn is_fused(&self) -> bool {
        #[cfg(feature = "gpu-runtime")]
        if let Blocks::Fused(_) = self.blocks {
            return true;
        }
        false
    }

    /// `x·wᵀ + b` for the token-wise embeddings: the GEMM kernel on the fused path (burn's
    /// autotuned matmul on 30k-token shapes can exceed the GPU ring timeout).
    #[cfg_attr(not(feature = "gpu-runtime"), allow(clippy::unused_self))]
    fn token_linear(&self, linear: &Linear, x: Tensor<2>) -> Tensor<2> {
        #[cfg(feature = "gpu-runtime")]
        if self.is_fused() {
            return crate::kernels::linear(
                x,
                &crate::kernels::DenseWeight::torch(linear.torch_weight()),
                linear.bias_or_zeros(),
                crate::kernels::Activation::Identity,
                burn::tensor::DType::F32,
            );
        }
        linear.forward(x)
    }

    /// Flow velocity for latents `[C, T, H, W]` (`H`, `W` even) at an integer `timestep`.
    pub fn forward(
        &self,
        latents: &[f32],
        (frames, height, width): (usize, usize, usize),
        timestep: f32,
        context: &TextContext,
    ) -> Result<Vec<f32>> {
        let (channels, patch, dim) = (self.config.in_channels, self.config.patch(), self.config.dim());
        if frames == 0 || height == 0 || width == 0 || height % patch != 0 || width % patch != 0 {
            return Err(GpuError::Input(format!(
                "Wan transformer latents need even nonzero sides, got {frames}x{height}x{width}"
            )));
        }
        if latents.len() != channels * frames * height * width {
            return Err(GpuError::Input(format!(
                "Wan transformer latents hold {} values, expected {channels}x{frames}x{height}x{width}",
                latents.len()
            )));
        }
        let burn = self.device.burn();
        let (rows, cols) = (height / patch, width / patch);
        let tokens = frames * rows * cols;
        let x: Tensor<6> = Tensor::from_data(
            TensorData::new(latents.to_vec(), [channels, frames, rows, patch, cols, patch]),
            burn,
        );
        let patches = x
            .permute([1, 2, 4, 0, 3, 5])
            .reshape([tokens, self.config.patch_features()]);
        let mut h = self.token_linear(&self.patch, patches);
        let temb = self.time_embedding(timestep);
        let modulation = self.time_proj.forward(silu(temb.clone())).reshape([6, dim]);
        let grid = (frames, rows, cols);
        match &self.blocks {
            #[cfg(feature = "gpu-runtime")]
            Blocks::Fused(blocks) => {
                let rope = RopeTables::new(grid, self.config.head_dim, burn);
                for (block, cross) in blocks.iter().zip(&context.cross) {
                    h = block.forward(h, cross, &modulation, &rope);
                }
            }
            Blocks::Resident(_) | Blocks::Streamed(_) => {
                h = self.reference_blocks(h, context, &modulation, grid)?;
            }
        }
        let table = self.out_table.clone() + temb;
        let (shift, scale) = (table.clone().slice(s![0..1]), table.slice(s![1..2]));
        let h = self.output_norm(h, shift, scale);
        let out = self
            .token_linear(&self.proj_out, h)
            .reshape([frames, rows, cols, patch, patch, channels])
            .permute([5, 0, 1, 3, 2, 4])
            .reshape([channels, frames, height, width]);
        read_finite(out)
    }

    /// `norm_out(h) * (1 + scale) + shift`.
    fn output_norm(&self, h: Tensor<2>, shift: Tensor<2>, scale: Tensor<2>) -> Tensor<2> {
        #[cfg(feature = "gpu-runtime")]
        if self.is_fused() {
            let dim = self.config.dim();
            let (shift, scale) = (shift.reshape([dim]), scale.reshape([dim]));
            return crate::kernels::modulated_layer_norm(
                h,
                scale.add_scalar(1.0),
                shift,
                EPS as f32,
                burn::tensor::DType::F32,
            );
        }
        self.norm_out.forward(h) * scale.add_scalar(1.0) + shift
    }

    /// The burn-op blocks, resident or streamed one at a time from the shards.
    fn reference_blocks(
        &self,
        mut h: Tensor<2>,
        context: &TextContext,
        modulation: &Tensor<2>,
        grid: (usize, usize, usize),
    ) -> Result<Tensor<2>> {
        let burn = self.device.burn();
        let rope = Rope::new(self.config.head_dim, grid, burn);
        let modulation = modulation.clone();
        match &self.blocks {
            Blocks::Resident(blocks) => {
                for block in blocks {
                    h = block.forward(h, &context.hidden, &modulation, &rope);
                }
            }
            Blocks::Streamed(shards) => {
                let layers = self.layers;
                h = std::thread::scope(|scope| -> Result<Tensor<2>> {
                    // A reader thread maps the next block while the device runs this one
                    let (sender, receiver) = sync_channel(1);
                    scope.spawn(move || {
                        for i in 0..layers {
                            let state = read_block(shards, i);
                            let failed = state.is_err();
                            if sender.send(state).is_err() || failed {
                                break;
                            }
                        }
                    });
                    for i in 0..layers {
                        let mut state = receiver.recv().map_err(|_| {
                            GpuError::Config("Wan transformer block reader stopped".to_owned())
                        })??;
                        let block = Block::load(&mut state, &format!("blocks.{i}"), &self.config, burn)?;
                        state.finish()?;
                        h = block.forward(h, &context.hidden, &modulation, &rope);
                        // Finish this block before the next upload so only one is resident
                        self.device.sync()?;
                    }
                    Ok(h)
                })?;
            }
            #[cfg(feature = "gpu-runtime")]
            Blocks::Fused(_) => unreachable!("fused blocks run in forward"),
        }
        Ok(h)
    }

    /// `time_embedder(timesteps_proj(t))`: flipped sinusoids, Linear, SiLU, Linear -> `[1, dim]`.
    fn time_embedding(&self, timestep: f32) -> Tensor<2> {
        let freq_dim = self.config.freq_dim;
        let half = freq_dim / 2;
        let log_period = -(10_000f64.ln()) as f32;
        let args: Vec<f32> = (0..half)
            .map(|i| timestep * (log_period * i as f32 / half as f32).exp())
            .collect();
        let sinusoid: Vec<f32> = args
            .iter()
            .map(|a| a.cos())
            .chain(args.iter().map(|a| a.sin()))
            .collect();
        let x: Tensor<2> = Tensor::from_data(TensorData::new(sinusoid, [1, freq_dim]), self.device.burn());
        self.time_2.forward(silu(self.time_1.forward(x)))
    }
}

/// The host tensors of block `index`.
fn read_block(shards: &[PathBuf], index: usize) -> Result<StateDict> {
    let scope = format!("blocks.{index}.");
    StateDict::from_safetensors_shards(shards, |name| name.starts_with(&scope))
}

fn load_block(
    shards: &[PathBuf],
    index: usize,
    config: &TransformerConfig,
    device: &Device,
) -> Result<Block> {
    let mut state = read_block(shards, index)?;
    let block = Block::load(&mut state, &format!("blocks.{index}"), config, device)?;
    state.finish()?;
    Ok(block)
}
