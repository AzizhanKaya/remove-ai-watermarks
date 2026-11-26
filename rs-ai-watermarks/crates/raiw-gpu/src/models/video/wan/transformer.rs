//! `WanTransformer3DModel` (Wan 2.1 T2V 1.3B): 1x2x2 patches, 30 blocks of RoPE self-attention,
//! text cross-attention and a GELU feed-forward, each modulated by the timestep (adaLN).
//!
//! The float32 checkpoint is 5.7 GB, more than an 8 GB display GPU holds next to activations,
//! so blocks are either resident or streamed: each forward memory-maps the shards and uploads
//! one block at a time. Token-wise work runs in bounded chunks and attention through
//! [`scaled_dot_product`], so no dispatch grows with the clip.

use std::path::PathBuf;
use std::sync::mpsc::sync_channel;

use burn::tensor::activation::{gelu_approximate, silu};
use burn::tensor::{Device, Tensor, TensorData, s};

use crate::device::ComputeDevice;
use crate::error::{GpuError, Result};
use crate::hub::{Fetch, Hub};
use crate::io::read_finite;
use crate::layers::Linear;
use crate::weights::StateDict;

use super::attention::{LayerNorm, RmsNorm, scaled_dot_product};
use super::prompt::{PromptEmbedding, TEXT_DIM, TEXT_TOKENS};
use super::rope::Rope;
use super::{LATENT_CHANNELS, TRANSFORMER_CONFIG, TRANSFORMER_SHARDS};

pub const DIM: usize = 1536;
pub const HEADS: usize = 12;
pub const HEAD_DIM: usize = DIM / HEADS;
pub const FFN_DIM: usize = 8960;
pub const LAYERS: usize = 30;
const FREQ_DIM: usize = 256;
const EPS: f64 = 1e-6;
/// Spatial patch side (the time patch is 1).
pub const PATCH: usize = 2;
const PATCH_FEATURES: usize = LATENT_CHANNELS * PATCH * PATCH;
/// Tokens per feed-forward chunk, bounding the `[tokens, 8960]` intermediate.
const FFN_TOKENS: usize = 4096;

/// Where transformer block weights live between forwards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlockResidency {
    /// Memory-map the shards and upload one block at a time on every forward (~190 MB peak).
    #[default]
    Streamed,
    /// Keep all 30 blocks on the device (5.5 GB of float32).
    Resident,
}

/// The prompt embedding projected by `condition_embedder.text_embedder`, `[512, 1536]`.
#[derive(Debug, Clone)]
pub struct TextContext {
    hidden: Tensor<2>,
}

#[derive(Debug, Clone)]
struct Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
}

impl Attention {
    fn load(state: &mut StateDict, prefix: &str, device: &Device) -> Result<Self> {
        let linear = |state: &mut StateDict, name: &str| {
            Linear::load(state, &format!("{prefix}.{name}"), DIM, DIM, device)
        };
        Ok(Self {
            to_q: linear(state, "to_q")?,
            to_k: linear(state, "to_k")?,
            to_v: linear(state, "to_v")?,
            to_out: linear(state, "to_out.0")?,
            norm_q: RmsNorm::load(state, &format!("{prefix}.norm_q"), DIM, EPS, device)?,
            norm_k: RmsNorm::load(state, &format!("{prefix}.norm_k"), DIM, EPS, device)?,
        })
    }

    fn heads(x: Tensor<2>) -> Tensor<3> {
        let tokens = x.dims()[0];
        x.reshape([tokens, HEADS, HEAD_DIM])
    }

    /// `[tokens, heads, head_dim]` -> `[1, heads, tokens, head_dim]`.
    fn batched(x: Tensor<3>) -> Tensor<4> {
        x.swap_dims(0, 1).unsqueeze_dim(0)
    }

    fn project_out(&self, out: Tensor<4>) -> Tensor<2> {
        let tokens = out.dims()[2];
        self.to_out
            .forward(out.squeeze_dim::<3>(0).swap_dims(0, 1).reshape([tokens, DIM]))
    }

    /// Self-attention with RoPE on queries and keys.
    fn self_attention(&self, x: Tensor<2>, rope: &Rope) -> Tensor<2> {
        let q = rope.apply(Self::heads(self.norm_q.forward(self.to_q.forward(x.clone()))));
        let k = rope.apply(Self::heads(self.norm_k.forward(self.to_k.forward(x.clone()))));
        let v = Self::heads(self.to_v.forward(x));
        self.project_out(scaled_dot_product(
            Self::batched(q),
            Self::batched(k),
            Self::batched(v),
        ))
    }

    /// Cross-attention from the tokens to the projected text context.
    fn cross_attention(&self, x: Tensor<2>, context: &Tensor<2>) -> Tensor<2> {
        let q = Self::heads(self.norm_q.forward(self.to_q.forward(x)));
        let k = Self::heads(self.norm_k.forward(self.to_k.forward(context.clone())));
        let v = Self::heads(self.to_v.forward(context.clone()));
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
    fn load(state: &mut StateDict, prefix: &str, device: &Device) -> Result<Self> {
        let table: Tensor<3> = state.tensor(&format!("{prefix}.scale_shift_table"), [1, 6, DIM], device)?;
        Ok(Self {
            table: table.reshape([6, DIM]),
            norm: LayerNorm::plain(DIM, EPS, device),
            attn1: Attention::load(state, &format!("{prefix}.attn1"), device)?,
            norm2: LayerNorm::load(state, &format!("{prefix}.norm2"), DIM, EPS, device)?,
            attn2: Attention::load(state, &format!("{prefix}.attn2"), device)?,
            ffn_in: Linear::load(state, &format!("{prefix}.ffn.net.0.proj"), DIM, FFN_DIM, device)?,
            ffn_out: Linear::load(state, &format!("{prefix}.ffn.net.2"), FFN_DIM, DIM, device)?,
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
}

/// `WanTransformer3DModel` from `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (`transformer/`).
#[derive(Debug, Clone)]
pub struct WanTransformer {
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
    device: ComputeDevice,
}

impl WanTransformer {
    /// Resolve the pinned shards through the hub cache (downloading if allowed).
    pub fn load(device: &ComputeDevice, residency: BlockResidency) -> Result<Self> {
        Self::load_from_hub(&Hub::from_env(), Fetch::Download, device, residency)
    }

    pub fn load_from_hub(
        hub: &Hub,
        fetch: Fetch,
        device: &ComputeDevice,
        residency: BlockResidency,
    ) -> Result<Self> {
        let config = hub.resolve(&TRANSFORMER_CONFIG, fetch)?;
        let shards = TRANSFORMER_SHARDS
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
        Self::from_shards(shards, device, residency)
    }

    /// Load local `transformer/*.safetensors` shards without digest checks.
    pub fn from_shards(
        shards: Vec<PathBuf>,
        device: &ComputeDevice,
        residency: BlockResidency,
    ) -> Result<Self> {
        let burn = device.burn();
        let mut state = StateDict::from_safetensors_shards(&shards, |name| !name.starts_with("blocks."))?;
        let patch = state.take("patch_embedding.weight", &[DIM, LATENT_CHANNELS, 1, PATCH, PATCH])?;
        let patch_bias = state.tensor("patch_embedding.bias", [DIM], burn)?;
        let patch = Linear::from_host(patch, PATCH_FEATURES, DIM, Some(patch_bias), burn);
        let linear = |state: &mut StateDict, name: &str, input: usize, output: usize| {
            Linear::load(state, &format!("condition_embedder.{name}"), input, output, burn)
        };
        let time_1 = linear(&mut state, "time_embedder.linear_1", FREQ_DIM, DIM)?;
        let time_2 = linear(&mut state, "time_embedder.linear_2", DIM, DIM)?;
        let time_proj = linear(&mut state, "time_proj", DIM, 6 * DIM)?;
        let text_1 = linear(&mut state, "text_embedder.linear_1", TEXT_DIM, DIM)?;
        let text_2 = linear(&mut state, "text_embedder.linear_2", DIM, DIM)?;
        let out_table: Tensor<3> = state.tensor("scale_shift_table", [1, 2, DIM], burn)?;
        let proj_out = Linear::load(&mut state, "proj_out", DIM, PATCH_FEATURES, burn)?;
        state.finish()?;
        let blocks = match residency {
            BlockResidency::Streamed => Blocks::Streamed(shards),
            BlockResidency::Resident => Blocks::Resident(
                (0..LAYERS)
                    .map(|i| load_block(&shards, i, burn))
                    .collect::<Result<Vec<_>>>()?,
            ),
        };
        Ok(Self {
            patch,
            time_1,
            time_2,
            time_proj,
            text_1,
            text_2,
            out_table: out_table.reshape([2, DIM]),
            norm_out: LayerNorm::plain(DIM, EPS, burn),
            proj_out,
            blocks,
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &ComputeDevice {
        &self.device
    }

    /// `condition_embedder.text_embedder(prompt_embeds)`, reused by every denoising step.
    pub fn encode_text(&self, prompt: &PromptEmbedding) -> TextContext {
        let embeds: Tensor<2> = Tensor::from_data(
            TensorData::new(prompt.values().to_vec(), [TEXT_TOKENS, TEXT_DIM]),
            self.device.burn(),
        );
        TextContext {
            hidden: self.text_2.forward(gelu_approximate(self.text_1.forward(embeds))),
        }
    }

    /// Flow velocity for latents `[16, T, H, W]` (`H`, `W` even) at an integer `timestep`.
    pub fn forward(
        &self,
        latents: &[f32],
        (frames, height, width): (usize, usize, usize),
        timestep: f32,
        context: &TextContext,
    ) -> Result<Vec<f32>> {
        if frames == 0 || height == 0 || width == 0 || height % PATCH != 0 || width % PATCH != 0 {
            return Err(GpuError::Input(format!(
                "Wan transformer latents need even nonzero sides, got {frames}x{height}x{width}"
            )));
        }
        if latents.len() != LATENT_CHANNELS * frames * height * width {
            return Err(GpuError::Input(format!(
                "Wan transformer latents hold {} values, expected {LATENT_CHANNELS}x{frames}x{height}x{width}",
                latents.len()
            )));
        }
        let burn = self.device.burn();
        let (rows, cols) = (height / PATCH, width / PATCH);
        let tokens = frames * rows * cols;
        let x: Tensor<6> = Tensor::from_data(
            TensorData::new(
                latents.to_vec(),
                [LATENT_CHANNELS, frames, rows, PATCH, cols, PATCH],
            ),
            burn,
        );
        let patches = x.permute([1, 2, 4, 0, 3, 5]).reshape([tokens, PATCH_FEATURES]);
        let mut h = self.patch.forward(patches);
        let rope = Rope::new(HEAD_DIM, (frames, rows, cols), burn);
        let temb = self.time_embedding(timestep);
        let modulation = self.time_proj.forward(silu(temb.clone())).reshape([6, DIM]);
        match &self.blocks {
            Blocks::Resident(blocks) => {
                for block in blocks {
                    h = block.forward(h, &context.hidden, &modulation, &rope);
                }
            }
            Blocks::Streamed(shards) => {
                h = std::thread::scope(|scope| -> Result<Tensor<2>> {
                    // A reader thread maps the next block while the device runs this one
                    let (sender, receiver) = sync_channel(1);
                    scope.spawn(move || {
                        for i in 0..LAYERS {
                            let state = read_block(shards, i);
                            let failed = state.is_err();
                            if sender.send(state).is_err() || failed {
                                break;
                            }
                        }
                    });
                    for i in 0..LAYERS {
                        let mut state = receiver.recv().map_err(|_| {
                            GpuError::Config("Wan transformer block reader stopped".to_owned())
                        })??;
                        let block = Block::load(&mut state, &format!("blocks.{i}"), burn)?;
                        state.finish()?;
                        h = block.forward(h, &context.hidden, &modulation, &rope);
                        // Finish this block before the next upload so only one is resident
                        self.device.sync()?;
                    }
                    Ok(h)
                })?;
            }
        }
        let table = self.out_table.clone() + temb;
        let (shift, scale) = (table.clone().slice(s![0..1]), table.slice(s![1..2]));
        let h = self.norm_out.forward(h) * scale.add_scalar(1.0) + shift;
        let out = self
            .proj_out
            .forward(h)
            .reshape([frames, rows, cols, PATCH, PATCH, LATENT_CHANNELS])
            .permute([5, 0, 1, 3, 2, 4])
            .reshape([LATENT_CHANNELS, frames, height, width]);
        read_finite(out)
    }

    /// `time_embedder(timesteps_proj(t))`: flipped sinusoids, Linear, SiLU, Linear -> `[1, dim]`.
    fn time_embedding(&self, timestep: f32) -> Tensor<2> {
        let half = FREQ_DIM / 2;
        let log_period = -(10_000f64.ln()) as f32;
        let args: Vec<f32> = (0..half)
            .map(|i| timestep * (log_period * i as f32 / half as f32).exp())
            .collect();
        let sinusoid: Vec<f32> = args
            .iter()
            .map(|a| a.cos())
            .chain(args.iter().map(|a| a.sin()))
            .collect();
        let x: Tensor<2> = Tensor::from_data(TensorData::new(sinusoid, [1, FREQ_DIM]), self.device.burn());
        self.time_2.forward(silu(self.time_1.forward(x)))
    }
}

/// The host tensors of block `index`.
fn read_block(shards: &[PathBuf], index: usize) -> Result<StateDict> {
    let scope = format!("blocks.{index}.");
    StateDict::from_safetensors_shards(shards, |name| name.starts_with(&scope))
}

fn load_block(shards: &[PathBuf], index: usize, device: &Device) -> Result<Block> {
    let mut state = read_block(shards, index)?;
    let block = Block::load(&mut state, &format!("blocks.{index}"), device)?;
    state.finish()?;
    Ok(block)
}
