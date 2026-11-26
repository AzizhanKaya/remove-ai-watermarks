# Wan 2.2 TI2V-5B port plan

Last updated: 2026-11-26. Wan 2.1 T2V-1.3B regenerations at strength 0.38 pass synthid.com
but look soft and unnatural ([video-synthid-trial-2026-11.md](video-synthid-trial-2026-11.md)).
Wan 2.2 TI2V-5B is the strongest Wan that fits the 8 GB RX 5500 XT, and only with int8 weights.
Python has no Wan 2.2 path, so the oracle is diffusers 0.43 itself (oracle venv), and the
end-to-end profile will be Rust-only and experimental.

Pinned: `Wan-AI/Wan2.2-TI2V-5B-Diffusers` @ `b8fff7315c768468a5333511427288870b2e9635`, in the
Hugging Face cache (transformer: 19 GB float32 in 5 shards; VAE: 2.7 GB float32; text encoder:
config only).

## Prompt (same as Python's Wan 2.1 profiles)

- Positive: `WAN21_DEFAULT_PROMPT` = "High fidelity natural video, sharp details, accurate
  motion, clean cinematic lighting." (`src/remove_ai_watermarks/video_wan.py`).
- Negative: `WAN21_DEFAULT_NEGATIVE_PROMPT` = "blurry, low quality, compression artifacts,
  flicker, warped edges, text, logo, watermark". It is never used: guidance 1.0 disables CFG,
  and a prompt-embedding cache skips it entirely.
- UMT5-XXL in bfloat16 encodes it once to `[1, 226, 4096]`. Wan 2.2's text-encoder config
  equals Wan 2.1's except the stored dtype (bfloat16 vs float32), so the 2.1 prompt cache
  should carry over. Check one tensor against a 2.2 shard before relying on it.

## What changes from Wan 2.1

| | Wan 2.1 T2V-1.3B | Wan 2.2 TI2V-5B |
| --- | --- | --- |
| Transformer | dim 1536 (12×128), FFN 8960, 30 layers | dim 3072 (24×128), FFN 14336, 30 layers |
| Latent channels | 16 | 48 |
| VAE | 8× spatial, plain | 16× spatial: 2×2 patchify + residual down/up blocks (`is_residual`), decoder base 256 |
| Spatial multiple | 16 | 32 |
| Tokens per 480p latent frame | 1560 | 390 |
| Scheduler | UniPC, flow shift 3 | UniPC, flow shift 5 (scheduler default) |
| Resident weights | f16, 2.7 GB | int8, ~5.0 GB |

`expand_timesteps` (a per-token timestep for image conditioning) reduces to the scalar
timestep in video-to-video, where every token shares one t.

## Done

- [x] Download the pinned revision.
- [x] `TransformerConfig` (`wan/config.rs`): the transformer, attention and `FusedBlock` read
  their sizes from `config.json`; Wan 2.1 is `TransformerConfig::WAN21_T2V_1_3B`.
- [x] W8A16 GEMM: `hgemm` is generic over the weight type, using per-output-channel int8 with
  scale `max|w| / 127`. Added `DenseWeight::int8` and `Precision::Int8`.
  - Accuracy: the int8 linear is off by 4-6e-4; Int8 fused blocks drift 1.34e-2 from float32
    (diffusers bf16 drifts 1.48e-2).
  - Speed: ~4.8 TFLOP/s.
- [x] VAE golden generator `tools/goldens/gpu_wan22_vae.py` (fp32, CPU, no tiling).

## To do, in order

1. **VAE goldens:** run `gpu_wan22_vae.py` (~3 GB RSS) to produce `goldens/gpu_wan22/vae.json`.
2. **VAE port:** generalize `vae3d.rs` behind a `VaeConfig` read from `vae/config.json`.
   - Patchify and unpatchify, with channel order `c·4 + pw·2 + ph`.
   - `AvgDown3D` and `DupUp3D` shortcuts, including the `first_chunk` handling.
   - Residual down blocks (`down_blocks.{i}.resnets` / `.downsampler` / `avg_shortcut`) and up
     blocks (`up_blocks.{i}.resnets` / `.upsampler`, `upsample_out_dim`).
   - `decoder_base_dim` 256 and the 48-channel latents mean/std.
   - Clamp the output to [-1, 1].
   - Conv kernels on every causal conv. Measure memory without tiling, then decide on tiling.
3. **Transformer goldens:** a diffusers block-streamed script for 2.2 (like
   `gpu_wan_transformer.py`) on a small grid. Pin the 5 shards with sha256 in a
   `WAN22_TRANSFORMER` `TransformerFiles`. Test Int8 and F16 blocks; full F16 needs ~10 GB, so
   use truncated layers.
4. **Pipeline:** generalize `pipeline.rs` with the latent channel count, spatial factor and
   multiple, latents mean/std, and flow shift (decide whether 5 holds at strength ≤ 0.4). Reuse
   the prompt cache (see above).
5. **Memory staging:** the int8 transformer (5 GB) plus the fp32 VAE (2.7 GB) plus activations
   do not fit in 8 GB.
   - Stage per clip: encode every window, drop the VAE, load the transformer, denoise every
     window, drop it, then decode.
   - Write an int8 cache file (~5 GB) once, so a run does not re-read and re-quantize 19 GB of
     float32 shards.
   - Load shard by shard to keep RSS ≤ ~5 GB, and measure peak VRAM and RSS.
6. **CLI profile:** a Rust-only experimental `wan22-*` profile in raiw-video and the CLI. Mark
   it as having no Python oracle.
7. **Timing on the user clip:**
   - Estimate at 480p for an 81-frame window (8190 tokens): ~107 TFLOP per forward, which is
     ~25 s at 4.5 TFLOP/s, or ~3 min for 7 steps, plus the VAE.
   - TI2V-5B's native size is 1280×704, which needs ~3× the compute per window.
8. **SynthID calibration:** the user checks a strength sweep on synthid.com (starting at 0.38,
   then lower for fidelity). Record the sha256 and verdicts in the trial doc.
9. **Optional:**
   - Fold the VAE's RMS norm + SiLU into the conv input load.
   - Per-group int8 (g = 128) if the drift shows.
   - f16 VAE weights if memory is tight.
   - Delete the 22.7 GB float32 UMT5-XXL download (needed only to re-encode the prompt), with
     the user's approval.
