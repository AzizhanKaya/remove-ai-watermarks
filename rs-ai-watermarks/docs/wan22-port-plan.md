# Wan 2.2 TI2V-5B port plan

Last updated: 2026-11-27. Wan 2.1 T2V-1.3B regenerations at strength 0.38 pass synthid.com
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
- [x] Goldens from diffusers float32 on CPU (`goldens/gpu_wan22/`):
  - `gpu_wan22_vae.py`: two untiled clips, and two tiled ones (diffusers' 256/192 and 512/480
    `enable_tiling`).
  - `gpu_wan22_transformer.py`: the block-streamed oracle of `gpu_wan_transformer.py`.
  - `gpu_wan22_pipeline.py`: `_run_wan_window` with `flow_shift` 5. diffusers'
    `WanVideoToVideoPipeline` derives its spatial factor as 8, so the script sets it to 16.
- [x] VAE (`vae3d.rs`, `VaeConfig` from `vae/config.json`):
  - Patchify (channel `c·4 + pw·2 + ph`).
  - `AvgDown3D` / `DupUp3D` shortcuts with `first_chunk`, residual down/up blocks,
    `upsample_out_dim`, `decoder_base_dim`, 48-channel statistics, output clamp.
  - Matches diffusers: latents 3.5e-6, decoded 2.4e-5 (CPU and GPU).
  - `VaeTiles` mirrors `tiled_encode`/`tiled_decode` for both models (patchified tiles, latent
    blends for encode; latent tiles, patchified blends for decode). It matches diffusers'
    tiled output to 4.5e-6 / 1.5e-5.
  - On the GPU no burn matmul runs (the autotune outlasted the watchdog on full frames):
    - stride-2 convs run as subsampled stride-1 kernel convs;
    - the 1x1 projections run on the conv kernel;
    - mid-block attention runs on the GEMM kernel.
  - Decoder stages after the last temporal upsample run frame by frame.
  - Memory at 480×864: untiled, the 30 causal caches alone take ~3 GB and the card fills
    (ring timeouts).
    - diffusers' encode tiles are in patchified pixels, so its 256 px default covers 512 px
      of the frame. Next to the transformer's resident remainder (~0.45 GiB) that still
      overflowed in the CLI.
    - Encode tiles are therefore separate (`with_encode_tiles`): Wan 2.2 encodes with 128/96
      (`VaeTiles::WAN22_ENCODE`) and decodes with diffusers' 256/192.
    - Each pass equals diffusers run with that `enable_tiling`.
    - 5 frames: encode 6.3 s at 3.7 GiB peak, decode 25.2 s at 4.0 GiB peak (desktop
      included).
- [x] Transformer: `WAN22_TRANSFORMER` pins the 5 shards. Streamed float32 matches diffusers to
  2.9-7.2e-6. Resident `Int8` is off by 2.3% of the peak `|out|` (Wan 2.1: 3.1%), in line with
  diffusers' own bf16 drift per block.
- [x] Pipeline: `WanModel` (`WAN21_T2V_1_3B`, `WAN22_TI2V_5B`) carries the files, the flow
  shift and the residency. Latent statistics, spatial factor and multiple (16 / 32) come from
  the configs.
- [x] Memory staging, `BlockResidency::Swapped(Precision)`:
  - The int8 blocks live in host memory (~5 GB RAM) and upload for each denoising pass.
  - Int8 blocks are quantized and built on the host. Building them on the GPU and parking
    them left the pool growing ~180 MB per block (7.9 GiB at the end of the load). f16 blocks
    still build on the GPU, since the packing runs a kernel, with a cleanup after each.
  - Parked blocks project the prompt's cross-attention keys/values by uploading only that
    projection (`FusedBlock::cross_kv_on`).
  - The VAE reloads from its checkpoint (no re-hash) for each encode and decode.
  - `memory_cleanup()` frees the pool between stages.
  - On Wan 2.1, `Swapped(F16)` reproduces `Fused(F16)` exactly (same levels) and adds ~3 s per
    window.
- [x] CLI: `--profile wan22-synthid` (alias of `wan22-synthid-experimental`), Rust-only. It runs
  on `NeuralBackends::wan22` (`BurnWan::wan22`, swapped int8 on a GPU), with strength 0.38 (not
  oracle-calibrated yet).

- [x] Pipeline parity on the GPU (`wan22_pipeline_matches_run_wan_window`): Streamed f32 is
  within 1 level of the diffusers window; Swapped Int8 is within max 10, mean 0.66 levels.
- [x] Kernel launchers submit after ≥ 100 GFLOP: amdgpu times whole submissions, and cubecl
  packs up to 32 dispatches into one.
- [x] Measured at 480p:
  - Transformer, int8, one 81-frame window (8190 tokens): 20.8 s per forward, ~2.4 min for
    7 steps (Wan 2.1 f16: ~12.5 min per window). Load: 78 s.
  - Peak VRAM in the transformer stage: 6.1 GiB.

- [x] Ring-timeout safety for experiments: poll `mem_info_vram_used` and `pkill -x` the
  process before the card fills (`docs/findings.md`).

## Where it stopped (2026-11-27)

- The user clip (`hf_20261124_221145_…mp4`, 480×854, 313 frames) was running with
  `video all --invisible --profile wan22-synthid` at this checkpoint. Its output would be
  `~/Downloads/…_clean_wan22.mp4`; if that file is missing, the run did not finish.
  - Two earlier attempts were killed by the VRAM switch: block loading on the GPU, then
    256 px encode tiles. Both are fixed above.
  - The third ran cleanly through window 1: load 94 s with the GPU idle, denoise peak
    6.6 GiB, then decode.
  - Command, from `rs-ai-watermarks/`:
    `HF_HUB_OFFLINE=1 ./target/release/rs-ai-watermarks video all --invisible --profile wan22-synthid <src> -o <src>_clean_wan22.mp4`.
    Build first with `cargo build --release -p rs-ai-watermarks --no-default-features --features metadata,video,gpu`.
- GPU tests not rerun after the kernel submit split and the swapped host loading (the GPU
  was busy with the clip): `cargo test --release -p raiw-gpu --features wgpu --test wan --
  --test-threads=1`, in particular `pipeline_matches_run_wan_window` and both transformer
  parity tests. VAE parity (all 6 cases) and the unit tests pass.

## To do, in order

1. **User clip:** rerun the command above (~1 h), then compare it with `_clean_wan21` and send
   it to the user for synthid.com.
2. **Speed:**
   - Loading hashes and reads 19 GB of float32 shards on every run. Write an int8 cache file
     (~5 GB) once.
   - Decode is now the slowest stage (~20 s per latent frame, tiled). Remove the `DupUp`
     repeat/permute copies with a gather kernel, fuse the shortcut into the upsample conv's
     residual, and give the conv kernel a real stride instead of subsampling.
   - TI2V-5B's native size is 1280×704, which needs ~3× the compute per window.
3. **SynthID calibration:** the user checks a strength sweep on synthid.com (starting at 0.38,
   then lower for fidelity). Record the sha256 and verdicts in the trial doc.
4. **Optional:**
   - Fold the VAE's RMS norm + SiLU into the conv input load.
   - Per-group int8 (g = 128) if the drift shows.
   - f16 VAE weights if memory is tight.
   - Delete the 22.7 GB float32 UMT5-XXL download (needed only to re-encode the prompt), with
     the user's approval.
