# Port status and resume points

Last updated: 2026-11-26. The Python package in `../src/remove_ai_watermarks` is the
behavioral oracle; every number below comes from differential golden tests or
side-by-side CLI runs against it. Measured findings, speed tables and hazards live
in [findings.md](findings.md); the user-file video trial is in
[video-synthid-trial-2026-11.md](video-synthid-trial-2026-11.md); the Wan 2.2 TI2V-5B port
plan and TODO list are in [wan22-port-plan.md](wan22-port-plan.md).

## Priorities agreed with the user

1. Core, video and GPU first (video is done; GPU speed is in progress).
2. Text, source classifier, image pixel decoders, image identify/visible and
   image regeneration come later.
3. Keep resource use low: at most ~3 concurrent workers, `jobs = 4`
   (`.cargo/config.toml`), tests with `--test-threads=2` (model/GPU tests one at a
   time), torch/OMP threads 4, short GPU dispatches, one model resident, no
   ChunkySeal or full-Wan loads in routine tests, ≤ 2.5 GB RSS per process. The
   OOM killer (earlyoom) already fired once and closed a desktop app.

## Crate status

| Crate | State | Lines / tests | Resume point |
| --- | --- | --- | --- |
| `raiw-core` | Done | 1.2k / 12 | — |
| `raiw-container` | Done, golden-exact on 104 fixtures | 5.3k / 26 | — |
| `raiw-c2pa` | Done; c2pa-rs `=0.91.0` (same line as c2pa-python 0.38), derived `c2pa_info` byte-identical on 33 C2PA fixtures | 5.2k / 28 | — |
| `raiw-metadata` | Done for video/audio containers (byte-identical strips); images match except WebP re-save and two `info:exif` record fields | 12.0k / 18 | Exact libwebp 1.6 encoder and Pillow `Exif.tobytes` writer; truncated GIF/BMP/TIFF message wording |
| `raiw-cv` | Done; OpenCV 4.10-exact on almost every op | 10.4k / 21 | DIS optical flow (only image regeneration alignment needs it); LINE_AA thick circle; speed (see findings) |
| `raiw-video` | Done: phases 1-5 | 19.0k / 51 | `video all --invisible` fused path not golden-compared; Wan never run on real weights |
| `raiw-gpu` | Done: 7 audio models, SD VAE (+tiling), VideoSeal/PixelSeal/ChunkySeal, Wan 2.1 VAE/DiT/UniPC/pipeline; CubeCL kernels (`kernels/`) drive the Wan DiT (`Fused(F16)`) and VAE convolutions on GPUs | 14.5k / 75 | Speed work in progress (fp16 SD VAE, batching, conv path, persistent kernel cache); Wan kernel follow-ups below |
| `raiw-audio` | DSP/IO/quality/policy done; burn models wired behind `gpu` | 10.9k / 84 | Speed (models 2-6x slower than Python); SilentCipher on non-44.1 kHz (soxr resampler); wire raiw-metadata into `MetadataSeam`; WavMark/Perth/SNAC/EnCodec not run on real models end to end |
| `raiw-text` | ~90% | 13.8k / 77 | clippy/fmt, CLI wiring, `score_synthid_text` (needs a burn LM), burn generation backend; compressed `encode_innamark` bytes differ from CPython zlib |
| `raiw-pixel` | DWT-DCT and InvisMark done (bit-exact) | 3.0k / 38 | Declare `mod pillow; mod jpeg; pub mod trustmark;` in `lib.rs`, add goldens `pixel_trustmark.py`/`pixel_jpeg.py`; `pixel_evidence` not started |
| `raiw-visible` | 19 engines, registry, eraser, image_io written; compiles | 6.5k / 13 | Write `tests/goldens.rs` against `goldens/visible/image-*.json` and fix mismatches; heif feature untested; Gemini speed risk |
| `raiw-classify` | ~10% (consts, result, pickle-free npz reader, all goldens) | 0.8k / 2 | Loaders and heads v3-v10 against `goldens/classify/bundles.json`, v4/v3 features, burn ViT (ViT-S/16@384, shared to block 9 then two 2-block branches; ONNX dump kept at `~/.cache/rs-ai-watermarks/classify/onnx/`), re-add `raiw-gpu`/`ureq`/`turbojpeg` deps |
| `raiw-identify` | Not started | — | Port `identify.py` once visible and pixel are verified |
| `raiw-regen` | Not started | — | Image diffusion profiles (Qwen-Image, SDXL, Z-Image, Chroma) |
| `raiw-mcp` | Done (17 tests live in the facade crate) | 1.8k | — |
| `rs-ai-watermarks` (CLI) | `metadata`, every `video` command, `mcp` wired | 2.3k / 21 | Wire `identify`, `visible`, `erase`, `invisible`, `all`, `batch`, `text`, `audio`, `classify-source` as their crates finish; add the image options hidden by the first Click dump (`all`: `--backend`, `--sensitivity`, `--pipeline`; `batch`: `--preset`, `--backend`, `--sensitivity`, `--vendor`, `--pipeline`; `text all`: `--rewriter-backend`) |

## Wan 2.1 CubeCL kernels (2026-11-26, tested on the RX 5500 XT)

`crates/raiw-gpu/src/kernels/` holds hand-written CubeCL kernels behind a burn backend
extension (`#[backend_extension(Cube, Fusion)]`, so they queue lazily in the fusion stream):

| Kernel | What | RX 5500 XT |
| --- | --- | --- |
| `attention` | FlashAttention-2 forward, row-major `[tokens, heads·128]`, f16 or f32 storage, f32 Q·Kᵀ/softmax, P·V in packed f16 over 16 keys | 4.6 TFLOP/s f16 (old chunked burn path 0.24) |
| `hgemm` | packed-f16 `x·W + b` for `[in, out]` f16 weights, GELU / gated-residual epilogues | 4.6-4.9 TFLOP/s (burn f32 matmul 1.8) |
| `gemm` | register-tiled `x·Wᵀ + b` for PyTorch `[out, in]` weights (f32 parity path) | 2.8 f32, 4.0 f16 |
| `norm` | LayerNorm + adaLN modulation; RMSNorm + RoPE reading a QKV column window | memory-bound |
| `conv` | causal 3-D conv as an implicit GEMM (no im2col), bias + residual + 2x upsampling fused | 2.8-3.1 TFLOP/s f32 (im2col path 1.0-1.8) |

No kernel autotunes and every launcher splits its dispatches (≤ 0.1 s each). Tile sweeps:
`cargo test --release -p raiw-gpu --test dit_kernels -- --ignored --nocapture --test-threads=1`.

Wiring: `BlockResidency::Fused(Precision)` keeps all 30 blocks resident (2.7 GB at f16) and
runs them on the kernels; `BlockResidency::for_device` picks `Fused(F16)` on a GPU and
raiw-video uses it. The VAE takes the conv kernel for every stride-1 causal conv and the
up-resample conv whenever the device is a GPU (float32, as in Python).

Results (480p; `tests/wan.rs` `wan_480p_timing`, `fused_blocks_match_reference_blocks`):

- Transformer forward, 21 latent frames (one 81-frame window, 32 760 tokens): 62 s warm
  (previous streamed f32 path projected ~300 s); 7 800 tokens: 6.9 s. Python on this machine
  (diffusers, torch CPU fp32, 12 threads): 2.53 s per block at 7 800 tokens, ~76 s per forward.
- VAE, 5 frames: encode 11.8 → 7.4 s, decode 20.8 → 12.6 s (decode is now conv-bound; Wan's
  256 px tiling, kept for parity, roughly doubles its work at 480x832).
- One 81-frame window (7 steps): ~49 min → ~12.5 min projected.
- Accuracy: fused F32 blocks match the burn-op blocks to 2.7e-6 (relative); fused F16 drifts
  1.8e-3 from them, while diffusers' own bfloat16 path drifts 1.5e-2 from float32 on the same
  inputs. VAE and full-pipeline goldens unchanged (latents 4e-6, pipeline ≤ 1 u8 level).

Follow-ups: the f32 conv kernel tops out near 3 TFLOP/s (ACO splits its shared vec4 reads into
two `ds_read_b64`); a packed-f16 Q·Kᵀ variant was slower (pair reads plus a 38 KB footprint
halve occupancy) and was dropped; the VAE's RMS norm + SiLU could fold into the conv input load.

## GPU speed work: checkpoint of 2026-11-26 (compiles, NOT yet tested on the GPU)

Root causes found for slow GPU commands: nothing persisted across processes
(burn `persistence` off, WGSL path) and burn-flex built without `rayon`/`simd`
(single-threaded CPU decoders). In `video message detect` the first 16-frame chunk
spent ~13 s in autotune and kernel compile; later chunks took ~1.2 s each.

Changed in `crates/raiw-gpu` (untested on the GPU):

- burn with `rayon` + `simd`; `wgpu` adds `burn/vulkan` (native SPIR-V); a shared
  `gpu-runtime` feature (`fusion`, `autotune`, `persistence`, `cubecl`).
- `device.rs`: one runtime client per process; autotune results and SPIR-V
  kernels cached in `~/.cache/remove-ai-watermarks/cubecl`.
- `layers/conv.rs`: strided Conv1d via im2col/matmul, depthwise Conv1d as shifted
  multiply-adds, ConvTranspose1d as matmul + overlap-add (GPU only).
- `layers/conv2d.rs`: stride-1 depthwise lowering for ConvNeXt 7x7 (revert if
  `depthwise_conv2d_timings` shows burn's autotuned kernel is faster).
- `layers/lstm.rs`: device matmul for the input projection, host recurrence;
  `Lstm::forward` and `SeanetEncoder/Decoder::forward` now return `Result`.
- `videoseal.rs`: one `[N·H·W, C]` matmul for its Linear layer.

Resume, in order:

1. `cargo test --release -p raiw-gpu --test kernels -- --test-threads=1`, then the
   ignored `depthwise_conv2d_timings` and `vae_matmul_throughput` (f32 vs f16).
2. The raiw-gpu parity suites one at a time (kernels, audioseal, codecs, videoseal,
   vae, wavmark, perth, silentcipher), then raiw-audio and raiw-video with `gpu`.
3. After-timings: `video message detect` cold (`rm -rf
   ~/.cache/remove-ai-watermarks/cubecl`) and warm (before: 18.3-19.5 s vs Python
   9.3 s); `video invisible --profile synthid --long-side 256 --fps 6` (before
   33.7 s vs 32.0 s); AudioSeal identify on `provenance/xai-grok-imagine-video.mp4`
   (before 18.8 s CPU vs 7.2 s); `audio invisible --method dac`.
4. VAE: opt-in fp16 `Precision` on `SdVae` (report the parity delta), multi-frame
   batches under ~3 GB VRAM, faster decoder Conv2d/GroupNorm; then Wan fp16-resident
   blocks and a CubeCL flash-attention kernel (target ~40 → 8-12 min per 81-frame
   480p window on the RX 5500 XT).
5. Reference clip for VAE timing: a 2.58 s, 62-frame stream copy of the user clip
   (recreate it; the scratch copy lived in tmpfs). Before ≈ 2.1 s per frame at
   `--native-guide` 480×854.

## How to work on it

```bash
cd rs-ai-watermarks
cargo test -p <crate> -- --test-threads=2
cargo build --release -p rs-ai-watermarks --no-default-features --features metadata,video,gpu
.venv/bin/python rs-ai-watermarks/tools/parity.py --release --filter video-   # from the repo root
```

- Goldens: `tools/goldens/<suite>.py` writes `goldens/<suite>/`. Run with
  `/opt/remove-ai-watermarks/.venv/bin/python` (cv2 4.10 oracle) or the oracle
  environment `~/.cache/rs-ai-watermarks/oracle-venv/bin/python` (pywt, scipy,
  audioseal, wavmark, perth, trustmark, onnxruntime, diffusers). Never `uv sync`
  or install into the development `.venv`.
- A golden script named `c2pa.py` shadows the `c2pa` package: every generator in
  `tools/goldens/` must drop its own directory from `sys.path` before importing
  the package.
- The facade crate gates subsystems behind features (`metadata`, `text`, `image`,
  `audio`, `video`, `classify`, `gpu`), so a crate under edit does not break the
  others' builds.
- Toolchain 1.96 (needed by c2pa-rs 0.91), edition 2024, `unsafe` denied.
