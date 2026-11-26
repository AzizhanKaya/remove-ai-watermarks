# Findings

Measured results of the Rust port against the Python reference (2026-11-25/26),
on this machine: 12 cores, 15 GB RAM, AMD Radeon RX 5500 XT 8 GB (RADV/Vulkan),
system ffmpeg n9.1. Python oracle timings use torch on CPU (the installed torch is
a CUDA build and the GPU is AMD).

## CLI parity (`tools/parity.py`, Rust release vs Python CLI)

| Case | Result | Speed vs Python |
| --- | --- | --- |
| `metadata --check` / `--remove` (images) | 65/66 identical incl. output sha256 | ~2x (debug build) |
| `video metadata`, `video identify --no-visible` | 30/30 identical | 13.5x |
| `video identify` (visible scan) | identical | 2.7-3.9x |
| `video visible` | identical incl. output sha256 | 1.3x |
| `video erase` | identical incl. output sha256 | 1.0x |
| `video all` | identical incl. output sha256 | 1.5x |

The one `metadata` difference is the WebP re-save: Pillow uses system libwebp
1.6.0 and no safe Rust binding exists, so the `image` crate's lossless encoder is
used (valid file, different bytes).

GPU commands, one-clip smoke tests (results compared field by field):

| Command | Parity | Rust / Python |
| --- | --- | --- |
| `video invisible --profile synthid --long-side 256 --fps 6` | JSON identical, PSNR metric differs by 2e-5 dB | 33.7 s / 32.0 s |
| `video message detect` | presence probability differs by 1.7e-8 | 19-21 s / 13.4 s |
| `video audio` | same decision (exit 2, no compatible signal) | 8.8 s / 8.9 s |

## Library parity highlights

- C2PA: derived info byte-identical on 33 fixtures; c2pa-rs must stay on `=0.91.0`.
- Container strips, ISOBMFF/EBML/RIFF/FLV/AV1/TIFF walks: exact on 104 fixtures.
- TC260 SM2 label signatures (RustCrypto `sm2`) verify the real MiniMax labels; an
  altered `ProduceID` fails, as in Python.
- OpenCV kernels: bit-exact for resize (vs non-IPP path), cvtColor 8/16-bit,
  GaussianBlur u8/u16, morphology, inpaint Telea/NS, remap, warpAffine, connected
  components, Canny, pyramidal LK; float paths within 1e-5.
- Video decode: system ffmpeg bgr24 pipe with
  `scale=in_color_matrix=bt601:in_range=auto:flags=bicubic` matches
  `cv2.VideoCapture` byte for byte on 40 fixtures (10-bit sources differ because cv2
  bundles FFmpeg 5.1).
- DWT-DCT: pywt's Haar step on x86_64 rounds multiply and add separately (no FMA).
- GPU models vs torch: AudioSeal, DAC, SNAC, EnCodec, WavMark, Perth,
  SilentCipher ≤ 5e-6; SD VAE ≤ 4.4e-5; VideoSeal/PixelSeal ≤ 5e-5; ChunkySeal
  3.9e-3 on fixture frames (same decisive bits); Wan 2.1 VAE/DiT ≤ 4e-5 and the
  full pipeline within 1 u8 level of `video_wan._run_wan_window`.

## Speed status

Faster than Python: metadata, C2PA (510 → 165 ms), containers, text (identify
107 → 9 ms), audio DSP (resample 5x, perceptual 4x), DWT-DCT (15x), InvisMark
(30x), every non-neural video command, SD VAE on GPU (encode 1.1 s vs 7.5 s,
decode 2.2 s vs 15.7 s at 512 px), Wan VAE on GPU (4-6x).

Not yet at parity:

- raiw-cv vs cv2 (after pass 1): remap 2.3-3.6x, warp_affine 2-2.4x, pyramidal LK
  5.4x, linear resize 4-7x, 4K cvt/Sobel slower; match_template now beats cv2.
  End-to-end video commands are still faster than Python.
- burn audio models: AudioSeal identify 18.8 s CPU / 45 s GPU vs 7.2 s; DAC
  65-73 s vs 16.8 s. Strided/depthwise convs still use burn's direct kernels.
- GPU cold start: every process re-loads weights and recompiles wgpu kernels
  (10-20 s); nothing is cached across processes yet.
- Wan 2.1 on the CubeCL kernels (see status.md): one 81-frame 480p window ≈ 12.5 min
  (transformer 62 s per step x 7, VAE ≈ 5 min); the f32 streamed path projected ~49 min.

## Hazards learned the hard way

- A 5 s DAC run on the GPU timed out and reset the desktop GPU: keep dispatches
  short (codec timing runs limited to 1 s; convs chunked to 64-96 MB).
- Running all ten SilentCipher layers as one fused graph corrupted the output at
  71 frames; flushing after each layer fixes it.
- burn's fused `attention` kept RADV compiling for over 7 minutes with the GPU
  idle; the burn-op Wan path uses a plain chunked softmax(QKᵀ)V (≤ 32 MB of scores per
  dispatch) and the GPU path its own flash-attention kernel.
- Autotuning burn's matmul on `[8192x1536]·[1536x8960]` (the Wan feed-forward) ran a
  candidate past the amdgpu ring timeout ("ring gfx_0.0.0 timeout", ring reset, no
  recovery needed). The Wan GPU path therefore never calls burn's matmul on token-sized
  shapes: its GEMM/conv kernels use fixed tiles and split dispatches.
- RADV/ACO on RDNA1: `fma(f32(h), f32(h), acc)` becomes `v_fma_mix_f32` (f16 storage costs no
  conversions), 2-wide f16 FMAs become `v_pk_fma_f16` (splats via `op_sel`), shared vec4 reads
  stay two `ds_read_b64`, and `if`-guarded loads cost a branch each (use `select` with a
  clamped index); `RADV_DEBUG=shaders,shaderstats MESA_SHADER_CACHE_DISABLE=true` dumps ISA.
- CubeCL `Slice::with_vector_size` on a shared buffer produced wrong values when reused as
  f16 pairs; keep one element type per shared allocation.
- Python's Wan prompt cache is `[1, 226, 4096]`: `load_video_wan_runtime` calls
  `WanVideoToVideoPipeline.encode_prompt` without `max_sequence_length`, whose default is
  226 (only `__call__` defaults to 512). The zero padding rows reach cross-attention as real
  bias-only keys, so the row count changes results; the Rust reader assumed 512 and now keeps
  the stored count. The cache can be produced without 12 GB of RAM by loading UMT5-XXL with
  accelerate disk offload (`device_map="auto"`, a 3 GiB CPU budget): same `encode_prompt`, ~4 GB
  RSS, 23 GB download.
- earlyoom killed a model test and the user's desktop app while three workers
  ran model tests with full swap.
- wgpu cannot report VRAM, so `vram_below` returns true and `--profile auto` never
  routes to Wan.

## Bug found in the Python reference (not fixed there)

`_internal/video_message_models.py::_load_chunkyseal_slice` always reads a 16-byte
ZIP data descriptor, but storage 313 of the pinned 3.1 GB ChunkySeal slice sits
past 4 GiB and uses the 24-byte ZIP64 form, so the shipped loader raises
"Corrupted ZIP data descriptor". The Rust reader handles both forms; the
ChunkySeal golden was produced with a corrected loader inside
`tools/goldens/gpu_videoseal.py`.

## Other behavioral notes

- SynthID 2D VAE operating point: the oracle record
  (`data/evaluations/video-synthid-oracle.csv`) has `noise_std 0.15, long_side 512,
  fps 12` → not detected and `0.10` → detected on a Veo 3 clip. `--native-guide`
  has only a PSNR measurement (32.48 dB), no oracle verdict.
- The 2D VAE regeneration leaves blotchy low-frequency texture on flat skin at
  0.15 and only slightly less at 0.12; a post-pass `nlmeans=s=4:p=7:r=13,
  hqdn3d=2:2:6:6` visibly cleans it (unverified against SynthID).
- Wan 2.1 synthid (`data/evaluations/video-wan-oracle.csv`): 6/6 not detected on
  SynthID-positive Veo clips, PSNR 27-35 dB; plain re-encode controls stay detected.
  The Python Wan path refuses anything but CUDA BF16 or MPS; the Rust port runs on
  AMD through wgpu.
