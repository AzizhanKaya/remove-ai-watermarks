# Video SynthID trial on a user-provided clip (2026-11-26)

A first real end-to-end run of the Rust `video all` pipeline on the user's own
AI-generated clip (a Higgsfield output, local file in `~/Downloads`, not part of
the repository).

## Source

- `hf_20261124_221145_fef060d8-….mp4`: H.264, 480×854, 24 fps, 13.04 s, 313 frames,
  no audio stream. The user's `.mov` upload is a Core Media HEVC re-encode of it.
- Rust and Python `identify` agree: no C2PA, no AI metadata, no visible mark, no
  invisible-watermark hint. Raw tags hold only `encoder=Lavf63.1.102` and
  `handler_name=VideoHandler`.
- synthid.com (user's screenshot): **SynthID detected**. Because nothing in the
  metadata names Google, `--profile auto` cannot route it; `--profile synthid` is
  required.

## Outputs (all written next to the source in `~/Downloads`)

| File suffix | Command | Geometry | Time | Notes |
| --- | --- | --- | --- | --- |
| `_clean_synthid` | `video all --invisible --profile synthid` | 288×512, 12 fps, 157 frames | 359 s | Oracle-certified operating point (noise 0.15) |
| `_clean_native` | `… --native-guide` | 480×854, 24 fps, 313 frames | 653 s | PSNR 36.18 dB vs source; not oracle-verified |
| `_clean_native_denoised` | `_clean_native` + `nlmeans=s=4:p=7:r=13,hqdn3d=2:2:6:6` | 480×854, 24 fps | 73 s (CPU) | Less blotchy |
| `_clean_native_n012` | `… --native-guide --noise-std 0.12` | 480×854, 24 fps | 662 s | Slightly less blotchy than 0.15 |
| `_clean_n012_trim` | `_clean_native_n012`, first 12 frames cut | 480×854, 24 fps, 301 frames, 12.54 s | seconds | The first 0.5 s of the source is grainier and darker |
| `_clean_n012_trim_denoised` | trimmed + the denoise filter | same | ~1 min (CPU) | Visually the cleanest; recommended to test first |

Every output: no C2PA, no AI metadata markers (Rust and Python `identify`), only
the local `encoder=Lavf63.16.100` tag, no audio. GPU at 99%, 4.3 GB VRAM, process
~50% CPU during regeneration.

## Observations

- The user found the outputs "very dirty": the 2D VAE latent-noise method turns
  flat skin into blotchy low-frequency texture. This is the method, not a port
  defect (Rust matches Python's output within 2e-5 dB PSNR).
- Removing audio is a non-issue here (the source has none). In general an audio
  track can carry its own SynthID (e.g. Veo 3 audio), so dropping it removes that
  mark; a silent video is common and the only processing trace in the container is
  the ffmpeg `encoder` tag the source already had.

## Wan 2.1 run on the RX 5500 XT (2026-11-26)

- Prompt cache: `video_wan`'s own cold-miss branch (`encode_prompt`, bfloat16, CPU) with
  UMT5-XXL disk-offloaded through accelerate (3 GiB CPU budget, ~4 GB RSS); 31 min download
  (22.7 GB at ~12 MB/s), 70 s to load and encode. The cache is `[1, 226, 4096]` (sha256
  `54269002…27ecc1`), which exposed the Rust reader's 512-row assumption (fixed).
- `video all --invisible --profile wan21-synthid` (Rust release, fused f16 kernels):
  `_clean_wan21`, 480×854, 24 fps, 313 frames, **61 min** (five 81-frame windows, 17-frame
  overlaps). PSNR 36.3 dB vs source (min 33.6). Natural skin without the 2D VAE blotches,
  faithful face and motion, no visible seam at the window crossfades.
- A 17-frame smoke run with a random prompt embedding looked smeared and plastic: the
  conditioning matters, so smoke tests must use the real cache.
- Not yet checked on synthid.com.

## Where we stopped

1. **Waiting on the user:** test `_clean_n012_trim_denoised` (and optionally
   `_clean_synthid`, `_clean_native`) on synthid.com. 0.12 and the denoise
   post-pass have no oracle verdict yet.
2. **If detected or still too dirty:** the stronger option is Wan 2.1
   (`--profile wan21-synthid`, 6/6 not detected in the oracle record, natural
   texture at native resolution). Needs, once, the UMT5-XXL prompt embedding
   (`tools/goldens/gpu_wan_prompt_cache.py`; ~23 GB download, ~12 GB RAM unless
   computed with layer streaming/disk offload) and ~2.5-3 h on this card for this
   clip; the Rust Wan pipeline has never run end to end on real weights.
3. **Speed work in progress** (GPU worker re-prioritized): fp16 SD VAE, multi-frame
   batches, faster Conv2d/GroupNorm decoder path, persistent kernel/autotune cache;
   then Wan fp16-resident blocks and a faster attention kernel. Realistic targets
   without changing results: 2D VAE path 11 min → ~4-5 min; Wan ~3 h → ~1-1.5 h.
   Reference workload: the 313-frame `--native-guide` run above (653-662 s).
