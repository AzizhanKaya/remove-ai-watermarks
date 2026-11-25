# rs-ai-watermarks

Rust port of `remove-ai-watermarks`: identify and remove visible, invisible, and
provenance AI marks in images, video, audio, and text. The Python package in
`../src/remove_ai_watermarks` is the behavioral reference; every Rust module is
proven against it with differential golden tests.

```bash
cargo run --release -- --help          # the `rs-ai-watermarks` binary
cargo test --workspace                 # unit + golden tests
cargo test --workspace --features gpu  # adds burn model tests (Vulkan/Metal/DX12/CPU)
```

## Crate map

| Crate | Role | Python reference |
| --- | --- | --- |
| `raiw-core` | Constants, Python-exact JSON, staged writes, hashing, schema checks, golden helpers | `_internal/constants.py` (media sets), `_internal/utils.py`, `_internal/schema.py` |
| `raiw-container` | ISOBMFF, EBML, RIFF, FLV, AV1 OBU, TIFF, PNG/JPEG/GIF walkers | `_internal/isobmff.py`, `ebml.py`, `riff.py`, `flv.py`, `av1_obu.py`, `tiff_strip.py` |
| `raiw-c2pa` | C2PA reading (`c2pa` crate, same c2pa-rs line as c2pa-python 0.38), JUMBF/CBOR fallback, vendor and soft-binding registries | `_internal/c2pa.py`, `_internal/extractor.py`, `_generated_c2pa_soft_bindings.py`, C2PA parts of `constants.py` |
| `raiw-metadata` | AI metadata scan/strip, readability probe, forensic metadata, portable record, TC260/SM2 | `metadata.py`, `forensic_metadata.py`, `metadata_record.py`, `_internal/audio_creation.py`, `_internal/tc260_*.py`, `_internal/sm2.py` |
| `raiw-text` | C2PATXT, C2PA HTML/ASCII armor, VSRMark, Unicode steganography, span protection, Ollama transport | `text.py`, `text_api.py`, `text_roles.py`, `_internal/standardized_variants.py`, `_internal/emoji_flag_codes.py`, `_internal/synthid_threefry.py` |
| `raiw-cv` | OpenCV 4.x-exact kernels (resize, cvtColor, GaussianBlur, Sobel, morphology, matchTemplate, inpaint, remap, warpAffine, optical flow, connected components, distance transform) | every `cv2.*` call site |
| `raiw-visible` | Visible mark registry, Gemini reverse alpha blend, text-mark engines, region eraser, image IO | `watermark_registry.py`, `*_engine.py`, `_text_mark_engine.py`, `_text_mark_variants.py`, `gemini_engine.py`, `pill_engine.py`, `region_eraser.py`, `image_io.py` |
| `raiw-pixel` | DWT-DCT, InvisMark, TrustMark (burn), pixel evidence | `dwt_dct.py`, `invisible_watermark.py`, `microsoft_invismark.py`, `trustmark_detector.py`, `pixel_evidence.py` |
| `raiw-identify` | Provenance report | `identify.py` |
| `raiw-audio` | ffmpeg IO, resampling, spectral profile, quality gates, AudioSeal/WavMark/Perth/SilentCipher/DAC/SNAC/EnCodec (burn) | `audio.py`, `_internal/audio_*.py`, `_internal/audioseal.py`, `wavmark.py`, `perth.py`, `silentcipher.py`, `_snac_network.py` |
| `raiw-video` | Stream-copy metadata, temporal visible removal, encoding, VideoSeal-family messages | `video*.py`, `_internal/video_message_*.py` |
| `raiw-classify` | Source-pipeline classifier | `source_classify.py`, `_internal/source_features_v4.py`, `forensic_124d.py`, `source_spectral.py` |
| `raiw-gpu` | Burn runtime: device selection (wgpu on Vulkan/Metal/DX12 for AMD, NVIDIA, Intel, Apple; CPU fallback), Hugging Face hub cache, PyTorch/safetensors weight loading, shared layers, and every neural architecture under `models::{audio, image, video, text}` so burn compiles in one crate | torch call sites, `audioseal`, `wavmark`, `perth`, `silentcipher`, DAC, SNAC, EnCodec, TrustMark, VideoSeal, LaMa, MI-GAN |
| `raiw-regen` | Invisible regeneration profiles | `invisible_engine.py`, `_internal/watermark_remover.py`, `watermark_profiles.py`, `two_stage_pipeline.py`, `*_zimage_pipeline.py`, `humanizer.py`, `text_restoration.py`, `tiling.py` |
| `raiw-mcp` | MCP stdio server, operation catalog | `mcp_server.py`, `integration.py` |
| `rs-ai-watermarks` | Facade library and the CLI binary | `cli.py`, `api.py`, `optional_deps.py`, `__init__.py` |

Dependencies flow downward only: `core` -> `container` -> `c2pa` -> `metadata`;
`core` -> `cv` -> `visible`/`pixel`/`classify`; `identify`, `audio`, `video`,
`regen` and `mcp` sit above them, and only `rs-ai-watermarks` sees everything.

## Differential golden tests

The loop is: pick a Python function, record its outputs on tracked fixtures and
synthetic inputs, port it, and iterate until the Rust output matches.

1. A generator `tools/goldens/<suite>.py` imports the Python package and writes
   `goldens/<suite>/<name>.json`. Run it from the repository root with the
   existing environment: `.venv/bin/python rs-ai-watermarks/tools/goldens/<suite>.py`
   (never `uv sync`; it would mutate the shared environment). Suites that need
   optional extras the development `.venv` lacks (pywt, scipy, audioseal, wavmark,
   perth, trustmark, onnxruntime) use a separate oracle environment outside the
   repository: `uv venv --python 3.12 ~/.cache/rs-ai-watermarks/oracle-venv` then
   `uv pip install --python ~/.cache/rs-ai-watermarks/oracle-venv/bin/python -e
   '.[video,detect,heif,trustmark,source-classify,lama,migan,audio-detect,audio-wavmark,audio-perth,audio-silentcipher,audio-codec,text-api,diffusion]' soundfile`.
2. Rust tests load them with `raiw_core::testing::golden(suite, name)` and compare
   with `assert_eq!` for discrete results or `assert_json_close` for floats.
3. Fixtures are referenced in place through `raiw_core::testing::fixture(...)`;
   binaries are never copied. Generated media goes to a temp directory.

Speed is part of the contract: every hot path gets a timing comparison against
the Python reference (`tools/bench/`), and the Rust side must be at least as fast.

## Conventions

- Edition 2024, toolchain 1.96 (the c2pa-rs 0.91 line needs it), `unsafe` denied, workspace clippy lints.
- Comments only as short section headers or one-line reminders.
- Library errors use `thiserror`; only the binary uses `anyhow`.
- JSON that the CLI prints goes through `raiw_core::to_py_json`, so key order,
  float formatting and ASCII escaping match `json.dumps` exactly. Serde structs
  keep the Python field names and order.
- Optional heavy backends sit behind cargo features (the Python extras): `gpu`
  pulls burn; every default path stays dependency-light.
