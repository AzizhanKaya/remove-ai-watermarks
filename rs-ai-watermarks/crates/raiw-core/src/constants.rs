//! Media-type registries shared by every subsystem (port of `_internal/constants.py`).

pub const JPEG_SUFFIXES: &[&str] = &[".jpg", ".jpeg", ".jpe", ".jfif"];
pub const SUPPORTED_FORMATS: &[&str] = &[
    ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".webp", ".heic", ".heif", ".avif",
];

pub const AUDIO_SCHEMES: &[&str] = &["audioseal", "wavmark", "perth", "silentcipher", "all", "auto"];
pub const AUDIO_PROFILES: &[&str] = &["spectral", "encodec", "combined", "dac", "snac"];
pub const AUDIO_METHODS: &[&str] = &["auto", "spectral", "encodec", "combined", "dac", "snac"];
pub const AUDIO_VENDORS: &[&str] = &[
    "auto",
    "google",
    "elevenlabs",
    "suno",
    "udio",
    "openai",
    "cartesia",
    "meta",
    "resemble",
    "microsoft",
];
pub const AUDIO_VOCODER_DEVICES: &[&str] = &["auto", "cpu", "mps", "cuda"];
pub const AUDIO_BANDWIDTHS_KBPS: &[u32] = &[6, 12, 24];
pub const AUDIO_EXTENSIONS: &[&str] = &[
    ".wav", ".flac", ".mp3", ".m4a", ".ogg", ".oga", ".opus", ".aac", ".mka",
];
pub const AUDIO_OUTPUT_EXTENSIONS: &[&str] = AUDIO_EXTENSIONS;
pub const AUDIO_OPUS_EXTENSIONS: &[&str] = &[".ogg", ".oga", ".opus"];
pub const AUDIO_LOSSY_OUTPUT_EXTENSIONS: &[&str] = &[".mp3", ".m4a", ".ogg", ".oga", ".opus", ".aac"];

pub const TEXT_EXTENSIONS: &[&str] = &[
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".html",
    ".htm",
    ".xml",
    ".svg",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
    ".tex",
];

pub const ISOBMFF_VIDEO_EXTENSIONS: &[&str] = &[".mp4", ".mov", ".m4v"];
pub const VIDEO_EXTENSIONS: &[&str] = &[".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".flv"];

pub const WAN_EXPERIMENTAL_PROFILES: &[&str] = &[
    "wan21-synthid",
    "wan21-seedance",
    "wan21-synthid-experimental",
    "wan21-seedance-experimental",
    "wan22-synthid",
    "wan22-synthid-experimental",
];
/// Rust-only Wan 2.2 TI2V 5B profiles (the Python package has no Wan 2.2 path).
pub const WAN22_EXPERIMENTAL_PROFILES: &[&str] = &["wan22-synthid", "wan22-synthid-experimental"];
pub const VIDEO_INVISIBLE_PROFILES: &[&str] = &[
    "auto",
    "synthid",
    "seedance",
    "wan21-synthid",
    "wan21-seedance",
    "wan22-synthid",
    "seedance-experimental",
    "wan21-synthid-experimental",
    "wan21-seedance-experimental",
    "wan22-synthid-experimental",
];
pub const VIDEO_PROFILE_ALIASES: &[(&str, &str)] = &[
    ("seedance", "seedance-experimental"),
    ("wan21-synthid", "wan21-synthid-experimental"),
    ("wan21-seedance", "wan21-seedance-experimental"),
    ("wan22-synthid", "wan22-synthid-experimental"),
];
pub const VIDEO_ENCODERS: &[&str] = &["software", "auto", "videotoolbox", "nvenc"];
pub const ALL_PRESETS: &[&str] = &["lossless-fast", "balanced", "deep-clean"];

pub const DEFAULT_VIDEO_MESSAGE_LONG_SIDE: u32 = 512;
pub const VIDEO_MESSAGE_SCHEMES: &[&str] = &["videoseal", "pixelseal", "chunkyseal"];
pub const VIDEO_MESSAGE_METHODS: &[&str] = &["auto", "codec", "vae"];
pub const VIDEO_MESSAGE_DEVICES: &[&str] = &["auto", "cpu", "cuda", "mps"];

pub const AI_METADATA_KEYS: &[&str] = &[
    "parameters",
    "postprocessing",
    "extras",
    "workflow",
    "prompt",
    "Dream",
    "SD:mode",
    "StableDiffusionVersion",
    "generation_time",
    "Model",
    "Model hash",
    "Seed",
];
pub const AI_KEYWORDS: &[&str] = &[
    "prompt",
    "negative_prompt",
    "sampler",
    "cfg_scale",
    "lora",
    "diffusion",
    "comfy",
    "midjourney",
    "dall-e",
    "dalle",
    "imagen",
    "firefly",
    "c2pa",
    "chatgpt",
    "gpt-4",
    "sora",
    "openai",
    "truepic",
    "stable_diffusion",
    "invokeai",
];

pub const PNG_SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";
pub const C2PA_CHUNK_TYPE: &[u8] = b"caBX";
pub const PNG_METADATA_CHUNKS: &[&[u8]] = &[b"tEXt", b"iTXt", b"zTXt", b"eXIf", b"iCCP"];
pub const RIFF_METADATA_CHUNKS: &[&[u8]] = &[b"EXIF", b"XMP ", b"ICCP", b"C2PA"];
pub const RIFF_CODED_IMAGE_CHUNKS: &[&[u8]] = &[b"VP8 ", b"VP8L", b"ALPH", b"ANMF"];
pub const C2PA_SIGNATURES: &[&[u8]] = &[
    b"c2pa",
    b"C2PA",
    b"jumb",
    b"jumd",
    b"JUMBF",
    b"jumbf",
    b"cbor",
    b"contentcreds",
    b"digid",
    b"assertions",
    b"manifest",
];

/// Whether `suffix` (lower-case, with dot) is in `set`.
pub fn has_suffix(set: &[&str], suffix: &str) -> bool {
    set.contains(&suffix)
}
