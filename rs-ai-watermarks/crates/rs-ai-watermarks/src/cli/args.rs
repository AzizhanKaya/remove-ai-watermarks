//! Argument types shared across commands: Click-style toggles, path checks and choice enums.

use std::path::PathBuf;

use clap::{Args, ValueEnum};

/// Declare Click-style `--on/--off` boolean pairs (the last spelling wins) and the
/// `TOGGLES` table integrations read their negative spelling and default from.
macro_rules! toggles {
    ($($name:ident($id:literal, $on:literal, $off:literal, $default:tt, $help:literal);)+) => {
        $(toggles!(@one $name, $id, $on, $off, $default, $help);)+

        pub const TOGGLES: &[raiw_mcp::ToggleSpec] = &[$(raiw_mcp::ToggleSpec {
            id: $id,
            off_id: concat!($id, "_off"),
            negative_flag: concat!("--", $off),
            default: toggles!(@default $default),
        }),+];
    };
    (@default unset) => { None };
    (@default $value:literal) => { Some($value) };
    (@pair $name:ident, $id:literal, $on:literal, $off:literal, $help:literal) => {
        #[derive(Debug, Clone, Copy, Args)]
        pub struct $name {
            #[arg(long = $on, id = $id, overrides_with = concat!($id, "_off"), help = $help)]
            on: bool,
            #[arg(long = $off, id = concat!($id, "_off"), overrides_with = $id, hide = true)]
            off: bool,
        }

        impl $name {
            /// The explicit choice, if either spelling was given.
            pub const fn explicit(self) -> Option<bool> {
                if self.on {
                    Some(true)
                } else if self.off {
                    Some(false)
                } else {
                    None
                }
            }
        }
    };
    (@one $name:ident, $id:literal, $on:literal, $off:literal, unset, $help:literal) => {
        toggles!(@pair $name, $id, $on, $off, $help);

        impl $name {
            pub const fn get(self) -> Option<bool> {
                self.explicit()
            }
        }
    };
    (@one $name:ident, $id:literal, $on:literal, $off:literal, $default:literal, $help:literal) => {
        toggles!(@pair $name, $id, $on, $off, $help);

        impl $name {
            pub const fn get(self) -> bool {
                match self.explicit() {
                    Some(value) => value,
                    None => $default,
                }
            }
        }
    };
}

toggles! {
    StripMetadata("strip_metadata", "strip-metadata", "keep-metadata", true,
        "Strip AI metadata from output (--keep-metadata keeps it).");
    KeepStandard("keep_standard", "keep-standard", "remove-all", true,
        "Keep standard metadata such as Author and Title (--remove-all drops it).");
    Detect("detect", "detect", "no-detect", true, "Detect watermark before removal.");
    Verify("verify", "verify", "no-verify", true, "Check selected compatible models before and after.");
    Invisible("invisible", "invisible", "no-invisible", false, "Opt into the invisible (pixel or waveform) stage.");
    TemporalConsistency("temporal_consistency", "temporal-consistency", "no-temporal-consistency", true,
        "Motion-align adjacent accepted fills to reduce frame-to-frame flicker.");
    TemporalStabilize("temporal_stabilize", "temporal-stabilize", "no-temporal-stabilize", true,
        "Motion-compensated optical-flow stabilization across VAE-regenerated frames.");
    AudioInvisible("audio_invisible", "audio-invisible", "no-audio-invisible", false,
        "Inspect and clean compatible audio watermarks across all audio streams.");
    AudioInspect("audio", "audio", "no-audio", false, "Inspect audio streams with compatible public decoders.");
    AdaptivePolish("adaptive_polish", "adaptive-polish", "no-adaptive-polish", unset,
        "Restore the input's detail level after removal (default: the profile decides).");
    Tile("tile", "tile", "no-tile", false, "Process large images in overlapping tiles instead of one forward pass.");
    Force("force", "force", "no-force", false,
        "Run the diffusion scrub even when no invisible AI watermark is locally detectable.");
    CpuOffload("cpu_offload", "cpu-offload", "no-cpu-offload", false,
        "Offload model components to host memory between GPU calls, at the cost of speed.");
    FidelityAnchor("fidelity_anchor", "fidelity-anchor", "no-fidelity-anchor", false,
        "With --text-manifest: blend 15% of the VAE donor across the whole frame.");
}

/// Numeric bounds Click declares through `IntRange`/`FloatRange`, keyed by parameter id.
pub const BOUNDS: &[raiw_mcp::BoundSpec] = &[
    raiw_mcp::BoundSpec::int("stream", Some(0), None),
    raiw_mcp::BoundSpec::int("target_streams", Some(0), None),
    raiw_mcp::BoundSpec::int("audio_streams", Some(0), None),
    raiw_mcp::BoundSpec::int("long_side", Some(8), None),
    raiw_mcp::BoundSpec::int("batch_size", Some(1), None),
    raiw_mcp::BoundSpec::int("wan_steps", Some(1), Some(50)),
    raiw_mcp::BoundSpec::float("fps", Some(1.0), None),
    raiw_mcp::BoundSpec::float("noise_std", Some(0.0), Some(1.0)),
    raiw_mcp::BoundSpec::int("dilate", Some(0), None).only_in(&["video", "erase"]),
];

/// A path that must exist and must not be a directory (Click `Path(exists=True, dir_okay=False)`).
pub fn existing_file(value: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(value);
    match std::fs::metadata(&path) {
        Ok(meta) if meta.is_dir() => Err(format!("File '{value}' is a directory.")),
        Ok(_) => Ok(path),
        Err(_) => Err(format!("File '{value}' does not exist.")),
    }
}

/// A path that must exist and must be a directory (Click `Path(exists=True, file_okay=False)`).
pub fn existing_dir(value: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(value);
    match std::fs::metadata(&path) {
        Ok(meta) if meta.is_dir() => Ok(path),
        Ok(_) => Err(format!("Directory '{value}' is a file.")),
        Err(_) => Err(format!("Directory '{value}' does not exist.")),
    }
}

/// Declare a `ValueEnum` whose variants spell exactly the Python choice strings.
macro_rules! choices {
    ($(#[$meta:meta])* $name:ident { $($variant:ident = $text:literal),+ $(,)? }) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ValueEnum)]
        pub enum $name {
            $(#[value(name = $text)] $variant),+
        }

        impl $name {
            pub const fn as_str(self) -> &'static str {
                match self {
                    $(Self::$variant => $text),+
                }
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(self.as_str())
            }
        }
    };
}

choices!(ImageMark {
    Auto = "auto", Gemini = "gemini", Doubao = "doubao", Jimeng = "jimeng", Qwen = "qwen", Wan = "wan",
    Kling = "kling", Yuanbao = "yuanbao", Dola = "dola", Workbuddy = "workbuddy", Samsung = "samsung",
    Runninghub = "runninghub", Baidu = "baidu", Liblib = "liblib", LiblibPill = "liblib_pill",
    Microsoft = "microsoft", GenericAiLabel = "generic_ai_label", Openart = "openart", Genspark = "genspark",
    JimengPill = "jimeng_pill",
});

choices!(VideoMark {
    Auto = "auto", Sora = "sora", Veo = "veo", Dreamina = "dreamina", Jimeng = "jimeng", JimengPill = "jimeng_pill",
    Seedance = "seedance", Doubao = "doubao", Dola = "dola", Hailuo = "hailuo", Vidu = "vidu", Wan = "wan",
    Qwen = "qwen", Yuanbao = "yuanbao", Baidu = "baidu", Runninghub = "runninghub", Microsoft = "microsoft",
    Pixverse = "pixverse", Luma = "luma", Heygen = "heygen", SyncLabs = "sync_labs", TiktokAi = "tiktok_ai",
    GenericAiLabel = "generic_ai_label", Kling = "kling",
});

choices!(FillBackend { Auto = "auto", Cv2 = "cv2", Migan = "migan", Lama = "lama" });
choices!(Sensitivity { Auto = "auto", Strict = "strict" });
choices!(InpaintMethod { Telea = "telea", Ns = "ns" });
choices!(ImageVendor { Auto = "auto", Openai = "openai", Google = "google", Microsoft = "microsoft", Meta = "meta" });
choices!(AllVendor {
    Auto = "auto", Openai = "openai", Google = "google", Microsoft = "microsoft", Meta = "meta",
    Elevenlabs = "elevenlabs", Suno = "suno", Udio = "udio", Cartesia = "cartesia", Resemble = "resemble",
});
choices!(Pipeline {
    QwenZimage = "qwen-zimage", Qwen21Zimage = "qwen21-zimage", SdxlZimage = "sdxl-zimage",
    ChromaZimage = "chroma-zimage", Auto = "auto",
});
choices!(AllPreset { LosslessFast = "lossless-fast", Balanced = "balanced", DeepClean = "deep-clean" });
choices!(BatchMode { Visible = "visible", Invisible = "invisible", Metadata = "metadata", All = "all" });
choices!(AudioBatchMode { All = "all", Metadata = "metadata" });
choices!(VideoBatchMode { All = "all", Visible = "visible", Metadata = "metadata" });

choices!(AudioScheme {
    Audioseal = "audioseal", Wavmark = "wavmark", Perth = "perth", Silentcipher = "silentcipher", All = "all",
    Auto = "auto",
});
choices!(AudioMethod {
    Auto = "auto", Spectral = "spectral", Encodec = "encodec", Combined = "combined", Dac = "dac", Snac = "snac",
});
choices!(Bandwidth { Six = "6", Twelve = "12", TwentyFour = "24" });
choices!(VocoderDevice { Auto = "auto", Cpu = "cpu", Mps = "mps", Cuda = "cuda" });
choices!(AudioVendor {
    Auto = "auto", Google = "google", Elevenlabs = "elevenlabs", Suno = "suno", Udio = "udio", Openai = "openai",
    Cartesia = "cartesia", Meta = "meta", Resemble = "resemble", Microsoft = "microsoft",
});

choices!(MessageScheme { Videoseal = "videoseal", Pixelseal = "pixelseal", Chunkyseal = "chunkyseal" });
choices!(MessageMethod { Auto = "auto", Codec = "codec", Vae = "vae" });
choices!(Device { Auto = "auto", Cuda = "cuda", Mps = "mps", Cpu = "cpu" });
choices!(VideoQuality { Delivery = "delivery", Master = "master" });
choices!(VideoProfile {
    Auto = "auto", Synthid = "synthid", Seedance = "seedance", Wan21Synthid = "wan21-synthid",
    Wan21Seedance = "wan21-seedance", SeedanceExperimental = "seedance-experimental",
    Wan21SynthidExperimental = "wan21-synthid-experimental", Wan21SeedanceExperimental = "wan21-seedance-experimental",
    Wan22Synthid = "wan22-synthid", Wan22SynthidExperimental = "wan22-synthid-experimental",
});
choices!(RewriterBackend { Transformers = "transformers", Mlx = "mlx", Ollama = "ollama" });

impl Bandwidth {
    pub const fn kbps(self) -> u32 {
        match self {
            Self::Six => 6,
            Self::Twelve => 12,
            Self::TwentyFour => 24,
        }
    }
}

/// `--check` / `--remove` / `-o` / `--keep-standard` shared by every metadata command.
#[derive(Debug, Clone, Args)]
pub struct MetadataArgs {
    /// Check for AI metadata (don't modify).
    #[arg(long)]
    pub check: bool,
    /// Remove AI metadata.
    #[arg(long)]
    pub remove: bool,
    #[command(flatten)]
    pub keep_standard: KeepStandard,
}

/// Waveform-processing options shared by `audio invisible|all|batch` and `video audio`.
#[derive(Debug, Clone, Args)]
pub struct AudioProcessArgs {
    /// Select automatically, or choose spectral, EnCodec, combined, 44.1 kHz DAC, or 44.1 kHz SNAC.
    #[arg(long, value_enum, default_value_t = AudioMethod::Auto)]
    pub method: AudioMethod,
    /// EnCodec bandwidth in kbps; applies to encodec and combined methods.
    #[arg(long, value_enum, default_value_t = Bandwidth::TwentyFour)]
    pub bandwidth: Bandwidth,
    /// Neural audio inference device for DAC, SNAC, EnCodec, and combined; spectral requires cpu.
    #[arg(long = "vocoder-device", value_enum, default_value_t = VocoderDevice::Auto)]
    pub vocoder_device: VocoderDevice,
    /// Known source vendor when metadata was stripped; authorizes scrubbing without --allow-unverified.
    #[arg(long, value_enum, default_value_t = AudioVendor::Auto)]
    pub vendor: AudioVendor,
    /// Process without a local source-positive decoder match, keeping the quality gate active.
    #[arg(long = "allow-unverified")]
    pub allow_unverified: bool,
    /// Permit writing an output that still tests positive under local decoders on the final attempt.
    #[arg(long = "allow-positive")]
    pub allow_positive: bool,
    /// Permit writing an output that fails the audio SNR/RMS/window integrity gate on the final attempt.
    #[arg(long = "allow-quality-override")]
    pub allow_quality_override: bool,
    /// Alias enabling --allow-unverified, --allow-positive, and --allow-quality-override.
    #[arg(long)]
    pub force: bool,
    #[command(flatten)]
    pub verify: Verify,
}

/// Video-pixel regeneration options shared by `video invisible|all|batch`.
#[derive(Debug, Clone, Args)]
pub struct VideoRegenArgs {
    /// SynthID shared latent-noise strength. Higher values change more detail.
    #[arg(long = "noise-std", default_value_t = 0.15, value_parser = unit_interval)]
    pub noise_std: f64,
    /// SynthID output long side in pixels.
    #[arg(long = "long-side", default_value_t = 512, value_parser = clap::value_parser!(u32).range(8..))]
    pub long_side: u32,
    /// SynthID output frame rate, capped at the source frame rate [default: 12.0].
    #[arg(long, value_parser = at_least_one)]
    pub fps: Option<f64>,
    /// SynthID inference batch size; Seedance uses a fixed batch size.
    #[arg(long = "batch-size", default_value_t = 4, value_parser = clap::value_parser!(u32).range(1..))]
    pub batch_size: u32,
    /// SynthID noise seed.
    #[arg(long, default_value_t = 0)]
    pub seed: i64,
    /// Preserve the source video's native frame rate instead of capping at --fps.
    #[arg(long = "keep-fps")]
    pub keep_fps: bool,
    #[command(flatten)]
    pub temporal_stabilize: TemporalStabilize,
    /// Run fast 512p 2D VAE regeneration and upscale cleanly to native resolution and frame rate.
    #[arg(long = "native-guide")]
    pub native_guide: bool,
    /// Inference steps for Wan 2.1 3D profiles (default: certified 20).
    #[arg(long = "wan-steps", value_parser = clap::value_parser!(u32).range(1..=50))]
    pub wan_steps: Option<u32>,
    /// Evict cached neural models and empty GPU caches between stages and files.
    #[arg(long = "low-memory")]
    pub low_memory: bool,
    /// VAE inference device.
    #[arg(long, value_enum, default_value_t = Device::Auto)]
    pub device: Device,
    /// Output quality; master requires seedance, wan21-synthid, or wan21-seedance.
    #[arg(long, value_enum, default_value_t = VideoQuality::Delivery)]
    pub quality: VideoQuality,
    /// Video-pixel profile. 'auto' routes ByteDance/Seedance and SynthID clips automatically.
    #[arg(long, value_enum, default_value_t = VideoProfile::Auto)]
    pub profile: VideoProfile,
}

impl VideoRegenArgs {
    /// `_effective_cli_fps`: `None` keeps the source rate (`--keep-fps`, or `--native-guide` with a default `--fps`).
    pub fn effective_fps(&self) -> Option<f64> {
        if self.keep_fps || (self.native_guide && self.fps.is_none()) {
            None
        } else {
            Some(self.fps.unwrap_or(12.0))
        }
    }
}

/// Audio cleaning options embedded in `video all|batch` with the `--audio-` prefix.
#[derive(Debug, Clone, Args)]
pub struct VideoAudioArgs {
    #[command(flatten)]
    pub audio_invisible: AudioInvisible,
    /// Compatible public models to check when --audio-invisible is set.
    #[arg(long = "audio-scheme", value_enum, default_value_t = AudioScheme::Auto)]
    pub audio_scheme: AudioScheme,
    /// Audio watermark removal method when --audio-invisible is enabled.
    #[arg(long = "audio-method", value_enum, default_value_t = AudioMethod::Auto)]
    pub audio_method: AudioMethod,
    /// Known audio vendor when metadata was stripped; enables wideband audio scrubbing.
    #[arg(long = "audio-vendor", value_enum, default_value_t = AudioVendor::Auto)]
    pub audio_vendor: AudioVendor,
    /// Alias enabling every --audio-allow-* override when --audio-invisible is set.
    #[arg(long = "audio-force")]
    pub audio_force: bool,
    /// Process without a local source-positive decoder match, keeping the quality gate active.
    #[arg(long = "audio-allow-unverified")]
    pub audio_allow_unverified: bool,
    /// Permit writing an output that still tests positive under local decoders on the final attempt.
    #[arg(long = "audio-allow-positive")]
    pub audio_allow_positive: bool,
    /// Permit writing an output that fails the audio integrity gate on the final attempt.
    #[arg(long = "audio-allow-quality-override")]
    pub audio_allow_quality_override: bool,
    /// Restrict --audio-invisible to specific zero-based audio stream ordinals (repeatable).
    #[arg(long = "audio-stream")]
    pub audio_streams: Vec<u32>,
}

/// Diffusion options shared by `invisible`, `all`, and `batch`.
#[derive(Debug, Clone, Args)]
pub struct DiffusionArgs {
    /// Denoising strength (0.0-1.0). Default: profile-adaptive.
    #[arg(long)]
    pub strength: Option<f64>,
    /// Random seed for reproducibility (default 0: every profile is certified at a fixed seed).
    #[arg(long)]
    pub seed: Option<i64>,
    /// Hugging Face API token.
    #[arg(long = "hf-token")]
    pub hf_token: Option<String>,
    /// Analog Humanizer film grain intensity (0 = off, typical: 2.0-6.0).
    #[arg(long, default_value_t = 0.0)]
    pub humanize: f64,
    /// Cap long side (px) before diffusion; 0 = native.
    #[arg(long = "max-resolution", default_value_t = 0)]
    pub max_resolution: u32,
    /// Canny ControlNet conditioning scale on the global stage.
    #[arg(long = "controlnet-scale", default_value_t = 1.0)]
    pub controlnet_scale: f64,
    /// Unsharp-mask sharpening strength (0 = off, typical: 0.3-0.8).
    #[arg(long, default_value_t = 0.0)]
    pub unsharp: f64,
    #[command(flatten)]
    pub adaptive_polish: AdaptivePolish,
    #[command(flatten)]
    pub tile: Tile,
    /// Tile dimension in px for --tile.
    #[arg(long = "tile-size", default_value_t = 1024)]
    pub tile_size: u32,
    /// Overlap between adjacent tiles in px (feather-blended).
    #[arg(long = "tile-overlap", default_value_t = 128)]
    pub tile_overlap: u32,
    #[command(flatten)]
    pub force: Force,
    #[command(flatten)]
    pub cpu_offload: CpuOffload,
}

/// Text-manifest options of `invisible` and `all`.
#[derive(Debug, Clone, Args)]
pub struct TextRestorationArgs {
    /// Experimental verified-text restoration manifest.
    #[arg(long = "text-manifest", value_parser = existing_file)]
    pub text_manifest: Option<PathBuf>,
    #[command(flatten)]
    pub fidelity_anchor: FidelityAnchor,
}

fn unit_interval(value: &str) -> Result<f64, String> {
    let parsed: f64 = value
        .parse()
        .map_err(|_| format!("'{value}' is not a valid float."))?;
    if (0.0..=1.0).contains(&parsed) {
        Ok(parsed)
    } else {
        Err(format!("{value} is not in the range 0<=x<=1."))
    }
}

fn at_least_one(value: &str) -> Result<f64, String> {
    let parsed: f64 = value
        .parse()
        .map_err(|_| format!("'{value}' is not a valid float."))?;
    if parsed >= 1.0 {
        Ok(parsed)
    } else {
        Err(format!("{value} is not in the range x>=1.0."))
    }
}

pub fn optional_unit_interval(value: &str) -> Result<f64, String> {
    unit_interval(value)
}
