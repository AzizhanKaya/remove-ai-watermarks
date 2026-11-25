//! C2PA identity, tool and action registries (port of the C2PA half of `constants.py`).
//!
//! Tables keep Python's declaration order: it is product priority when a manifest names
//! more than one organization, and the order of every derived label list.

use std::sync::LazyLock;

/// One C2PA identity and its AI-product or neutral-signer attribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C2paAiVendor {
    pub issuer: &'static [u8],
    pub org: &'static str,
    pub platform: Option<&'static str>,
    pub needle: Option<&'static str>,
    pub synthid: bool,
    pub asserts_ai: bool,
    pub synthid_requires_watermark_action: bool,
    pub signer_platform: Option<&'static str>,
    pub raw_signer_platforms: &'static [(&'static [u8], &'static str)],
}

const fn vendor(
    issuer: &'static str,
    org: &'static str,
    platform: &'static str,
    needle: &'static str,
) -> C2paAiVendor {
    C2paAiVendor {
        issuer: issuer.as_bytes(),
        org,
        platform: Some(platform),
        needle: Some(needle),
        synthid: false,
        asserts_ai: false,
        synthid_requires_watermark_action: false,
        signer_platform: None,
        raw_signer_platforms: &[],
    }
}

/// A signer identity registered for display only: no AI platform and no AI assertion.
const fn signer(issuer: &'static str, org: &'static str) -> C2paAiVendor {
    C2paAiVendor {
        platform: None,
        needle: None,
        ..vendor(issuer, org, "", "")
    }
}

impl C2paAiVendor {
    const fn asserts_ai(mut self) -> Self {
        self.asserts_ai = true;
        self
    }

    const fn synthid(mut self) -> Self {
        self.synthid = true;
        self
    }

    const fn synthid_via_watermark_action(mut self) -> Self {
        self.synthid_requires_watermark_action = true;
        self
    }

    const fn signer_platform(mut self, platform: &'static str) -> Self {
        self.signer_platform = Some(platform);
        self
    }

    const fn raw_signer_platforms(mut self, pairs: &'static [(&'static [u8], &'static str)]) -> Self {
        self.raw_signer_platforms = pairs;
        self
    }
}

/// `C2PA_AI_VENDORS`, in product-priority order.
pub const C2PA_AI_VENDORS: &[C2paAiVendor] = &[
    vendor(
        "Microsoft",
        "Microsoft",
        "Microsoft (Copilot / Designer)",
        "Microsoft",
    ),
    vendor("Adobe", "Adobe", "Adobe Firefly", "Adobe"),
    vendor(
        "OpenAI",
        "OpenAI",
        "OpenAI (ChatGPT / GPT Image / DALL·E / Sora)",
        "OpenAI",
    )
    .synthid()
    .synthid_via_watermark_action(),
    vendor("Google", "Google LLC", "Google (Gemini / Imagen)", "Google").synthid(),
    vendor(
        "Stability AI",
        "Stability AI",
        "Stability AI (Stable Image / DreamStudio)",
        "Stability AI",
    ),
    vendor(
        "Black Forest Labs",
        "Black Forest Labs",
        "Black Forest Labs (FLUX)",
        "Black Forest Labs",
    ),
    vendor(
        "volcengine",
        "ByteDance (Volcano Engine)",
        "ByteDance Volcano Engine",
        "Volcano Engine",
    ),
    vendor(
        "北京火山引擎科技有限公司",
        "ByteDance (Volcano Engine)",
        "ByteDance Volcano Engine",
        "Volcano Engine",
    ),
    vendor(
        "Byteplus",
        "BytePlus (ByteDance)",
        "BytePlus (ByteDance)",
        "BytePlus",
    ),
    vendor(
        "Dreamina",
        "ByteDance (Dreamina)",
        "ByteDance Dreamina",
        "Dreamina",
    )
    .asserts_ai(),
    // Listed before Canva: the Nova Canvas software agent contains the Canva token.
    vendor("Amazon Web Services", "Amazon", "Amazon Bedrock (Nova)", "Amazon"),
    vendor("Canva", "Canva", "Canva (Magic Media)", "Canva"),
    vendor("Eleven Labs", "ElevenLabs", "ElevenLabs", "ElevenLabs"),
    vendor("Features & Labels", "fal.ai", "fal.ai", "fal.ai").asserts_ai(),
    vendor("fal-ai", "fal.ai", "fal.ai", "fal.ai").asserts_ai(),
    vendor("Bria", "Bria Artificial Intelligence", "Bria AI", "Bria").asserts_ai(),
    vendor("Ideogram", "Ideogram", "Ideogram", "Ideogram").asserts_ai(),
    vendor("xAI Grok Imagine", "xAI Grok Imagine", "xAI Grok Imagine", "xAI").asserts_ai(),
    vendor("Producer.ai", "Producer.ai", "Producer.ai", "Producer.ai").asserts_ai(),
    vendor("Synchronicity Labs", "Sync Labs", "Sync Labs", "Sync Labs").asserts_ai(),
    vendor("PixVerse", "PixVerse", "PixVerse", "PixVerse").asserts_ai(),
    vendor("Suno Inc", "Suno", "Suno", "Suno").asserts_ai(),
    vendor("pixelbin.io", "PixelBin.io", "PixelBin.io", "PixelBin.io"),
    vendor(
        "SPRING (SG) PTE. LTD.",
        "SPRING (SG) PTE. LTD.",
        "SPRING (SG) PTE. LTD.",
        "SPRING",
    ),
    // Signers of ordinary edits and publications: identity for display, never an AI verdict.
    signer("TikTok Inc.", "TikTok").signer_platform("TikTok (C2PA signer)"),
    signer("Bytedance Pte", "ByteDance")
        .signer_platform("ByteDance (C2PA signer)")
        .raw_signer_platforms(&[
            (b"Bytedance Pte. Ltd.", "ByteDance (C2PA signer)"),
            (b"CapCut/", "CapCut (C2PA signer)"),
            (b"capcut c2pa-rs", "CapCut (C2PA signer)"),
        ]),
    signer("Anthropic Claude Content Signing", "Anthropic Claude")
        .signer_platform("Anthropic Claude (C2PA signer)"),
    signer("Samsung Galaxy", "Samsung Galaxy").signer_platform("Samsung Galaxy (C2PA)"),
    signer("com.asus.gallery", "ASUS Gallery").signer_platform("ASUS Gallery (C2PA signer)"),
    signer("Google Photos", "Google Photos"),
    vendor("RUNWAY AI, INC.", "Runway", "Runway", "Runway"),
    vendor("Runway AI, Inc", "Runway", "Runway", "Runway"),
    vendor("Pruna AI", "Pruna AI", "Pruna AI", "Pruna AI").asserts_ai(),
    vendor("Kling AI", "Kuaishou (Kling AI)", "Kuaishou Kling AI", "Kling AI").asserts_ai(),
    vendor("Topaz Labs", "Topaz Labs", "Topaz Labs", "Topaz"),
    vendor(
        "CyberLink",
        "CyberLink PhotoDirector",
        "CyberLink PhotoDirector",
        "CyberLink",
    ),
    vendor(
        "PhotoDirector",
        "CyberLink PhotoDirector",
        "CyberLink PhotoDirector",
        "PhotoDirector",
    ),
    vendor(
        "AUTOENHANCE.AI",
        "Autoenhance.ai",
        "Autoenhance.ai",
        "Autoenhance.ai",
    ),
    vendor("vivo Mobile Communication", "vivo", "vivo", "vivo"),
    vendor("Pixelcut", "Pixelcut", "Pixelcut", "Pixelcut"),
    signer("Truepic", "Truepic"),
];

/// Google signers that re-encode rather than generate, so their identity alone is not SynthID.
pub const SYNTHID_EDIT_SIGNERS: &[&[u8]] = &[b"YouTube"];

/// `C2PA_ISSUERS`: issuer token to organization, in registry order (issuers are unique).
pub fn c2pa_issuers() -> impl Iterator<Item = (&'static [u8], &'static str)> {
    C2PA_AI_VENDORS.iter().map(|vendor| (vendor.issuer, vendor.org))
}

/// `C2PA_IDENTITY_AI_ORGS`: organizations whose signer identity alone asserts AI origin.
pub static C2PA_IDENTITY_AI_ORGS: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    let mut orgs: Vec<&str> = Vec::new();
    for vendor in C2PA_AI_VENDORS.iter().filter(|vendor| vendor.asserts_ai) {
        if !orgs.contains(&vendor.org) {
            orgs.push(vendor.org);
        }
    }
    orgs
});

pub fn is_identity_ai_org(org: &str) -> bool {
    C2PA_IDENTITY_AI_ORGS.contains(&org)
}

/// `C2PA_SIGNER_PLATFORM_BY_ORG`.
pub fn signer_platform_by_org(org: &str) -> Option<&'static str> {
    C2PA_AI_VENDORS
        .iter()
        .rev()
        .find(|vendor| vendor.org == org && vendor.signer_platform.is_some())?
        .signer_platform
}

/// `C2PA_SIGNER_PLATFORMS`: raw signer tokens and the neutral platform each names.
pub static C2PA_SIGNER_PLATFORMS: LazyLock<Vec<(&'static [u8], &'static str)>> = LazyLock::new(|| {
    C2PA_AI_VENDORS
        .iter()
        .filter_map(|vendor| vendor.signer_platform.map(|platform| (vendor, platform)))
        .flat_map(|(vendor, platform)| {
            if vendor.raw_signer_platforms.is_empty() {
                vec![(vendor.issuer, platform)]
            } else {
                vendor.raw_signer_platforms.to_vec()
            }
        })
        .collect()
});

/// Product-specific claim generators that may sign through a different upstream issuer.
pub const C2PA_CLAIM_GENERATOR_PLATFORMS: &[(&str, &str)] = &[
    ("grok imagine", "xAI Grok Imagine"),
    ("suno", "Suno"),
    ("adobe_firefly", "Adobe Firefly"),
    ("firefly", "Adobe Firefly"),
    ("dreamina", "ByteDance Dreamina"),
    ("pippit", "ByteDance Pippit AI"),
    ("dola", "ByteDance Dola"),
    ("sync labs", "Sync Labs"),
    ("pixelbin.io", "PixelBin.io"),
    ("higgsfield ai", "Higgsfield AI"),
    ("recraft.ai", "Recraft"),
    ("topaz labs image api", "Topaz Labs"),
    ("tiktok ad creative toolbox", "TikTok Ad Creative Toolbox"),
    ("fastvid", "FastVid"),
    ("ks-c2pa-service", "Kuaishou Kling AI"),
    ("pruna ai", "Pruna AI"),
    ("capcut", "CapCut (C2PA signer)"),
];

/// `C2PA_AI_TOOLS`: tool token to display label.
pub const C2PA_AI_TOOLS: &[(&[u8], &str)] = &[
    (b"GPT-4o", "GPT-4o"),
    (b"ChatGPT", "ChatGPT"),
    (b"Sora", "Sora"),
    (b"DALL-E", "DALL·E"),
    (b"DALL", "DALL·E"),
    (b"Imagen", "Imagen"),
    (b"Firefly", "Firefly"),
    (b"Dreamina", "Dreamina"),
    (b"Pippit", "Pippit"),
    (b"Dola", "Dola"),
    (b"TIKTOK AD Creative Toolbox", "TikTok Ad Creative Toolbox"),
];

/// `C2PA_ACTIONS`: `c2pa.<action>` label to its short name.
pub const C2PA_ACTIONS: &[(&[u8], &str)] = &[
    (b"c2pa.created", "created"),
    (b"c2pa.converted", "converted"),
    (b"c2pa.edited", "edited"),
    (b"c2pa.filtered", "filtered"),
    (b"c2pa.cropped", "cropped"),
    (b"c2pa.resized", "resized"),
    (b"c2pa.opened", "opened"),
    (b"c2pa.placed", "placed"),
    (b"c2pa.watermarked.unbound", "watermarked.unbound"),
];

/// Lower-case generator names that mark a metadata value as AI-generated.
pub const AI_GENERATOR_TOKENS: &[&str] = &[
    "firefly",
    "dall-e",
    "dalle",
    "midjourney",
    "stable diffusion",
    "stable-diffusion",
    "stablediffusion",
    "comfyui",
    "automatic1111",
    "invokeai",
    "imagen",
    "gpt-image",
    "gpt image",
    "nano banana",
    "nightcafe",
    "ideogram",
    "leonardo",
    "flux",
    "dreamstudio",
    "novelai",
    "reve.com",
    "luma ai",
    "lumalabs",
    "aphrodite ai",
    "apple photos clean up",
    "apple photos generative edit",
    "apple image playground",
    "fal-ai",
];

/// The short name of an action label, or the label without its `c2pa.` prefix.
pub fn action_name(action: &str) -> String {
    C2PA_ACTIONS
        .iter()
        .find(|(label, _)| *label == action.as_bytes())
        .map_or_else(
            || action.strip_prefix("c2pa.").unwrap_or(action).to_owned(),
            |(_, name)| (*name).to_owned(),
        )
}
