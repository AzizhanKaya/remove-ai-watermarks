//! Pinned artifact identity, schema names, dimensions, and the fixed graphics-veto head.

pub const PROVIDERS: [&str; 10] = [
    "openai",
    "google",
    "bytedance",
    "flux",
    "stable_diffusion",
    "meta",
    "midjourney",
    "ideogram",
    "firefly",
    "xai",
];
pub const N_CLS: usize = PROVIDERS.len();
pub const UNKNOWN: &str = "unknown";

pub const WEIGHTS_ENV: &str = "RAIW_SOURCE_CLASSIFY_WEIGHTS";
pub const WEIGHTS_REPO: &str = "wiltodelta/raiw-source-classify";
pub const WEIGHTS_REVISION: &str = "5f3d7294cad4350d5511e3fb1c1f8eacb8033f59";
pub const GITHUB_WEIGHTS_REPO: &str = "wiltodelta/raiw-source-classify";
pub const GITHUB_WEIGHTS_TAG: &str = "v10";
pub const CONFIG_FILE: &str = "config.json";
pub const MODEL_FILE: &str = "multi-provider-source-v10.npz";
pub const MODEL_SHA256: &str = "b3eb961cc68cf283b399bebf11cdf61b2bf41b98fabca0579f8b61d9141f695f";
pub const SOURCE_CLASSIFY_EXTRA: &str = "'remove-ai-watermarks[source-classify]'";

pub const MODEL_SCHEMA_V3: &str = "multi-provider-source-v3";
pub const MODEL_SCHEMA_V4: &str = "multi-provider-source-v4";
pub const MODEL_SCHEMA_V5: &str = "multi-provider-source-v5";
pub const MODEL_SCHEMA_V7: &str = "multi-provider-source-v7";
pub const MODEL_SCHEMA_V8: &str = "multi-provider-source-v8";
pub const MODEL_SCHEMA_V9: &str = "multi-provider-source-v9";
pub const MODEL_SCHEMA_V10: &str = "multi-provider-source-v10";
pub const MODEL_SCHEMA: &str = MODEL_SCHEMA_V10;

pub const MIN_SIDE: usize = 256;
pub const FORENSIC_DIM: usize = 124;
pub const SPECTRAL_GRID: usize = 16;
pub const SPECTRAL_DIM: usize = 3 * SPECTRAL_GRID * SPECTRAL_GRID;
pub const BASE_FEATURE_DIM: usize = FORENSIC_DIM + SPECTRAL_DIM;
pub const EXTRA_FEATURE_DIM: usize = 64;
pub const FEATURE_DIM: usize = BASE_FEATURE_DIM + EXTRA_FEATURE_DIM;
pub const FEATURE_DIM_V4: usize = 512;
pub const REGIME_START_V4: usize = 480;
pub const REGIME_DIM_V4: usize = 32;

pub const VIT_SHORT_EDGE_V7: usize = 440;
pub const VIT_DETAIL_SHORT_EDGE_V7: usize = 672;
pub const VIT_CROP_SIZE_V7: usize = 384;
pub const VIT_OUT_DIM_V7: usize = 28;
pub const SYN_IN_DIM_V7: usize = 67;
pub const MM_PROV_IN_DIM_V7: usize = 82;
pub const VIT_OUT_DIM_V8: usize = 59;
pub const SYN_IN_DIM_V8: usize = 106;
pub const MM_PROV_IN_DIM_V8: usize = 195;

pub const CLAIM: &str = "Identifies the likely AI generation/export pixel pipeline across 10 provider \
families with decoupled regime-aware acceptance, threshold-invariant attribution, \
and stacked Stage-2 version attribution. This is not provenance \
verification and does not detect or verify SynthID or any invisible watermark.";

/// Named `[lo, hi)` slices of a routed-tile ViT output vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VitSlices {
    pub orig_cf: Option<(usize, usize)>,
    pub teacher_syn: (usize, usize),
    pub prov_logits: (usize, usize),
    pub gfx_logit: (usize, usize),
    pub dom_syn_logit: (usize, usize),
    pub arc_emb: (usize, usize),
    pub proto_sim: (usize, usize),
    pub sp_stats: (usize, usize),
}

/// Schema, patch-grid geometry, and named slices for routed-tile ViT models (`v9`, `v10`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoutedSpec {
    pub schema: &'static str,
    pub bundle_tag: &'static str,
    pub img_size: usize,
    pub patch_size: usize,
    pub grid: usize,
    pub min_dist_sq: i64,
    pub vit_out_dim: usize,
    pub syn_in_dim: usize,
    pub mm_prov_in_dim: usize,
    pub local_in_dim: usize,
    pub layout: &'static [(&'static str, usize, usize)],
    pub slices: VitSlices,
}

pub const VIT_LAYOUT_V9: &[(&str, usize, usize)] = &[
    ("teacher_syn", 0, 3),
    ("prov_logits", 3, 15),
    ("gfx_logit", 15, 16),
    ("dom_syn_logit", 16, 17),
    ("arc_emb", 17, 41),
    ("proto_sim", 41, 51),
    ("sp_stats", 51, 64),
];

pub const VIT_LAYOUT_V10: &[(&str, usize, usize)] = &[
    ("orig_cf", 0, 2),
    ("teacher_syn", 2, 5),
    ("prov_logits", 5, 17),
    ("gfx_logit", 17, 18),
    ("dom_syn_logit", 18, 19),
    ("arc_emb", 19, 43),
    ("proto_sim", 43, 53),
    ("sp_stats", 53, 66),
];

pub const ROUTED_SPEC_V9: RoutedSpec = RoutedSpec {
    schema: MODEL_SCHEMA_V9,
    bundle_tag: "v9",
    img_size: 392,
    patch_size: 14,
    grid: 28,
    min_dist_sq: 81,
    vit_out_dim: 64,
    syn_in_dim: 113,
    mm_prov_in_dim: 205,
    local_in_dim: 36,
    layout: VIT_LAYOUT_V9,
    slices: VitSlices {
        orig_cf: None,
        teacher_syn: (0, 3),
        prov_logits: (3, 15),
        gfx_logit: (15, 16),
        dom_syn_logit: (16, 17),
        arc_emb: (17, 41),
        proto_sim: (41, 51),
        sp_stats: (51, 64),
    },
};

pub const ROUTED_SPEC_V10: RoutedSpec = RoutedSpec {
    schema: MODEL_SCHEMA_V10,
    bundle_tag: "v10",
    img_size: 384,
    patch_size: 16,
    grid: 24,
    min_dist_sq: 64,
    vit_out_dim: 66,
    syn_in_dim: 117,
    mm_prov_in_dim: 209,
    local_in_dim: 39,
    layout: VIT_LAYOUT_V10,
    slices: VitSlices {
        orig_cf: Some((0, 2)),
        teacher_syn: (2, 5),
        prov_logits: (5, 17),
        gfx_logit: (17, 18),
        dom_syn_logit: (18, 19),
        arc_emb: (19, 43),
        proto_sim: (43, 53),
        sp_stats: (53, 66),
    },
};

pub const IMAGENET_MEAN_RGB: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD_RGB: [f32; 3] = [0.229, 0.224, 0.225];

// Born-digital graphics veto over x_512[464:480] + x_512[502:512].
pub const GFX_MEAN_26: [f64; 26] = [
    -0.00866062,
    -0.00369257,
    -0.03726238,
    -0.0227263,
    0.02860176,
    0.32059201,
    0.19851867,
    0.66889976,
    0.1629835,
    0.01311125,
    0.05012025,
    0.40246035,
    0.27661094,
    0.84704429,
    0.46758068,
    0.14369827,
    0.14918268,
    0.40246035,
    0.0852835,
    0.46758068,
    0.32059201,
    0.66889976,
    0.01311125,
    0.05012025,
    0.48955523,
    0.45048192,
];
pub const GFX_SCALE_26: [f64; 26] = [
    0.1038899, 0.09136275, 0.08558824, 0.07392866, 0.13310937, 0.18244784, 0.07912959, 0.25684499,
    0.24843859, 0.06171596, 0.16283455, 0.2023694, 0.25040855, 0.17989174, 0.23715294, 0.10365412,
    0.21141834, 0.2023694, 0.19076729, 0.23715294, 0.18244784, 0.25684499, 0.06171596, 0.16283455,
    0.19453504, 0.12261727,
];
pub const GFX_COEF_26: [f64; 26] = [
    0.28593333,
    -0.01833078,
    -0.54306846,
    0.23017136,
    -0.33946035,
    -0.20786306,
    0.41448862,
    -0.10714731,
    -0.19169794,
    0.09411158,
    0.19919772,
    0.22211316,
    -0.4822873,
    -0.57568771,
    0.56877772,
    -0.08537862,
    -0.67524892,
    0.22211316,
    -0.11560082,
    0.56877772,
    -0.20786306,
    -0.10714731,
    0.09411158,
    0.19919772,
    0.42927782,
    -0.20077164,
];
pub const GFX_INTERCEPT: f64 = -4.46995339;

pub fn provider_index(name: &str) -> Option<usize> {
    PROVIDERS.iter().position(|p| *p == name)
}
