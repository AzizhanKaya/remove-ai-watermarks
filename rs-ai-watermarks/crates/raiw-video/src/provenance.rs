//! Container-provenance rules over AI-metadata markers: AI claims, platform attribution, invisible
//! watermark hints, recipes, and the per-mark confirm/veto predicates (from `video.py` and the
//! provenance half of `video_visible.py`).

use raiw_c2pa::registry::C2PA_AI_VENDORS;
use raiw_c2pa::soft_binding::{self, REGISTRY, SoftBindingKind};
use raiw_c2pa::text::casefold;
use raiw_c2pa::{c2pa_vendor_platform, claim_generator_platform};
use raiw_core::tc260::{producer_codes_for_mark, producer_for_code, tc260_producer_in};
use raiw_metadata::AiMetadata;

/// The provider marks in specificity order (`VIDEO_VISIBLE_MARKS`).
pub const VIDEO_VISIBLE_MARKS: [&str; 23] = [
    "sora",
    "veo",
    "dreamina",
    "jimeng",
    "jimeng_pill",
    "seedance",
    "doubao",
    "dola",
    "hailuo",
    "vidu",
    "wan",
    "qwen",
    "yuanbao",
    "baidu",
    "runninghub",
    "microsoft",
    "pixverse",
    "luma",
    "heygen",
    "sync_labs",
    "tiktok_ai",
    "generic_ai_label",
    "kling",
];

fn get<'a>(markers: &'a AiMetadata, key: &str) -> &'a str {
    markers.get(key).map_or("", String::as_str)
}

/// `_video_markers_claim_ai`: the image path's C2PA rule.
pub fn markers_claim_ai(markers: &AiMetadata) -> bool {
    raiw_c2pa::markers_claim_ai(markers.iter().map(|(k, v)| (k.as_str(), v.as_str())))
}

/// `c2pa_credential_level` over marker strings: `invalid`, `verified` or `unverified`.
pub fn c2pa_credential_level(markers: &AiMetadata) -> &'static str {
    let invalid = |key: &str| get(markers, key) == "invalid";
    if invalid("c2pa_integrity") || invalid("c2pa_signature") || invalid("c2pa_signer_validity") {
        "invalid"
    } else if get(markers, "c2pa_integrity") == "valid" && get(markers, "c2pa_signature") == "valid" {
        "verified"
    } else {
        "unverified"
    }
}

fn nonempty(text: &str) -> Option<&str> {
    (!text.is_empty()).then_some(text)
}

/// `C2PA_SIGNER_PLATFORM_BY_ORG` in dict order (first position, last value per org).
fn signer_platforms_by_org() -> Vec<(&'static str, &'static str)> {
    let mut out: Vec<(&str, &str)> = Vec::new();
    for vendor in C2PA_AI_VENDORS {
        let Some(platform) = vendor.signer_platform else {
            continue;
        };
        if let Some(entry) = out.iter_mut().find(|(org, _)| *org == vendor.org) {
            entry.1 = platform;
        } else {
            out.push((vendor.org, platform));
        }
    }
    out
}

const LOCAL_GENERATION_METADATA_KEYS: [&str; 3] = ["parameters", "workflow", "prompt"];
const TOPAZ_VIDEO_METADATA_KEYS: [&str; 8] = [
    "topazcloud",
    "topazcloud_provider",
    "topazcloud_application",
    "topazcloud_timestamp",
    "videoai",
    "videoai_provider",
    "videoai_application",
    "videoai_timestamp",
];
const TOPAZ_AI_DESCRIPTION: &str = "AI-enhanced using Topaz Labs";

/// `_tc260_video_platform`: a registered producer's video platform label.
pub fn tc260_video_platform(producer: &str) -> Option<&'static str> {
    producer_for_code(producer).and_then(|row| row.video_platform)
}

/// `_platform_from_video_metadata`: the generating platform named by the markers.
pub fn platform_from_video_metadata(markers: &AiMetadata) -> Option<String> {
    let generator = get(markers, "claim_generator");
    let ai_tool = get(markers, "ai_tool");
    let joined: Vec<&str> = markers.values().map(String::as_str).collect();
    let joined = joined.join("\n");
    if let Some(platform) = claim_generator_platform(nonempty(generator))
        .or_else(|| c2pa_vendor_platform(generator))
        .or_else(|| claim_generator_platform(nonempty(ai_tool)))
        .or_else(|| c2pa_vendor_platform(ai_tool))
        .or_else(|| c2pa_vendor_platform(get(markers, "app_provenance")))
        .or_else(|| c2pa_vendor_platform(&joined))
    {
        return Some(platform.to_owned());
    }
    let issuer = get(markers, "issuer");
    if let Some((_, platform)) = signer_platforms_by_org()
        .into_iter()
        .find(|(org, _)| issuer.contains(org))
    {
        return Some(platform.to_owned());
    }
    if markers.contains_key("aigc_label") {
        return Some(
            tc260_video_platform(get(markers, "aigc_producer"))
                .unwrap_or("China AIGC-labeled content (TC260 standard)")
                .to_owned(),
        );
    }
    let app_aigc = get(markers, "app_aigc");
    if let Some(inner) = app_aigc
        .strip_prefix("App AIGC disclosure (")
        .and_then(|s| s.strip_suffix(')'))
    {
        return Some(inner.to_owned());
    }
    let app_provenance = get(markers, "app_provenance");
    if let Some(inner) = app_provenance
        .strip_prefix("App export provenance (")
        .and_then(|s| s.strip_suffix(')'))
    {
        return Some(inner.to_owned());
    }
    if markers.contains_key("kling_sei") {
        return tc260_video_platform("kling").map(str::to_owned);
    }
    if markers.contains_key("xai_signature") {
        return Some("xAI Grok Imagine".to_owned());
    }
    if markers
        .keys()
        .any(|k| TOPAZ_VIDEO_METADATA_KEYS.contains(&k.to_lowercase().as_str()))
        || get(markers, "description").trim() == TOPAZ_AI_DESCRIPTION
    {
        return Some("Topaz Labs".to_owned());
    }
    if markers
        .keys()
        .any(|k| LOCAL_GENERATION_METADATA_KEYS.contains(&k.to_lowercase().as_str()))
    {
        return Some("Stable Diffusion / local pipeline (Automatic1111, ComfyUI, InvokeAI)".to_owned());
    }
    let audio = get(markers, "audio_generator").trim();
    if !audio.is_empty() {
        return Some(
            claim_generator_platform(Some(audio))
                .or_else(|| c2pa_vendor_platform(audio))
                .map_or_else(|| audio.to_owned(), str::to_owned),
        );
    }
    None
}

/// `_video_invisible_watermark_hints`: declared schemes and provider-policy hints, sorted.
pub fn invisible_watermark_hints(markers: &AiMetadata) -> Vec<String> {
    let entries = soft_binding::registry_entries_in(get(markers, "soft_binding_algorithm").as_bytes());
    let mut hints: Vec<String> = soft_binding::labels(entries, Some(SoftBindingKind::Watermark));
    if let Some(declared) = markers.get("soft_binding") {
        let declared = format!(", {declared}, ");
        let watermark = soft_binding::labels(REGISTRY.iter(), Some(SoftBindingKind::Watermark));
        let fingerprint = soft_binding::labels(REGISTRY.iter(), Some(SoftBindingKind::Fingerprint));
        for label in watermark {
            if !fingerprint.contains(&label) && declared.contains(&format!(", {label}, ")) {
                hints.push(label);
            }
        }
    }
    if markers.contains_key("synthid_watermark") {
        hints.push("SynthID".to_owned());
    }
    if markers_claim_ai(markers)
        && platform_from_video_metadata(markers).as_deref() == Some("Amazon Bedrock (Nova)")
    {
        hints.push("Amazon Nova Reel".to_owned());
    }
    hints.sort();
    hints.dedup();
    hints
}

const BYTEDANCE_FALLBACK_TOKENS: [&str; 10] = [
    "bytedance",
    "byteplus",
    "seedance",
    "dreamina",
    "capcut",
    "doubao",
    "jimeng",
    "dola",
    "pippit",
    "volcengine",
];
const BYTEDANCE_FALLBACK_SPRING: &str = "spring (sg)";
const BYTEDANCE_VISIBLE_MARKS: [&str; 6] =
    ["dreamina", "jimeng", "jimeng_pill", "seedance", "doubao", "dola"];

fn has_bytedance_token(text: &str) -> bool {
    BYTEDANCE_FALLBACK_TOKENS.iter().any(|t| text.contains(t)) || text.contains(BYTEDANCE_FALLBACK_SPRING)
}

/// `_is_bytedance_video_provenance`.
pub fn is_bytedance_video_provenance(markers: &AiMetadata, visible_mark: Option<&str>) -> bool {
    if visible_mark.is_some_and(|m| BYTEDANCE_VISIBLE_MARKS.contains(&m)) {
        return true;
    }
    if markers.is_empty() {
        return false;
    }
    if has_bytedance_video_provenance(markers) {
        return true;
    }
    if producer_for_code(get(markers, "aigc_producer")).is_some_and(|row| row.vendor == "ByteDance") {
        return true;
    }
    if has_bytedance_token(
        &platform_from_video_metadata(markers)
            .unwrap_or_default()
            .to_lowercase(),
    ) {
        return true;
    }
    let haystack: Vec<String> = markers.iter().map(|(k, v)| format!("{k} {v}")).collect();
    has_bytedance_token(&haystack.join(" ").to_lowercase())
}

/// `_has_capcut_pixel_watermark_provenance`.
pub fn has_capcut_pixel_watermark_provenance(markers: &AiMetadata, visible_mark: Option<&str>) -> bool {
    if visible_mark == Some("seedance") {
        return true;
    }
    if markers.is_empty() {
        return false;
    }
    if casefold(get(markers, "app_aigc")).contains("capcut") {
        return true;
    }
    let identity = casefold(
        &[
            get(markers, "claim_generator"),
            get(markers, "issuer"),
            get(markers, "ai_tool"),
        ]
        .join(" "),
    );
    identity.contains("capcut") && markers_claim_ai(markers)
}

/// `video_invisible_target_cohort`: which verified pixel-domain watermark family applies.
pub fn video_invisible_target_cohort(
    markers: &AiMetadata,
    visible_mark: Option<&str>,
    explicit_vendor: Option<&str>,
) -> Option<&'static str> {
    match explicit_vendor {
        Some("google" | "openai") => return Some("synthid"),
        Some("bytedance") => return Some("seedance"),
        Some("meta") => return Some("meta"),
        _ => {}
    }
    if visible_mark == Some("veo") {
        return Some("synthid");
    }
    if has_capcut_pixel_watermark_provenance(markers, visible_mark) {
        return Some("seedance");
    }
    if !markers.is_empty() {
        if markers.contains_key("synthid_watermark") {
            return Some("synthid");
        }
        if !invisible_watermark_hints(markers).is_empty() {
            return Some("declared_watermark");
        }
    }
    None
}

/// `video_has_invisible_provenance`.
pub fn video_has_invisible_provenance(
    markers: &AiMetadata,
    visible_mark: Option<&str>,
    explicit_vendor: Option<&str>,
) -> bool {
    video_invisible_target_cohort(markers, visible_mark, explicit_vendor).is_some()
}

// Per-mark vendor predicates (`has_*_provenance` in `video_visible.py`).

fn producer_is(markers: &AiMetadata, mark: &str) -> bool {
    tc260_producer_in(
        get(markers, "aigc_producer"),
        producer_codes_for_mark(mark).iter().copied(),
    )
}

pub fn has_sora_provenance(markers: &AiMetadata) -> bool {
    get(markers, "claim_generator").to_lowercase().contains("sora")
}

pub fn has_veo_provenance(markers: &AiMetadata) -> bool {
    let identity = format!("{} {}", get(markers, "claim_generator"), get(markers, "issuer")).to_lowercase();
    identity.contains("google")
        && get(markers, "source_type")
            .to_lowercase()
            .contains("trainedalgorithmicmedia")
}

pub fn has_bytedance_video_provenance(markers: &AiMetadata) -> bool {
    let identity = [
        get(markers, "claim_generator"),
        get(markers, "ai_tool"),
        get(markers, "issuer"),
    ]
    .join(" ")
    .to_lowercase();
    ["bytedance", "byteplus", "spring (sg)", "dola"]
        .iter()
        .any(|t| identity.contains(t))
        && get(markers, "source_type")
            .to_lowercase()
            .contains("trainedalgorithmicmedia")
}

pub fn has_dreamina_video_provenance(markers: &AiMetadata) -> bool {
    has_bytedance_video_provenance(markers)
        || producer_is(markers, "dreamina")
        || casefold(get(markers, "claim_generator")).contains("dreamina")
}

fn has_valid_c2pa_video_provenance(
    markers: &AiMetadata,
    issuer_tokens: &[&str],
    identity_tokens: &[&str],
    generator_claim_tokens: &[&str],
) -> bool {
    if c2pa_credential_level(markers) == "invalid" {
        return false;
    }
    let issuer = casefold(get(markers, "issuer"));
    let generator = casefold(get(markers, "claim_generator"));
    let identity = format!("{issuer} {generator}");
    let source_type = casefold(get(markers, "source_type"));
    let vendor = issuer_tokens.iter().any(|t| issuer.contains(t))
        || identity_tokens.iter().any(|t| identity.contains(t));
    let ai = source_type.contains("trainedalgorithmicmedia")
        || generator_claim_tokens.iter().any(|t| generator.contains(t));
    vendor && ai
}

/// `confirms_video_provenance`: provenance names the visible mark's own vendor.
pub fn confirms_video_provenance(mark: &str, markers: &AiMetadata) -> bool {
    match mark {
        "sora" => has_sora_provenance(markers),
        "veo" => has_veo_provenance(markers),
        "dreamina" | "jimeng" | "jimeng_pill" => has_dreamina_video_provenance(markers),
        "seedance" | "dola" => has_bytedance_video_provenance(markers),
        "doubao" | "hailuo" | "vidu" | "wan" | "qwen" | "yuanbao" | "baidu" | "runninghub" => {
            producer_is(markers, mark)
        }
        "microsoft" => has_valid_c2pa_video_provenance(markers, &["microsoft"], &[], &[]),
        "pixverse" => has_valid_c2pa_video_provenance(markers, &["pixverse"], &[], &[]),
        "sync_labs" => has_valid_c2pa_video_provenance(markers, &[], &["sync labs", "synchronicity"], &[]),
        "tiktok_ai" => {
            has_valid_c2pa_video_provenance(markers, &[], &["tiktok"], &["tiktok ad creative toolbox"])
        }
        _ => false,
    }
}

fn issuer_tokens(mark: &str) -> &'static [&'static str] {
    match mark {
        "sora" => &["openai"],
        "veo" => &["google"],
        "dreamina" | "jimeng" | "jimeng_pill" => &["bytedance", "byteplus", "spring (sg)", "dreamina"],
        "seedance" | "doubao" | "dola" => &["bytedance", "byteplus", "spring (sg)"],
        "hailuo" => &["minimax"],
        "vidu" => &["shengshu", "vidu"],
        "wan" => &["alibaba", "tongyi", "wan"],
        "qwen" => &["alibaba", "tongyi", "qwen"],
        "yuanbao" => &["tencent", "yuanbao", "hunyuan"],
        "baidu" => &["baidu", "ernie"],
        "runninghub" => &["runninghub"],
        "microsoft" => &["microsoft"],
        "pixverse" => &["pixverse"],
        "luma" => &["luma"],
        "heygen" => &["heygen"],
        "sync_labs" => &["sync labs", "synchronicity"],
        "tiktok_ai" => &["tiktok", "bytedance"],
        "kling" => &["kling", "kuaishou"],
        _ => &[],
    }
}

/// `contradicts_video_provenance`: provenance names a producer other than the mark's vendor.
pub fn contradicts_video_provenance(mark: &str, markers: &AiMetadata) -> bool {
    if mark == "generic_ai_label" || confirms_video_provenance(mark, markers) {
        return false;
    }
    let producer = get(markers, "aigc_producer").trim();
    if !producer.is_empty() {
        let Some(row) = producer_for_code(producer) else {
            return true;
        };
        if matches!(
            mark,
            "dreamina" | "jimeng" | "jimeng_pill" | "seedance" | "dola" | "tiktok_ai"
        ) {
            return row.vendor != "ByteDance";
        }
        return row.image_mark != Some(mark) && row.video_mark != Some(mark);
    }
    let issuer = casefold(get(markers, "issuer"));
    let claims_ai = casefold(get(markers, "source_type")).contains("trainedalgorithmicmedia");
    !issuer.is_empty()
        && claims_ai
        && c2pa_credential_level(markers) != "invalid"
        && !issuer_tokens(mark).iter().any(|t| issuer.contains(t))
}

/// `_non_contradicted_marks`.
pub fn non_contradicted_marks<'a>(candidates: &[&'a str], markers: &AiMetadata) -> Vec<&'a str> {
    if markers.is_empty() {
        return candidates.to_vec();
    }
    candidates
        .iter()
        .copied()
        .filter(|m| !contradicts_video_provenance(m, markers))
        .collect()
}

/// `_VISIBLE_PLATFORM`: the platform each visible mark names.
pub fn visible_platform(mark: &str) -> Option<&'static str> {
    let fixed = match mark {
        "sora" => Some("OpenAI Sora"),
        "veo" => Some("Google Veo"),
        "jimeng" | "jimeng_pill" => Some("ByteDance Jimeng"),
        "seedance" => Some("ByteDance Seedance"),
        "dola" => Some("ByteDance Dola"),
        "runninghub" => Some("RunningHub"),
        "microsoft" => Some("Microsoft"),
        "pixverse" => Some("PixVerse"),
        "luma" => Some("Luma AI"),
        "heygen" => Some("HeyGen"),
        "sync_labs" => Some("Sync Labs"),
        "tiktok_ai" => Some("TikTok"),
        "generic_ai_label" => Some("AI-generated content (GB 45438-2025 label)"),
        _ => None,
    };
    let tc260 = raiw_core::tc260::TC260_PRODUCERS
        .iter()
        .filter(|row| row.video_mark == Some(mark))
        .filter_map(|row| row.video_platform)
        .next_back();
    tc260.or(fixed)
}

/// `_VIDEO_MARK_VENDOR_GROUP`.
pub fn vendor_group(mark: &str) -> Option<&'static str> {
    Some(match mark {
        "sora" => "openai",
        "veo" => "google",
        "dreamina" | "jimeng" | "jimeng_pill" | "seedance" | "doubao" | "dola" | "tiktok_ai" => "bytedance",
        "hailuo" => "minimax",
        "vidu" => "shengshu",
        "wan" | "qwen" => "alibaba",
        "yuanbao" => "tencent",
        "baidu" => "baidu",
        "runninghub" => "runninghub",
        "microsoft" => "microsoft",
        "pixverse" => "pixverse",
        "luma" => "luma",
        "heygen" => "heygen",
        "sync_labs" => "sync_labs",
        "generic_ai_label" => "generic_tc260",
        "kling" => "kuaishou",
        _ => return None,
    })
}
