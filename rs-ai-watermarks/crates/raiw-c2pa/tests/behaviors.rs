//! Behavioral assertions ported from the Python suite (`test_metadata_internals.py`,
//! `test_metadata.py`, `test_identify.py`, `test_c2pa_registry_sync.py`, `test_designer_synthid_suppression.py`,
//! `test_metadata_audit_regressions.py`, `test_security_clamp.py`).

use raiw_c2pa::soft_binding::{self, SoftBindingKind};
use raiw_c2pa::{
    AiSourceKind, C2paInfo, Check, SignerTrust, SignerValidity, attribution, fallback, png, registry, store,
};
use raiw_core::testing::fixture;
use serde_json::{Value, json};

const DST: &str = "http://cv.iptc.org/newscodes/digitalsourcetype/";

fn info(store_value: &Value) -> C2paInfo {
    store::info_from_manifest_value(store_value).expect("normalized store")
}

fn parsed(payload: &[u8]) -> C2paInfo {
    fallback::parse_c2pa_chunk(payload)
}

#[test]
fn cbor_text_heads() {
    assert_eq!(
        fallback::cbor_text_after(b"name\x63abc", b"name").as_deref(),
        Some("abc")
    );
    let long = [&b"name\x78\x1e"[..], &[b'x'; 30]].concat();
    assert_eq!(fallback::cbor_text_after(&long, b"name"), Some("x".repeat(30)));
    let longer = [&b"name\x79"[..], &300u16.to_be_bytes(), &[b'y'; 300]].concat();
    assert_eq!(fallback::cbor_text_after(&longer, b"name"), Some("y".repeat(300)));
    assert_eq!(fallback::cbor_text_after(b"nothing here", b"name"), None);
    assert_eq!(fallback::cbor_text_after(b"prefixname", b"name"), None);
    assert_eq!(fallback::cbor_text_after(b"name\x00abc", b"name"), None);
    assert!(fallback::cbor_text_after(b"name\x61\xff", b"name").is_some());
}

#[test]
fn raw_chunk_guards_and_source_types() {
    let gpt = parsed(b"...name\x69gpt-imageOpenAI trainedAlgorithmicMedia c2pa.watermarked.unbound");
    assert_eq!(gpt.claim_generator.as_deref(), Some("gpt-image"));
    assert!(gpt.issuer.as_deref().unwrap().contains("OpenAI"));
    assert!(gpt.synthid_watermark.is_some());
    let unprintable = parsed(b"...name\x64\x81abcOpenAI trainedAlgorithmicMedia");
    assert_eq!(unprintable.claim_generator, None);
    let procedural = parsed(b"...name\x69some-tool OpenAI algorithmicMedia");
    assert_eq!(
        (procedural.source_type.as_deref(), procedural.synthid_watermark),
        (Some("algorithmicMedia"), None)
    );
    let composite =
        parsed(b"...name\x69some-tool OpenAI compositeWithTrainedAlgorithmicMedia c2pa.watermarked.unbound");
    assert!(
        composite
            .source_type
            .as_deref()
            .unwrap()
            .contains("compositeWithTrainedAlgorithmicMedia")
    );
    assert!(composite.synthid_watermark.is_some());
    assert_eq!(
        parsed(b"x compositeWithTrainedAlgorithmicMedia x algorithmicMedia x").ai_source_kind,
        Some(AiSourceKind::Enhanced)
    );
    let iscc = parsed(b"c2pa Google trainedAlgorithmicMedia io.iscc.v0");
    assert_eq!(
        (iscc.soft_binding.as_deref(), iscc.synthid_vendors.as_slice()),
        (Some("ISCC (content code)"), ["Google LLC".to_owned()].as_slice())
    );
    assert_eq!(
        parsed(b"jumb c2pa c2pa.ai-disclosure modelType human_validated").ai_source_kind,
        Some(AiSourceKind::Enhanced)
    );
    assert_eq!(
        attribution::synthid_verdict("OpenAI"),
        "present according to OpenAI provenance"
    );
}

#[test]
fn synthid_evidence_rules() {
    let vendors = |buffer: &[u8]| attribution::synthid_evidence_vendors_in(buffer, None, None);
    assert!(vendors(b"c2pa OpenAI trainedAlgorithmicMedia").is_empty());
    assert_eq!(
        vendors(b"c2pa OpenAI trainedAlgorithmicMedia c2pa.watermarked.unbound"),
        ["OpenAI"]
    );
    assert_eq!(vendors(b"c2pa Google trainedAlgorithmicMedia"), ["Google LLC"]);
    let edit = b"c2pa Google LLC Google Photos c2pa.deleted compositeWithTrainedAlgorithmicMedia".to_vec();
    assert_eq!(vendors(&edit), ["Google LLC"]);
    assert_eq!(
        vendors(&[&edit[..], b" OpenAI c2pa.watermarked.unbound"].concat()),
        ["Google LLC", "OpenAI"]
    );
    let upload =
        b"c2pa Google LLC YouTube Video Processing Services c2pa.transcoded trainedAlgorithmicMedia".to_vec();
    assert!(vendors(&upload).is_empty());
    assert_eq!(
        vendors(&[&upload[..], b" Added imperceptible SynthID watermark"].concat()),
        ["Google LLC"]
    );
}

#[test]
fn ai_disclosure_kinds_agree_between_structured_and_raw_paths() {
    for levels in [
        &["human_validated"][..],
        &["fully_autonomous"],
        &["prompt_guided"],
        &["human_validated", "prompt_guided"],
    ] {
        let assertions: Vec<Value> = levels
            .iter()
            .map(|level| json!({"label": "c2pa.ai-disclosure", "data": {"modelType": "diffusion", "contentProfile": {"humanOversightLevel": level}}}))
            .collect();
        let structured =
            info(&json!({"active_manifest": "a", "manifests": {"a": {"assertions": assertions}}}));
        let raw = parsed(format!("jumb c2pa c2pa.ai-disclosure {}", levels.join(" ")).as_bytes());
        assert_eq!(structured.ai_source_kind, raw.ai_source_kind, "{levels:?}");
        let expected = if levels == ["human_validated"] {
            AiSourceKind::Enhanced
        } else {
            AiSourceKind::Generated
        };
        assert_eq!(structured.ai_source_kind, Some(expected));
    }
}

#[test]
fn soft_binding_labels_and_token_boundaries() {
    let vendors = |buffer: &[u8]| soft_binding::vendors_in(buffer);
    assert_eq!(
        vendors(b"...alg:\"com.adobe.trustmark.P\"..."),
        ["Adobe TrustMark"]
    );
    assert_eq!(vendors(b"com.digimarc.validate.1"), ["Digimarc Validate"]);
    assert_eq!(vendors(b"ai.steg.api blah"), ["Steg.AI"]);
    assert_eq!(vendors(b"ai.trufo.pawprint.watermark"), ["Trufo PawPrint"]);
    assert_eq!(vendors(b"com.aiwatermark.pixelseal.1"), ["AIWatermark PixelSeal"]);
    assert_eq!(vendors(b"io.iscc.v0"), ["ISCC (content code)"]);
    assert_eq!(
        vendors(b"com.adobe.hiermark.A"),
        ["Adobe Hierarchical Audio watermark (HierMark) variant A"]
    );
    assert_eq!(
        vendors(b"com.adobe.flowmark.A"),
        ["Adobe FlowMark video watermarking variant A"]
    );
    assert!(vendors(b"no soft binding here").is_empty());
    assert!(vendors(b"com.adobe.trustmark.P2").is_empty());
    assert!(vendors(b"").is_empty());
    assert!(vendors(b"alg: com.suno.fingerprint.v4").contains(&"Suno Audio Fingerprint".to_owned()));
    for entry in soft_binding::REGISTRY {
        let token = entry.algorithm.as_bytes();
        let names = |buffer: &[u8]| {
            soft_binding::registry_entries_in(buffer)
                .iter()
                .any(|e| e.algorithm == entry.algorithm)
        };
        assert!(names(&[b"\"", token, b"\""].concat()), "{}", entry.algorithm);
        for wrapped in [
            [b"prefix", token].concat(),
            [token, b".suffix"].concat(),
            [b"-", token].concat(),
            [token, b"1"].concat(),
        ] {
            assert!(!names(&wrapped), "{}", String::from_utf8_lossy(&wrapped));
        }
    }
}

#[test]
fn packaged_registry_covers_the_official_baseline() {
    let registry = soft_binding::REGISTRY;
    assert!(registry.len() >= 53);
    assert!(
        registry
            .windows(2)
            .all(|pair| pair[0].identifier < pair[1].identifier)
    );
    assert!(
        registry
            .iter()
            .filter(|e| e.date_entered.starts_with("2026-"))
            .count()
            >= 28
    );
    let kinds: std::collections::HashSet<_> = registry.iter().map(|e| e.kind).collect();
    assert_eq!(
        kinds,
        [SoftBindingKind::Watermark, SoftBindingKind::Fingerprint]
            .into_iter()
            .collect()
    );
    for algorithm in ["com.adobe.hiermark.A", "com.adobe.flowmark.A"] {
        assert!(registry.iter().any(|e| e.algorithm == algorithm));
    }
    assert!(soft_binding::SOURCE_URL.ends_with("/softbinding-algorithm-list.json"));
    assert!(soft_binding::sync::is_commit_sha(soft_binding::SOURCE_REVISION));
    assert_eq!(soft_binding::SOURCE_LICENSE, "CC BY 4.0");
    assert_eq!(
        (
            registry[0].algorithm,
            soft_binding::registered_label(registry[0].algorithm)
        ),
        ("com.digimarc.validate.1", Some("Digimarc Validate"))
    );
    let apis = [
        ("ai.trufo.pawprint.watermark", "https://c2pa.trufo.ai/v1"),
        ("ai.trufo.pawprint.fingerprint", "https://c2pa.trufo.ai/v1"),
        ("com.aiwatermark.videoseal.1", "https://aiwatermark.com/api/v1"),
        ("com.aiwatermark.pixelseal.1", "https://aiwatermark.com/api/v1"),
        ("com.aiwatermark.audioseal.1", "https://aiwatermark.com/api/v1"),
        (
            "me.deepmark.audio.vigil.128",
            "https://resolution-api.deepmark.me",
        ),
        (
            "io.blockfact.audio.watermark.32",
            "https://api.blockfact.io/api/soft-binding/v1",
        ),
        ("com.joinmonolith.sha256", "https://api.joinmonolith.com/api/c2pa"),
        ("studio.aidr.silentcipher.1", "https://aidr.studio/api/c2pa/v1"),
        ("com.etchv.watermark.1", "https://api.etchv.com/c2pa/v1"),
        (
            "com.ottxp.koltheia.photonote.1",
            "https://koltheia.ottxp.com/api/softbinding/v1",
        ),
        ("com.art-koffee.phash.1", "https://www.art-koffee.com/c2pa/v1"),
    ];
    for entry in registry {
        let expected: Vec<&str> = apis
            .iter()
            .filter(|(alg, _)| *alg == entry.algorithm)
            .map(|(_, url)| *url)
            .collect();
        assert_eq!(entry.resolution_apis, expected.as_slice(), "{}", entry.algorithm);
    }
}

#[test]
fn registry_attribution() {
    let platform = |issuer: &[u8]| {
        registry::C2PA_AI_VENDORS
            .iter()
            .find(|v| v.issuer == issuer)
            .and_then(|v| v.platform)
    };
    assert_eq!(platform(b"Ideogram"), Some("Ideogram"));
    assert!(registry::is_identity_ai_org("Ideogram"));
    assert_eq!(platform(b"volcengine"), Some("ByteDance Volcano Engine"));
    assert_eq!(platform(b"Byteplus"), Some("BytePlus (ByteDance)"));
    assert_eq!(platform(b"Dreamina"), Some("ByteDance Dreamina"));
    assert!(registry::C2PA_CLAIM_GENERATOR_PLATFORMS.contains(&("dreamina", "ByteDance Dreamina")));
    for issuer in [&b"PixVerse"[..], b"Suno Inc"] {
        assert!(
            registry::C2PA_AI_VENDORS
                .iter()
                .any(|v| v.issuer == issuer && v.asserts_ai)
        );
    }
    assert!(!attribution::issuer_word_matches("Nova Canvas").contains(&"Canva".to_owned()));
    assert_eq!(attribution::issuer_word_matches("Canva"), ["Canva"]);
    assert!(attribution::issuer_word_matches("Adobe_Firefly").contains(&"Adobe".to_owned()));
    assert!(attribution::tool_word_matches("Adobe_Firefly").contains(&"Firefly".to_owned()));
    assert_eq!(
        attribution::claim_generator_platform(Some("Sunoco Studio 2.1")),
        None
    );
    assert_eq!(
        attribution::claim_generator_platform(Some("suno-v4 export")),
        Some("Suno")
    );
    assert_eq!(
        attribution::claim_generator_platform(Some("Adobe_Firefly")),
        Some("Adobe Firefly")
    );
}

#[test]
fn removal_hints() {
    let mut base = parsed(b"jumb c2pa");
    assert!(!base.has_removal_hint());
    base.soft_binding_vendors = vec!["Microsoft InvisMark".into()];
    assert!(base.has_invismark() && base.has_removal_hint());
    let mut fingerprint = parsed(b"jumb c2pa");
    fingerprint.soft_binding = Some("Adobe Image Comparator Network".into());
    fingerprint.soft_binding_vendors = vec!["Adobe Image Comparator Network".into()];
    assert!(!fingerprint.has_removal_hint());
    let mut identity = parsed(b"jumb c2pa");
    identity.c2pa_identity_ai = true;
    assert!(identity.has_removal_hint());
}

fn manifest(store_body: Value) -> Value {
    let mut manifests = serde_json::Map::new();
    manifests.insert("active".into(), store_body);
    json!({"active_manifest": "active", "manifests": manifests})
}

#[test]
fn structured_extraction_follows_only_reachable_credentials() {
    let unreachable = info(&json!({"active_manifest": "active", "manifests": {
        "active": {"signature_info": {"issuer": "Adobe"}, "assertions": []},
        "unreachable": {"signature_info": {"issuer": "OpenAI"}, "assertions": [{"label": "c2pa.actions.v2", "data": {"actions": [{"action": "c2pa.created", "digitalSourceType": "trainedAlgorithmicMedia"}]}}]}}}));
    assert_eq!(unreachable.issuer.as_deref(), Some("Adobe"));
    assert!(
        unreachable.source_type.is_none()
            && unreachable.ai_source_kind.is_none()
            && !unreachable.c2pa_identity_ai
    );

    let reachable = info(&json!({"active_manifest": "update", "manifests": {
        "update": {"claim_generator": "c2pa-tool/0.1.0", "ingredients": [{"active_manifest": "created"}], "assertions": []},
        "created": {"claim_generator": "Dreamina/7.5.0", "assertions": []}}}));
    assert_eq!(
        (reachable.ai_tool.as_deref(), reachable.c2pa_identity_ai),
        (Some("Dreamina"), true)
    );

    for failure in ["assertion.dataHash.mismatch", "signingCredential.ocsp.revoked"] {
        let tainted = info(&json!({"active_manifest": "update",
            "validation_results": {"activeManifest": {"success": [{"code": "assertion.dataHash.match"}, {"code": "claimSignature.validated"}],
                "failure": [{"code": "signingCredential.untrusted"}]}, "ingredientDeltas": [{"validationDeltas": {"failure": [{"code": failure}]}}]},
            "manifests": {
                "update": {"claim_generator": "c2pa-tool/0.1.0", "ingredients": [{"active_manifest": "created", "validation_results": {"activeManifest": {"failure": [{"code": failure}]}}}], "assertions": []},
                "created": {"claim_generator": "Dreamina/7.5.0", "assertions": []}}}));
        let v = &tainted.validation;
        assert_eq!(
            (v.integrity, v.signature, v.signer_trust),
            (Check::Valid, Check::Valid, SignerTrust::Untrusted)
        );
        assert!(tainted.ai_tool.is_none() && !tainted.c2pa_identity_ai);
    }
}

#[test]
fn signer_identities_are_normalized() {
    let xai = info(&manifest(
        json!({"signature_info": {"issuer": "Self-signed ephemeral certificate (Content Authenticity SDK) -- LOCAL USE ONLY", "common_name": "xAI Grok Imagine"}, "assertions": []}),
    ));
    assert_eq!(
        (xai.issuer.as_deref(), xai.c2pa_identity_ai),
        (Some("xAI Grok Imagine"), true)
    );
    for (issuer, org) in [("Bytedance Pte. Ltd.", "ByteDance"), ("TikTok Inc.", "TikTok")] {
        let publisher = info(&manifest(
            json!({"signature_info": {"issuer": issuer}, "assertions": []}),
        ));
        assert_eq!(
            (publisher.issuer.as_deref(), publisher.c2pa_identity_ai),
            (Some(org), false)
        );
    }
    let toolbox = info(
        &json!({"active_manifest": "m1", "manifests": {"m1": {"claim_generator_info": [{"name": "TIKTOK AD Creative Toolbox"}], "signature_info": {"issuer": "TikTok"}}}}),
    );
    assert_eq!(toolbox.ai_tool.as_deref(), Some("TikTok Ad Creative Toolbox"));
    let unknown = info(
        &json!({"active_manifest": "m", "validation_results": {"activeManifest": {"success": [{"code": "assertion.dataHash.match"}, {"code": "claimSignature.validated"}]}},
        "manifests": {"m": {"signature_info": {"issuer": "Example Studio, Inc.", "common_name": "Example Studio, Inc."},
            "assertions": [{"label": "c2pa.actions.v2", "data": {"actions": [{"action": "c2pa.created", "softwareAgent": "Example Renderer", "digitalSourceType": format!("{DST}trainedAlgorithmicMedia")}]}}]}}}),
    );
    assert_eq!(unknown.issuer, None);
}

#[test]
fn soft_binding_assertions() {
    let paint = info(&manifest(
        json!({"assertions": [{"label": "c2pa.soft-binding", "data": {"alg": "com.microsoft.invismark.1", "blocks": [{"scope": "the entire image", "value": "83424621-03cb-40e3-9808-a9fae837156d"}]}}]}),
    ));
    assert_eq!(paint.soft_binding.as_deref(), Some("Microsoft InvisMark"));
    assert_eq!(
        paint.soft_binding_algorithm.as_deref(),
        Some("com.microsoft.invismark.1")
    );
    assert_eq!(
        paint.soft_binding_value.as_deref(),
        Some("83424621-03cb-40e3-9808-a9fae837156d")
    );
    let unattributed = info(&manifest(
        json!({"assertions": [{"label": "c2pa.soft-binding", "data": {"blocks": [{"value": "not-attributable"}]}}]}),
    ));
    assert_eq!(unattributed.soft_binding_value, None);
    let suno = info(
        &json!({"active_manifest": "urn:c2pa:test", "manifests": {"urn:c2pa:test": {"claim_generator_info": [{"name": "Suno"}],
        "assertions": [{"label": "c2pa.actions.v2", "data": {"actions": [{"action": "c2pa.created", "digitalSourceType": format!("{DST}trainedAlgorithmicMedia")}]}},
                       {"label": "c2pa.soft-binding", "data": {"alg": "com.suno.fingerprint.v4", "blocks": [{"value": "abc-123"}]}}],
        "signature_info": {"issuer": "Suno Inc"}}}}),
    );
    assert_eq!(
        (suno.issuer.as_deref(), suno.soft_binding.as_deref()),
        (Some("Suno"), Some("Suno Audio Fingerprint"))
    );
}

fn designer(with_soft_binding: bool) -> Value {
    let mut assertions = vec![json!({"label": "c2pa.actions", "data": {"actions": [
        {"action": "c2pa.created", "softwareAgent": {"name": "Azure OpenAI ImageGen"}, "digitalSourceType": format!("{DST}trainedAlgorithmicMedia")},
        {"action": "c2pa.watermarked"}]}})];
    if with_soft_binding {
        assertions.push(json!({"label": "c2pa.soft-binding", "data": {"alg": "com.microsoft.invismark.1", "blocks": [{"value": "bf7a2993-cc1f-47e1-b1f0-cd8839aabb22"}]}}));
    }
    json!({"active_manifest": "designer", "manifests": {"designer": {
        "signature_info": {"issuer": "Microsoft Corporation", "common_name": "Microsoft Corporation"},
        "claim_generator_info": [{"name": "Microsoft Responsible AI Provenance", "version": "1.0"}],
        "assertions": assertions}}})
}

#[test]
fn synthid_scoping_follows_the_signer() {
    let named = info(&designer(true));
    assert_eq!(named.ai_source_kind, Some(AiSourceKind::Generated));
    assert_eq!(
        named.soft_binding_algorithm.as_deref(),
        Some("com.microsoft.invismark.1")
    );
    assert!(named.synthid_watermark.is_none() && named.synthid_vendors.is_empty());
    assert!(info(&designer(false)).synthid_watermark.is_none());

    let google = info(&manifest(
        json!({"signature_info": {"issuer": "Google LLC"}, "assertions": [
        {"label": "c2pa.actions", "data": {"actions": [{"action": "c2pa.created", "digitalSourceType": format!("{DST}trainedAlgorithmicMedia")}]}},
        {"label": "c2pa.soft-binding", "data": {"alg": "io.iscc.v0"}}]}),
    ));
    assert_eq!(google.soft_binding.as_deref(), Some("ISCC (content code)"));
    assert_eq!(google.synthid_vendors, ["Google LLC"]);
    assert_eq!(
        google.synthid_watermark.as_deref(),
        Some("present according to Google LLC provenance")
    );

    let partner = info(
        &json!({"active_manifest": "urn:uuid:partner-firefly", "manifests": {"urn:uuid:partner-firefly": {
        "signature_info": {"issuer": "Adobe Inc.", "common_name": "Adobe Firefly C2PA"}, "claim_generator": "Adobe_Firefly",
        "claim_generator_info": [{"name": "Adobe Firefly", "org.contentauth.c2pa_rs": "0.90.4"}],
        "assertions": [{"label": "c2pa.actions.v2", "data": {"actions": [{"action": "c2pa.opened", "parameters": {
            "com.adobe.details": "Gemini Flash", "com.adobe.digitalSourceType": format!("{DST}trainedAlgorithmicMedia"),
            "com.adobe.modelId": "gemini-flash", "com.adobe.modelVersion": "nano-banana-3", "com.adobe.type": "remoteProvider.3rdParty"}}]}}]}}}),
    );
    assert_eq!(
        partner.source_type.as_deref(),
        Some("trainedAlgorithmicMedia (AI-generated)")
    );
    assert!(partner.synthid_watermark.is_none() && partner.synthid_vendors.is_empty());

    for descriptions in [vec![], vec!["Added imperceptible SynthID watermark"]] {
        let mut actions = vec![
            json!({"action": "c2pa.deleted", "digitalSourceType": "compositeWithTrainedAlgorithmicMedia"}),
        ];
        actions.extend(
            descriptions
                .iter()
                .map(|d| json!({"action": "c2pa.edited", "description": d})),
        );
        let photos = info(
            &json!({"active_manifest": "edit", "manifests": {"edit": {"signature_info": {"issuer": "Google LLC", "common_name": "Google Photos"},
            "assertions": [{"label": "c2pa.actions.v2", "data": {"actions": actions}}]}}}),
        );
        assert_eq!(photos.synthid_vendors, ["Google LLC"]);
    }
}

#[test]
fn revoked_credentials_are_disqualifying() {
    let revoked = info(&json!({"active_manifest": "created",
        "validation_results": {"activeManifest": {"success": [{"code": "assertion.dataHash.match"}, {"code": "claimSignature.validated"}], "failure": [{"code": "signingCredential.ocsp.revoked"}]}},
        "manifests": {"created": {"signature_info": {"issuer": "OpenAI"}, "assertions": [{"label": "c2pa.actions.v2", "data": {"actions": [{"action": "c2pa.created", "digitalSourceType": "trainedAlgorithmicMedia"}]}}]}}}));
    assert_eq!(revoked.validation.signer_validity, SignerValidity::Invalid);
    assert!(revoked.has_invalid_credential());
}

#[test]
fn real_samples_through_the_reader() {
    let chatgpt = raiw_c2pa::extract_c2pa_info(&fixture("provenance/chatgpt-1.png")).unwrap();
    assert!(chatgpt.issuer.as_deref().unwrap().contains("OpenAI"));
    assert!(
        chatgpt
            .source_type
            .as_deref()
            .unwrap()
            .contains("trainedAlgorithmicMedia")
    );
    assert_eq!(chatgpt.claim_generator.as_deref(), Some("ChatGPT"));
    assert!(chatgpt.synthid_watermark.is_none());
    let v = &chatgpt.validation;
    assert_eq!(
        (v.integrity, v.signature, v.signer_trust, v.signer_validity),
        (
            Check::Valid,
            Check::Valid,
            SignerTrust::Untrusted,
            SignerValidity::Expired
        )
    );
    assert!(v.codes.contains(&"assertion.dataHash.match".to_owned()));
    let flux = raiw_c2pa::extract_c2pa_info(&fixture("provenance/flux-1.jpg")).unwrap();
    assert!(flux.c2pa_manifest.starts_with("C2PA manifest store"));
    assert!(flux.issuer.as_deref().unwrap().contains("Black Forest Labs"));
    for name in [
        "provenance/higgsfield-gpt-image.jpg",
        "provenance/microsoft-bing-image-creator.jpg",
        "provenance/microsoft-invismark.png",
        "provenance/microsoft-paint-invismark.png",
        "visible/microsoft/provider-original.png",
    ] {
        let found = raiw_c2pa::extract_c2pa_info(&fixture(name)).unwrap();
        let v = &found.validation;
        assert_eq!(
            (v.integrity, v.signature, v.signer_trust),
            (Check::Valid, Check::Valid, SignerTrust::Untrusted),
            "{name}"
        );
        assert_ne!(v.signer_validity, SignerValidity::Invalid, "{name}");
        assert!(
            !v.codes.contains(&"signingCredential.invalid".to_owned()),
            "{name}"
        );
        if name.contains("microsoft") {
            assert!(v.codes.contains(&"claim.malformed".to_owned()), "{name}");
            assert!(
                found
                    .soft_binding_algorithm
                    .as_deref()
                    .unwrap()
                    .contains("com.microsoft.invismark.1"),
                "{name}"
            );
        }
    }
    let chunk = png::extract_c2pa_chunk(&fixture("provenance/chatgpt-1.png")).unwrap();
    assert_eq!(&chunk[4..8], b"caBX");
    let fallback = png::extract_c2pa_info_png(&fixture("provenance/chatgpt-1.png")).unwrap();
    assert!(fallback.c2pa_manifest.starts_with("C2PA manifest ("));
    assert_eq!(
        (fallback.validation.integrity, fallback.validation.source.as_str()),
        (Check::Unknown, "fallback")
    );
}

#[test]
fn png_guards_and_injection() {
    let dir = tempfile::tempdir().unwrap();
    let fake = dir.path().join("fake.png");
    std::fs::write(&fake, b"\xff\xd8\xff\xe0 not a png at all, just garbage bytes").unwrap();
    assert!(!png::has_c2pa_metadata(&fake) && png::extract_c2pa_chunk(&fake).is_none());
    let clamp = dir.path().join("clamp.png");
    std::fs::write(
        &clamp,
        [
            &b"\x89PNG\r\n\x1a\n"[..],
            &0x7FFF_FFFFu32.to_be_bytes(),
            b"caBX",
            b"jumbc2pa-not-really",
        ]
        .concat(),
    )
    .unwrap();
    assert!(!png::has_c2pa_metadata(&clamp));
    assert!(raiw_c2pa::extract_c2pa_info(&clamp).is_none());
    let err =
        png::inject_c2pa_chunk(&dir.path().join("in.jpg"), &dir.path().join("out.png"), b"").unwrap_err();
    assert!(err.to_string().contains("only supported for PNG"));

    let clean = fixture("provenance/mj-1.png");
    assert!(raiw_c2pa::extract_c2pa_info(&clean).is_none());
    let chunk = png::extract_c2pa_chunk(&fixture("provenance/chatgpt-1.png")).unwrap();
    let out = dir.path().join("injected.png");
    png::inject_c2pa_chunk(&clean, &out, &chunk).unwrap();
    assert!(png::has_c2pa_metadata(&out));
    assert!(
        png::extract_c2pa_info_png(&out)
            .unwrap()
            .issuer
            .unwrap()
            .contains("OpenAI")
    );
}
