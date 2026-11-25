//! Manifest-store JSON (the validating reader's output) to normalized [`C2paInfo`].
//!
//! Provenance is read only from the active manifest and the ingredient graph reachable
//! through credentials that validated; the store's URNs are never part of the result.

use std::collections::{HashSet, VecDeque};

use raiw_core::{JsonStyle, to_py_json};
use serde_json::{Map, Value};

use crate::attribution::{
    any_identity_ai_org, issuer_word_matches, synthid_evidence_vendors_in, synthid_verdict, tool_word_matches,
};
use crate::info::{AiSourceKind, C2paInfo, Check, SignerTrust, SignerValidity, Validation, ValidationSource};
use crate::registry::action_name;
use crate::soft_binding::{self, SoftBindingKind};
use crate::text::is_printable;

const CONTENT_BINDING_MATCHES: [&str; 5] = [
    "assertion.dataHash.match",
    "assertion.bmffHash.match",
    "assertion.boxesHash.match",
    "assertion.collectionHash.match",
    "assertion.multiAssetHash.match",
];
const CONTENT_BINDING_FAILURE_MARKERS: [&str; 8] = [
    "Hash.incorrectFileCount",
    "Hash.malformed",
    "Hash.mismatch",
    "Hash.missingPart",
    "assertion.hashedURI.mismatch",
    "claim.hardBindings.missing",
    "hashedUri.mismatch",
    "ingredient.manifest.mismatch",
];
const SIGNATURE_FAILURES: [&str; 3] = [
    "claimSignature.mismatch",
    "claimSignature.missing",
    "claimSignature.outsideValidity",
];
/// Credentials the issuer disowned; an expired certificate is deliberately absent.
const CREDENTIAL_FAILURES: [&str; 2] = ["signingCredential.invalid", "signingCredential.ocsp.revoked"];
const AI_DISCLOSURE_ENHANCED_LEVELS: [&str; 1] = ["human_validated"];
pub(crate) const AI_DISCLOSURE_LEVELS: [&str; 3] = ["human_validated", "fully_autonomous", "prompt_guided"];

fn push_unique(values: &mut Vec<String>, value: &str) {
    if !values.iter().any(|existing| existing == value) {
        values.push(value.to_owned());
    }
}

fn unique(values: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for value in values {
        push_unique(&mut out, value);
    }
    out
}

fn short_printable(value: &Value) -> Option<&str> {
    value
        .as_str()
        .filter(|text| is_printable(text) && text.chars().count() <= 256)
}

/// The source kind a set of `humanOversightLevel` values implies (C2PA 2.4, 18.28.3).
pub(crate) fn disclosure_kind(oversight: &[String]) -> AiSourceKind {
    if !oversight.is_empty()
        && oversight
            .iter()
            .all(|level| AI_DISCLOSURE_ENHANCED_LEVELS.contains(&level.as_str()))
    {
        AiSourceKind::Enhanced
    } else {
        AiSourceKind::Generated
    }
}

fn active_manifest(store: &Map<String, Value>) -> Option<&Map<String, Value>> {
    let label = store.get("active_manifest")?.as_str()?;
    store.get("manifests")?.as_object()?.get(label)?.as_object()
}

/// Ordered success, informational and failure codes of this store's active credential.
fn status_codes(store: &Map<String, Value>) -> (Vec<String>, Vec<String>, Vec<String>) {
    let codes = |values: Option<&Value>| -> Vec<String> {
        values
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|value| {
                value
                    .as_object()?
                    .get("code")?
                    .as_str()
                    .filter(|code| !code.is_empty())
            })
            .map(str::to_owned)
            .collect()
    };
    let active = store
        .get("validation_results")
        .and_then(Value::as_object)
        .and_then(|r| r.get("activeManifest"));
    let active = active.and_then(Value::as_object);
    let (successes, informationals, mut failures) = match active {
        Some(set) => (
            codes(set.get("success")),
            codes(set.get("informational")),
            codes(set.get("failure")),
        ),
        None => (Vec::new(), Vec::new(), Vec::new()),
    };
    if successes.is_empty() && informationals.is_empty() && failures.is_empty() {
        // Older readers expose only the flat non-success list.
        failures = codes(store.get("validation_status"));
    }
    (successes, informationals, failures)
}

/// The validation dimensions of one store (or ingredient) credential.
pub fn validation_fields(store: &Map<String, Value>) -> Validation {
    let (successes, informationals, failures) = status_codes(store);
    let all: Vec<String> = successes
        .iter()
        .chain(&informationals)
        .chain(&failures)
        .cloned()
        .collect();
    let is_binding_failure = |code: &str| {
        CONTENT_BINDING_FAILURE_MARKERS
            .iter()
            .any(|marker| code.contains(marker))
    };
    let failed_codes: Vec<String> = failures
        .iter()
        .filter(|code| {
            is_binding_failure(code)
                || SIGNATURE_FAILURES.contains(&code.as_str())
                || CREDENTIAL_FAILURES.contains(&code.as_str())
        })
        .cloned()
        .collect();
    let has = |codes: &[String], code: &str| codes.iter().any(|c| c == code);
    let binding_matched = successes.iter().any(|code| {
        CONTENT_BINDING_MATCHES
            .iter()
            .any(|suffix| code.ends_with(suffix))
    });
    let integrity = if failed_codes.iter().any(|code| is_binding_failure(code)) {
        Check::Invalid
    } else if binding_matched {
        Check::Valid
    } else {
        Check::Unknown
    };
    let signature = if failed_codes
        .iter()
        .any(|code| SIGNATURE_FAILURES.contains(&code.as_str()))
    {
        Check::Invalid
    } else if has(&successes, "claimSignature.validated") {
        Check::Valid
    } else {
        Check::Unknown
    };
    let signer_trust = if has(&successes, "signingCredential.trusted") {
        SignerTrust::Trusted
    } else if has(&failures, "signingCredential.untrusted") {
        SignerTrust::Untrusted
    } else {
        SignerTrust::Unknown
    };
    let signer_validity = if failed_codes
        .iter()
        .any(|code| CREDENTIAL_FAILURES.contains(&code.as_str()))
    {
        SignerValidity::Invalid
    } else if has(&failures, "signingCredential.expired") {
        SignerValidity::Expired
    } else if has(&successes, "claimSignature.insideValidity") {
        SignerValidity::Valid
    } else {
        SignerValidity::Unknown
    };
    let state = store
        .get("validation_state")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .unwrap_or("unknown");
    Validation {
        source: ValidationSource::Reader,
        state: state.to_owned(),
        integrity,
        signature,
        signer_trust,
        signer_validity,
        codes: unique(&all),
        failed_codes,
    }
}

/// The active manifest followed by every ingredient manifest reachable through valid credentials.
fn manifest_chain(store: &Map<String, Value>) -> Vec<&Map<String, Value>> {
    let (Some(manifests), Some(active)) = (
        store.get("manifests").and_then(Value::as_object),
        store.get("active_manifest").and_then(Value::as_str),
    ) else {
        return Vec::new();
    };
    let mut pending = VecDeque::from([active]);
    let mut seen = HashSet::new();
    let mut chain = Vec::new();
    while let Some(label) = pending.pop_front() {
        if !seen.insert(label) {
            continue;
        }
        let Some(manifest) = manifests.get(label).and_then(Value::as_object) else {
            continue;
        };
        chain.push(manifest);
        let Some(ingredients) = manifest.get("ingredients").and_then(Value::as_array) else {
            continue;
        };
        for ingredient in ingredients.iter().filter_map(Value::as_object) {
            if let Some(child) = ingredient.get("active_manifest").and_then(Value::as_str)
                && !seen.contains(child)
                && !validation_fields(ingredient).is_invalid()
            {
                pending.push_back(child);
            }
        }
    }
    chain
}

#[derive(Default)]
struct Evidence {
    issuers: Vec<String>,
    tools: Vec<String>,
    actions: Vec<String>,
    disclosures: Vec<Map<String, Value>>,
    source_types: Vec<String>,
    algorithms: Vec<String>,
    binding_values: Vec<String>,
    identity_strings: Vec<String>,
    claim_generator_asserts_ai: bool,
    records_synthid_action: bool,
}

impl Evidence {
    fn add_tools(&mut self, value: &str, asserts_ai: bool) {
        let matches = tool_word_matches(value);
        if asserts_ai && !matches.is_empty() {
            self.claim_generator_asserts_ai = true;
        }
        self.tools.extend(matches);
    }

    fn collect(&mut self, manifest: &Map<String, Value>) {
        if let Some(signature) = manifest.get("signature_info").and_then(Value::as_object) {
            for key in ["issuer", "common_name", "certificate_issuer"] {
                if let Some(value) = signature.get(key).and_then(Value::as_str) {
                    self.issuers.extend(issuer_word_matches(value));
                    self.identity_strings.push(value.to_owned());
                }
            }
        }
        for ingredient in manifest
            .get("ingredients")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let Some(source) = ingredient.get("digitalSourceType").and_then(Value::as_str) {
                self.source_types.push(source.to_owned());
            }
        }
        if let Some(generator) = manifest.get("claim_generator").and_then(Value::as_str) {
            self.identity_strings.push(generator.to_owned());
            self.add_tools(generator, true);
        }
        for candidate in manifest
            .get("claim_generator_info")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let Some(name) = candidate.get("name").and_then(Value::as_str) {
                self.add_tools(name, true);
                self.identity_strings.push(name.to_owned());
            }
        }
        for assertion in manifest
            .get("assertions")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let Some(assertion) = assertion.as_object() else {
                continue;
            };
            let (Some(label), Some(data)) = (
                assertion.get("label").and_then(Value::as_str),
                assertion.get("data").and_then(Value::as_object),
            ) else {
                continue;
            };
            self.collect_assertion(label, data);
        }
    }

    fn collect_assertion(&mut self, label: &str, data: &Map<String, Value>) {
        if label.starts_with("c2pa.soft-binding")
            && let Some(algorithm) = data.get("alg").and_then(Value::as_str)
        {
            self.algorithms.push(algorithm.to_owned());
            for block in data.get("blocks").and_then(Value::as_array).into_iter().flatten() {
                if let Some(value) = block.get("value").and_then(short_printable) {
                    self.binding_values.push(value.to_owned());
                }
            }
        }
        if label.starts_with("c2pa.ingredient") {
            // C2PA 2.4 lets a manifest-less ingredient carry digitalSourceType directly.
            if let Some(source) = data.get("digitalSourceType").and_then(Value::as_str) {
                self.source_types.push(source.to_owned());
            }
            return;
        }
        if label.starts_with("c2pa.ai-disclosure") {
            self.disclosures.push(data.clone());
            return;
        }
        if !label.starts_with("c2pa.actions") {
            return;
        }
        for action in data
            .get("actions")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let Some(action) = action.as_object() else {
                continue;
            };
            if let Some(name) = action.get("action").and_then(Value::as_str) {
                self.actions.push(action_name(name));
            }
            if action
                .get("description")
                .and_then(Value::as_str)
                .is_some_and(|d| d.contains("SynthID"))
            {
                self.records_synthid_action = true;
            }
            if let Some(source) = action.get("digitalSourceType").and_then(Value::as_str) {
                self.source_types.push(source.to_owned());
            }
            let adobe = action
                .get("parameters")
                .and_then(Value::as_object)
                .and_then(|p| p.get("com.adobe.digitalSourceType"));
            if let Some(source) = adobe.and_then(Value::as_str) {
                self.source_types.push(source.to_owned());
            }
            let agent = action
                .get("softwareAgent")
                .and_then(Value::as_object)
                .and_then(|agent| agent.get("name"));
            if let Some(name) = agent.and_then(Value::as_str) {
                self.add_tools(name, false);
            }
        }
    }
}

/// Record `c2pa.ai-disclosure` fields and return the source kind they imply.
fn apply_disclosures(disclosures: &[Map<String, Value>], info: &mut C2paInfo) -> Option<AiSourceKind> {
    if disclosures.is_empty() {
        return None;
    }
    let mut models = Vec::new();
    let mut oversight = Vec::new();
    for disclosure in disclosures {
        if let Some(model) = ["modelName", "modelIdentifier", "modelType"]
            .iter()
            .find_map(|key| disclosure.get(*key).and_then(short_printable))
        {
            models.push(model.to_owned());
        }
        let level = disclosure
            .get("contentProfile")
            .and_then(Value::as_object)
            .and_then(|p| p.get("humanOversightLevel"));
        if let Some(level) = level.and_then(Value::as_str) {
            oversight.push(level.to_owned());
        }
    }
    let models = unique(&models);
    info.ai_disclosure = Some(if models.is_empty() {
        "present".to_owned()
    } else {
        models.join(", ")
    });
    if !oversight.is_empty() {
        info.human_oversight = Some(unique(&oversight).join(", "));
    }
    Some(disclosure_kind(&oversight))
}

fn apply_structured_fields(store: &Map<String, Value>, info: &mut C2paInfo) {
    let chain = manifest_chain(store);
    if chain.is_empty() {
        return;
    }
    let mut evidence = Evidence::default();
    for manifest in chain {
        evidence.collect(manifest);
    }
    let issuers = unique(&evidence.issuers);
    let tools = unique(&evidence.tools);
    let actions = unique(&evidence.actions);
    info.issuer = (!issuers.is_empty()).then(|| issuers.join(", "));
    info.ai_tool = (!tools.is_empty()).then(|| tools.join(", "));
    info.actions = (!actions.is_empty()).then(|| actions.join(", "));
    info.c2pa_identity_ai = evidence.claim_generator_asserts_ai || any_identity_ai_org(&issuers);

    let types = &evidence.source_types;
    let generated = types.iter().any(|v| {
        v.contains("trainedAlgorithmicMedia") && !v.contains("compositeWithTrainedAlgorithmicMedia")
    });
    let enhanced = types
        .iter()
        .any(|v| v.contains("compositeWithTrainedAlgorithmicMedia"));
    let procedural = types.iter().any(|v| v.contains("algorithmicMedia"));
    let disclosure = apply_disclosures(&evidence.disclosures, info);
    if generated {
        info.source_type = Some("trainedAlgorithmicMedia (AI-generated)".into());
        info.ai_source_kind = Some(AiSourceKind::Generated);
    } else if enhanced {
        info.source_type = Some("compositeWithTrainedAlgorithmicMedia (AI-enhanced)".into());
        info.ai_source_kind = Some(AiSourceKind::Enhanced);
    } else if let Some(kind) = disclosure {
        info.source_type = Some(format!("c2pa.ai-disclosure ({})", kind.as_str()));
        info.ai_source_kind = Some(kind);
    } else if procedural {
        info.source_type = Some("algorithmicMedia".into());
    }

    let has_watermark_action = actions.iter().any(|action| action.starts_with("watermarked"));
    info.watermarked = has_watermark_action;
    let algorithms = unique(&evidence.algorithms);
    let entries = soft_binding::registry_entries_in(algorithms.join("\n").as_bytes());
    let blocks_synthid = entries
        .iter()
        .any(|entry| entry.kind == SoftBindingKind::Watermark)
        || entries.len() < algorithms.len();
    if info.ai_source_kind.is_some() && !blocks_synthid {
        // Scope SynthID to the signer/generator identity, never a vendor named elsewhere.
        let identity = to_py_json(&evidence.identity_strings, JsonStyle::UNICODE);
        let vendors = synthid_evidence_vendors_in(
            identity.as_bytes(),
            Some(has_watermark_action),
            Some(evidence.records_synthid_action),
        );
        if !vendors.is_empty() {
            info.synthid_watermark = Some(synthid_verdict(&vendors.join(", ")));
            info.synthid_vendors = vendors;
        }
    }
    if !algorithms.is_empty() {
        let labels = soft_binding::labels(entries.iter().copied(), None);
        if !labels.is_empty() {
            info.soft_binding = Some(labels.join(", "));
            info.soft_binding_vendors = labels;
        }
        info.soft_binding_algorithm = Some(algorithms.join(", "));
    }
    if !evidence.binding_values.is_empty() {
        info.soft_binding_value = Some(unique(&evidence.binding_values).join(", "));
    }
}

fn printable(value: Option<&Value>) -> Option<&str> {
    value
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty() && is_printable(s))
}

fn claim_generator(store: &Map<String, Value>) -> Option<String> {
    let active = active_manifest(store)?;
    if let Some(direct) = printable(active.get("claim_generator")) {
        return Some(direct.to_owned());
    }
    let first = active
        .get("claim_generator_info")?
        .as_array()?
        .first()?
        .as_object()?;
    printable(first.get("name")).map(str::to_owned)
}

/// Python's truthiness of a decoded JSON value.
fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|n| n != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

/// Python's `str(value)` for a decoded JSON value.
fn py_str(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => py_repr(other),
    }
}

fn py_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(number) => match (number.as_i64(), number.as_u64(), number.as_f64()) {
            (Some(int), _, _) => int.to_string(),
            (None, Some(uint), _) => uint.to_string(),
            (_, _, Some(float)) => raiw_core::pyjson::py_float_repr(float),
            _ => number.to_string(),
        },
        Value::String(text) => format!("'{}'", text.replace('\\', "\\\\").replace('\'', "\\'")),
        Value::Array(items) => format!("[{}]", items.iter().map(py_repr).collect::<Vec<_>>().join(", ")),
        Value::Object(map) => {
            let fields: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{}: {}", py_repr(&Value::String(k.clone())), py_repr(v)))
                .collect();
            format!("{{{}}}", fields.join(", "))
        }
    }
}

/// Normalize a decoded manifest store whose JSON text was `byte_count` bytes long.
pub fn info_from_store(store: &Map<String, Value>, byte_count: usize) -> C2paInfo {
    let mut info = C2paInfo::new(byte_count, validation_fields(store));
    apply_structured_fields(store, &mut info);
    info.claim_generator = claim_generator(store);
    let time = active_manifest(store)
        .and_then(|m| m.get("signature_info"))
        .and_then(Value::as_object)
        .and_then(|s| s.get("time"));
    info.timestamp = time.filter(|value| truthy(value)).map(py_str);
    info
}

/// `c2pa_info_from_manifest_store`: `None` for invalid JSON, an empty store or an `error` key.
pub fn info_from_manifest_json(store: &str) -> Option<C2paInfo> {
    let decoded: Value = serde_json::from_str(store).ok()?;
    let map = decoded
        .as_object()
        .filter(|map| !map.is_empty() && !map.get("error").is_some_and(truthy))?;
    Some(info_from_store(map, store.len()))
}

/// `c2pa_info_from_manifest_store` for an already decoded store, measured as `json.dumps` would.
pub fn info_from_manifest_value(store: &Value) -> Option<C2paInfo> {
    let map = store
        .as_object()
        .filter(|map| !map.is_empty() && !map.get("error").is_some_and(truthy))?;
    Some(info_from_store(map, to_py_json(store, JsonStyle::UNICODE).len()))
}
