//! The normalized C2PA evidence (Python's `c2pa_info` dict) and its verdict helpers.

use serde::ser::{Serialize, SerializeMap, Serializer};

/// Where the evidence came from: the validating reader or the raw JUMBF/CBOR scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValidationSource {
    Reader,
    Fallback,
}

impl ValidationSource {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Reader => "reader",
            Self::Fallback => "fallback",
        }
    }
}

/// State of the content binding or the claim signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Check {
    Valid,
    Invalid,
    Unknown,
}

impl Check {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Valid => "valid",
            Self::Invalid => "invalid",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SignerTrust {
    Trusted,
    Untrusted,
    Unknown,
}

impl SignerTrust {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Trusted => "trusted",
            Self::Untrusted => "untrusted",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SignerValidity {
    Valid,
    Invalid,
    Expired,
    Unknown,
}

impl SignerValidity {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Valid => "valid",
            Self::Invalid => "invalid",
            Self::Expired => "expired",
            Self::Unknown => "unknown",
        }
    }
}

/// Whether provenance says the asset was generated or enhanced by a trained model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AiSourceKind {
    Generated,
    Enhanced,
}

impl AiSourceKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Generated => "generated",
            Self::Enhanced => "enhanced",
        }
    }
}

/// `c2pa_credential_level`: how far the reader tied the manifest to these bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CredentialLevel {
    Invalid,
    Verified,
    Unverified,
}

impl CredentialLevel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Invalid => "invalid",
            Self::Verified => "verified",
            Self::Unverified => "unverified",
        }
    }
}

/// The validation dimensions (`c2pa_validation_*`, `c2pa_integrity`, `c2pa_signature`, ...).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Validation {
    pub source: ValidationSource,
    pub state: String,
    pub integrity: Check,
    pub signature: Check,
    pub signer_trust: SignerTrust,
    pub signer_validity: SignerValidity,
    /// Every status code, de-duplicated in reader order.
    pub codes: Vec<String>,
    /// The failures that drove a dimension to invalid.
    pub failed_codes: Vec<String>,
}

impl Validation {
    /// The raw-scan placeholder: nothing was validated.
    pub fn fallback() -> Self {
        Self {
            source: ValidationSource::Fallback,
            state: "unknown".into(),
            integrity: Check::Unknown,
            signature: Check::Unknown,
            signer_trust: SignerTrust::Unknown,
            signer_validity: SignerValidity::Unknown,
            codes: Vec::new(),
            failed_codes: Vec::new(),
        }
    }

    /// A failed binding, signature, or credential.
    pub fn is_invalid(&self) -> bool {
        self.integrity == Check::Invalid
            || self.signature == Check::Invalid
            || self.signer_validity == SignerValidity::Invalid
    }
}

/// Normalized C2PA evidence; serializes exactly like Python's `c2pa_info` dict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C2paInfo {
    pub c2pa_manifest: String,
    pub validation: Validation,
    pub issuer: Option<String>,
    pub ai_tool: Option<String>,
    pub actions: Option<String>,
    pub c2pa_identity_ai: bool,
    pub ai_disclosure: Option<String>,
    pub human_oversight: Option<String>,
    pub source_type: Option<String>,
    pub ai_source_kind: Option<AiSourceKind>,
    pub watermarked: bool,
    pub synthid_vendors: Vec<String>,
    pub synthid_watermark: Option<String>,
    pub soft_binding_vendors: Vec<String>,
    pub soft_binding: Option<String>,
    pub soft_binding_algorithm: Option<String>,
    pub soft_binding_value: Option<String>,
    pub claim_generator: Option<String>,
    pub c2pa_spec: Option<String>,
    pub timestamp: Option<String>,
    pub timestamps: Vec<String>,
}

pub const C2PA_TYPE: &str = "C2PA (Coalition for Content Provenance and Authenticity)";

impl C2paInfo {
    /// Base evidence for a manifest of `byte_count` bytes.
    pub fn new(byte_count: usize, validation: Validation) -> Self {
        let container = match validation.source {
            ValidationSource::Reader => "C2PA manifest store",
            ValidationSource::Fallback => "C2PA manifest",
        };
        Self {
            c2pa_manifest: format!("{container} ({byte_count} bytes)"),
            validation,
            issuer: None,
            ai_tool: None,
            actions: None,
            c2pa_identity_ai: false,
            ai_disclosure: None,
            human_oversight: None,
            source_type: None,
            ai_source_kind: None,
            watermarked: false,
            synthid_vendors: Vec::new(),
            synthid_watermark: None,
            soft_binding_vendors: Vec::new(),
            soft_binding: None,
            soft_binding_algorithm: None,
            soft_binding_value: None,
            claim_generator: None,
            c2pa_spec: None,
            timestamp: None,
            timestamps: Vec::new(),
        }
    }

    /// `c2pa_info_has_invalid_credential`.
    pub fn has_invalid_credential(&self) -> bool {
        self.validation.is_invalid()
    }

    /// `c2pa_credential_level`; signer trust is deliberately not a condition.
    pub fn credential_level(&self) -> CredentialLevel {
        if self.has_invalid_credential() {
            CredentialLevel::Invalid
        } else if self.validation.integrity == Check::Valid && self.validation.signature == Check::Valid {
            CredentialLevel::Verified
        } else {
            CredentialLevel::Unverified
        }
    }

    /// `c2pa_info_has_invismark`: the manifest declares Microsoft InvisMark.
    pub fn has_invismark(&self) -> bool {
        self.soft_binding_vendors
            .iter()
            .any(|vendor| vendor == "Microsoft InvisMark")
            || self
                .soft_binding
                .as_deref()
                .is_some_and(|label| label.contains("Microsoft InvisMark"))
            || self
                .soft_binding_algorithm
                .as_deref()
                .is_some_and(|alg| alg.contains("com.microsoft.invismark"))
    }

    /// `c2pa_info_has_removal_hint`: an AI or watermark claim that keeps removal fail-safe.
    pub fn has_removal_hint(&self) -> bool {
        self.ai_source_kind.is_some()
            || self.c2pa_identity_ai
            || self.synthid_watermark.is_some()
            || self.has_invismark()
    }

    /// The evidence as an ordered JSON object.
    pub fn to_value(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or_default()
    }

    /// Field/value pairs as Python's `str(value)` renders them (lists become Python reprs).
    pub fn string_entries(&self) -> Vec<(&'static str, String)> {
        let mut out = Vec::new();
        for (key, value) in self.fields() {
            let text = match value {
                Field::Str(text) => text.to_owned(),
                Field::Bool(flag) => if flag { "True" } else { "False" }.to_owned(),
                Field::List(items) => {
                    let quoted: Vec<String> = items.iter().map(|item| py_str_repr(item)).collect();
                    format!("[{}]", quoted.join(", "))
                }
            };
            out.push((key, text));
        }
        out
    }

    fn fields(&self) -> Vec<(&'static str, Field<'_>)> {
        let v = &self.validation;
        let mut fields = vec![
            ("has_c2pa", Field::Bool(true)),
            ("type", Field::Str(C2PA_TYPE)),
            ("c2pa_manifest", Field::Str(&self.c2pa_manifest)),
        ];
        let validation = [
            ("c2pa_validation_source", Field::Str(v.source.as_str())),
            ("c2pa_validation_state", Field::Str(&v.state)),
            ("c2pa_integrity", Field::Str(v.integrity.as_str())),
            ("c2pa_signature", Field::Str(v.signature.as_str())),
            ("c2pa_signer_trust", Field::Str(v.signer_trust.as_str())),
            ("c2pa_signer_validity", Field::Str(v.signer_validity.as_str())),
            ("c2pa_validation_codes", Field::List(&v.codes)),
            ("c2pa_failed_codes", Field::List(&v.failed_codes)),
        ];
        let reader = v.source == ValidationSource::Reader;
        if !reader {
            fields.extend(validation);
        }
        let mut owned: Vec<(&'static str, Field<'_>)> = Vec::new();
        push_str(&mut owned, "issuer", self.issuer.as_deref());
        push_str(&mut owned, "ai_tool", self.ai_tool.as_deref());
        push_str(&mut owned, "actions", self.actions.as_deref());
        if self.c2pa_identity_ai {
            owned.push(("c2pa_identity_ai", Field::Bool(true)));
        }
        if reader {
            push_str(&mut owned, "ai_disclosure", self.ai_disclosure.as_deref());
            push_str(&mut owned, "human_oversight", self.human_oversight.as_deref());
        }
        push_str(&mut owned, "source_type", self.source_type.as_deref());
        push_str(
            &mut owned,
            "ai_source_kind",
            self.ai_source_kind.map(AiSourceKind::as_str),
        );
        if !reader {
            push_str(&mut owned, "ai_disclosure", self.ai_disclosure.as_deref());
            push_str(&mut owned, "human_oversight", self.human_oversight.as_deref());
        }
        if self.watermarked {
            owned.push(("watermarked", Field::Bool(true)));
        }
        if !self.synthid_vendors.is_empty() {
            owned.push(("synthid_vendors", Field::List(&self.synthid_vendors)));
        }
        push_str(&mut owned, "synthid_watermark", self.synthid_watermark.as_deref());
        if !self.soft_binding_vendors.is_empty() {
            owned.push(("soft_binding_vendors", Field::List(&self.soft_binding_vendors)));
        }
        push_str(&mut owned, "soft_binding", self.soft_binding.as_deref());
        push_str(
            &mut owned,
            "soft_binding_algorithm",
            self.soft_binding_algorithm.as_deref(),
        );
        push_str(
            &mut owned,
            "soft_binding_value",
            self.soft_binding_value.as_deref(),
        );
        fields.extend(owned);
        if reader {
            fields.extend(validation);
        }
        push_str(&mut fields, "claim_generator", self.claim_generator.as_deref());
        push_str(&mut fields, "c2pa_spec", self.c2pa_spec.as_deref());
        push_str(&mut fields, "timestamp", self.timestamp.as_deref());
        if !self.timestamps.is_empty() {
            fields.push(("timestamps", Field::List(&self.timestamps)));
        }
        fields
    }
}

fn push_str<'a>(fields: &mut Vec<(&'static str, Field<'a>)>, key: &'static str, value: Option<&'a str>) {
    if let Some(value) = value {
        fields.push((key, Field::Str(value)));
    }
}

/// Python's `repr(str)` for the plain strings evidence lists hold.
fn py_str_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

#[derive(Debug, Clone, Copy)]
enum Field<'a> {
    Str(&'a str),
    Bool(bool),
    List(&'a [String]),
}

impl Serialize for C2paInfo {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let fields = self.fields();
        let mut map = serializer.serialize_map(Some(fields.len()))?;
        for (key, value) in fields {
            match value {
                Field::Str(text) => map.serialize_entry(key, text)?,
                Field::Bool(flag) => map.serialize_entry(key, &flag)?,
                Field::List(items) => map.serialize_entry(key, items)?,
            }
        }
        map.end()
    }
}
