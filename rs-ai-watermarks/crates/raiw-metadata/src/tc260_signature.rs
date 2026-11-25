//! TC260 `SecurityData` label signatures (TC260-PG-202511A), port of `_internal/tc260_signature.py`
//! and the verification half of `_internal/sm2.py`.
//!
//! A producer may record SM3withSM2 signatures (`PubSD` entries of type `DS`), the public key
//! (`PubKey`) and content bindings in the label's `ReservedCode1`. Only the producer's label
//! signature is checked; the embedded key proves integrity unless it matches a pinned signer.

use serde_json::{Map, Value};
use sm2::dsa::signature::Verifier;
use sm2::dsa::{Signature, VerifyingKey};

use raiw_core::pyjson::{JsonStyle, to_py_json};
use raiw_core::tc260::TC260_PRODUCERS;

pub const SM3_WITH_SM2: &str = "1.2.156.10197.1.501";

/// GB/T 32918.2's default signer identity, used when a signature names none.
pub const DEFAULT_USER_ID: &str = "1234567812345678";

const LABEL_TBS_TYPES: [&str; 2] = ["Md", "LabelMataData"];
const PRODUCER_FIELDS: [&str; 3] = ["Label", "ContentProducer", "ProduceID"];

/// Outcome of the label-signature check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SignatureStatus {
    Verified,
    Failed,
    Absent,
    Unsupported,
}

impl SignatureStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Verified => "verified",
            Self::Failed => "failed",
            Self::Absent => "absent",
            Self::Unsupported => "unsupported",
        }
    }
}

/// Outcome of checking one label's `SecurityData` label signature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tc260Signature {
    pub label: SignatureStatus,
    pub signer: Option<&'static str>,
}

impl Tc260Signature {
    pub fn describe(&self) -> String {
        match (self.label, self.signer) {
            (SignatureStatus::Verified, Some(signer)) => {
                format!("TC260 label signature verified (pinned {signer} key)")
            }
            (SignatureStatus::Verified, None) => {
                "TC260 label signature verified with its own embedded key (no pinned signer)".into()
            }
            (SignatureStatus::Failed, _) => "TC260 label signature does not verify".into(),
            (SignatureStatus::Unsupported, _) => {
                "TC260 SecurityData signature algorithm or signed-content selection \
                 is not supported; integrity unverified"
                    .into()
            }
            (SignatureStatus::Absent, _) => "TC260 SecurityData without a label signature".into(),
        }
    }
}

// Keys

/// A candidate public key as SEC1 bytes, already validated on the curve.
#[derive(Debug, Clone)]
struct CandidateKey {
    sec1: Vec<u8>,
}

impl CandidateKey {
    fn parse(sec1: Vec<u8>) -> Option<Self> {
        VerifyingKey::from_sec1_bytes(DEFAULT_USER_ID, &sec1)
            .ok()
            .map(|_| Self { sec1 })
    }

    fn x(&self) -> &[u8] {
        &self.sec1[1..33]
    }

    fn verifies(&self, message: &[u8], signature: &[u8]) -> bool {
        let Ok(key) = VerifyingKey::from_sec1_bytes(DEFAULT_USER_ID, &self.sec1) else {
            return false;
        };
        let Ok(signature) = Signature::from_slice(signature) else {
            return false;
        };
        key.verify(message, &signature).is_ok()
    }
}

fn compressed(x: &[u8], odd: bool) -> Option<CandidateKey> {
    let mut sec1 = Vec::with_capacity(33);
    sec1.push(if odd { 3 } else { 2 });
    sec1.extend_from_slice(x);
    CandidateKey::parse(sec1).and_then(|key| uncompressed(&key))
}

fn uncompressed(key: &CandidateKey) -> Option<CandidateKey> {
    let verifying = VerifyingKey::from_sec1_bytes(DEFAULT_USER_ID, &key.sec1).ok()?;
    CandidateKey::parse(verifying.to_sec1_bytes().into_vec())
}

/// Pinned signer keys: the even-y point for each producer's registered x coordinate.
fn pinned_keys() -> impl Iterator<Item = (&'static str, CandidateKey)> {
    TC260_PRODUCERS
        .iter()
        .filter_map(|row| Some((row.signer?, compressed(&row.signing_key_x?, false)?)))
}

/// The points one `KeyValue` can denote, and the pinned signer when it matches one.
type KeyCandidates = (Vec<CandidateKey>, Option<&'static str>);

fn candidate_keys(key_hex: &str) -> KeyCandidates {
    let Ok(mut raw) = hex::decode(key_hex) else {
        return (Vec::new(), None);
    };
    if raw.len() == 65 && raw[0] == 4 {
        raw.remove(0);
    }
    if raw.len() == 33 && raw[0] == 0 {
        raw.remove(0); // MiniMax pads a bare x coordinate with 00
    }
    let points: Vec<CandidateKey> = match raw.len() {
        64 => {
            raw.insert(0, 4);
            CandidateKey::parse(raw).into_iter().collect()
        }
        33 if matches!(raw[0], 2 | 3) => compressed(&raw[1..], raw[0] == 3).into_iter().collect(),
        32 => [false, true]
            .into_iter()
            .filter_map(|odd| compressed(&raw, odd))
            .collect(),
        _ => return (Vec::new(), None),
    };
    for (signer, pinned) in pinned_keys() {
        if points.iter().any(|p| p.x() == pinned.x()) {
            return (vec![pinned], Some(signer));
        }
    }
    (points, None)
}

// Label message

/// Appendix A.1: `"Label":"1","ContentProducer":"...","ProduceID":"..."`.
fn label_message<'a>(field: &impl Fn(&str) -> Option<&'a str>) -> Vec<u8> {
    PRODUCER_FIELDS
        .iter()
        .map(|name| {
            let value = field(name).unwrap_or_default();
            format!(
                "{}:{}",
                to_py_json(name, JsonStyle::COMPACT),
                to_py_json(value, JsonStyle::UNICODE)
            )
        })
        .collect::<Vec<_>>()
        .join(",")
        .into_bytes()
}

fn object(value: Option<&Value>) -> Map<String, Value> {
    value.and_then(Value::as_object).cloned().unwrap_or_default()
}

fn objects(value: Option<&Value>) -> Vec<Map<String, Value>> {
    value
        .and_then(Value::as_array)
        .map(|items| items.iter().filter_map(|i| i.as_object().cloned()).collect())
        .unwrap_or_default()
}

/// Python `str(value)` for the JSON scalars a `KeyID`/`KeyValue` can hold.
fn py_str(value: Option<&Value>, default: &str) -> String {
    value.map_or_else(|| default.to_owned(), raiw_core::pyjson::py_str)
}

/// Check the producer's `SecurityData` in `ReservedCode1`; `None` when it has none.
pub fn check_tc260_signature<'a>(field: impl Fn(&str) -> Option<&'a str>) -> Option<Tc260Signature> {
    let parsed: Value = serde_json::from_str(field("ReservedCode1").unwrap_or_default()).ok()?;
    let top = object(Some(&parsed));
    let nested = object(top.get("SecurityData"));
    let security = if nested.is_empty() { top } else { nested };
    if security.get("Type").and_then(Value::as_str).map(str::trim) != Some("TC260PG") {
        return None;
    }
    let public = objects(security.get("PubSD").or_else(|| security.get("PubSd")));
    let mut keys: Vec<(String, KeyCandidates)> = Vec::new();
    for entry in public.iter().rev() {
        if entry.get("Type") == Some(&Value::from("PubKey"))
            && entry.get("AlgID") == Some(&Value::from(SM3_WITH_SM2))
        {
            let id = py_str(entry.get("KeyID"), "0");
            let parsed = candidate_keys(&py_str(entry.get("KeyValue"), ""));
            match keys.iter_mut().find(|(known, _)| *known == id) {
                Some(slot) => slot.1 = parsed,
                None => keys.push((id, parsed)),
            }
        }
    }
    let message = label_message(&field);
    let mut results = Vec::new();
    let mut unsupported = false;
    let mut signer = None;
    for entry in &public {
        if entry.get("Type") != Some(&Value::from("DS")) {
            continue;
        }
        let tbs_type = object(entry.get("TBSData"))
            .get("Type")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let label_tbs = tbs_type.as_deref().is_some_and(|t| LABEL_TBS_TYPES.contains(&t));
        if entry.get("AlgID") != Some(&Value::from(SM3_WITH_SM2)) || !label_tbs {
            unsupported = true;
            continue;
        }
        let id = py_str(entry.get("KeyID"), "0");
        let (points, key_signer) = keys
            .iter()
            .find(|(known, _)| *known == id)
            .map_or((&[][..], None), |(_, (p, s))| (&p[..], *s));
        let signature = hex::decode(py_str(entry.get("Signature"), "")).unwrap_or_default();
        let verified =
            signature.len() == 64 && points.iter().any(|point| point.verifies(&message, &signature));
        results.push(verified);
        if verified {
            signer = signer.or(key_signer);
        }
    }
    if !results.is_empty() {
        let label = if results.iter().any(|v| *v) {
            SignatureStatus::Verified
        } else {
            SignatureStatus::Failed
        };
        return Some(Tc260Signature { label, signer });
    }
    Some(Tc260Signature {
        label: if unsupported {
            SignatureStatus::Unsupported
        } else {
            SignatureStatus::Absent
        },
        signer: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPENSSL_X: &str = "393A2AD9E27E43ACAAAB85A19D3B85591E14406157175ADDD37BF77FF0CAF9ED";
    const OPENSSL_Y: &str = "37FCF3A8B4F55C038523261FCA6636AFE6A7AFA2A1482A606F5CB3A18B2E2A1E";

    #[test]
    fn bare_x_lifts_to_both_parities_and_full_points_parse() {
        let (points, signer) = candidate_keys(OPENSSL_X);
        assert_eq!(points.len(), 2);
        assert!(signer.is_none());
        let full = format!("04{OPENSSL_X}{OPENSSL_Y}");
        let (points, _) = candidate_keys(&full);
        assert_eq!(points.len(), 1);
        let off_curve = format!(
            "04{OPENSSL_X}{}",
            "37FCF3A8B4F55C038523261FCA6636AFE6A7AFA2A1482A606F5CB3A18B2E2A1F"
        );
        assert!(candidate_keys(&off_curve).0.is_empty());
        assert!(candidate_keys("zz").0.is_empty());
    }

    #[test]
    fn pinned_minimax_key_is_on_the_curve() {
        let pinned: Vec<_> = pinned_keys().collect();
        assert_eq!(pinned.len(), 1);
        assert_eq!(pinned[0].0, "MiniMax");
        let odd_spelling = format!("03{}", hex::encode(pinned[0].1.x()));
        let (points, signer) = candidate_keys(&odd_spelling);
        assert_eq!(signer, Some("MiniMax"));
        assert_eq!(points[0].x(), pinned[0].1.x());
    }

    #[test]
    fn label_message_follows_appendix_order() {
        let label = |k: &str| match k {
            "Label" => Some("1"),
            "ContentProducer" => Some("海螺"),
            _ => None,
        };
        assert_eq!(
            String::from_utf8(label_message(&label)).unwrap(),
            "\"Label\":\"1\",\"ContentProducer\":\"海螺\",\"ProduceID\":\"\""
        );
    }

    #[test]
    fn non_tc260_reserved_codes_are_ignored() {
        assert!(check_tc260_signature(|_| None).is_none());
        assert!(
            check_tc260_signature(|k| (k == "ReservedCode1").then_some("{\"Type\":\"other\"}")).is_none()
        );
        let absent = check_tc260_signature(|k| {
            (k == "ReservedCode1").then_some("{\"SecurityData\":{\"Type\":\" TC260PG \"}}")
        });
        assert_eq!(
            absent,
            Some(Tc260Signature {
                label: SignatureStatus::Absent,
                signer: None
            })
        );
    }
}
