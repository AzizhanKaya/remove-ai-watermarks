//! AI generator attribution from text values and XMP (`generator_from_metadata`).

use std::sync::LazyLock;

use raiw_container::markers::MAX_TC260_VALUE_BYTES;
use regex::bytes::Regex;

use super::app::{app_generator_bytes, app_generator_str};
use super::keys::has_generator_token;
use crate::py::{latin1, strip};
use crate::pyjson::{self, PyValue};

static XMP_GENERATOR: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?-u)(?:CreatorTool|photoshop:Credit)[>"'=\s]{1,4}([^<"']{1,80})"#)
        .unwrap_or_else(|err| unreachable!("static regex: {err}"))
});

/// `_structured_generator_from_metadata`: a recognized `generationParams.modelName`.
pub fn structured_generator(value: &str) -> Option<String> {
    if value.chars().count() as u64 > MAX_TC260_VALUE_BYTES || !value.trim_start().starts_with('{') {
        return None;
    }
    let parsed = pyjson::loads(value)?;
    let PyValue::Str(model) = parsed.get("generationParams")?.get("modelName")? else {
        return None;
    };
    let normalized = strip(model);
    if normalized.is_empty() || model.chars().count() > 200 {
        return None;
    }
    has_generator_token(&normalized.to_lowercase()).then(|| normalized.to_owned())
}

/// The generator one text value names, in the order the candidate loop tests them.
fn generator_in_value(value: &str) -> Option<String> {
    if let Some(app) = app_generator_str(value) {
        return Some(app.to_owned());
    }
    if let Some(structured) = structured_generator(value) {
        return Some(structured);
    }
    has_generator_token(&value.to_lowercase()).then(|| strip(value).to_owned())
}

/// `generator_from_metadata(candidates, scan)`.
pub fn generator_from_metadata<S: AsRef<str>>(candidates: &[S], scan: &[u8]) -> Option<String> {
    if !scan.is_empty()
        && let Some(app) = app_generator_bytes(scan)
    {
        return Some(app.to_owned());
    }
    candidates
        .iter()
        .find_map(|value| generator_in_value(value.as_ref()))
        .or_else(|| {
            XMP_GENERATOR
                .captures_iter(scan)
                .find_map(|caps| generator_in_value(&latin1(&caps[1])))
        })
}

/// `generator_from_metadata((), payload) is not None`.
pub fn names_ai_generator(payload: &[u8]) -> bool {
    generator_from_metadata::<&str>(&[], payload).is_some()
}
