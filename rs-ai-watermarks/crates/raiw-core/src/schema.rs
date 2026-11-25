//! Validation for versioned JSON transport contracts.

use serde_json::Value;

#[derive(Debug, thiserror::Error)]
#[error("Unsupported {contract} schema: {value}; supported versions: {versions}")]
pub struct UnsupportedSchema {
    pub contract: String,
    pub value: String,
    pub versions: String,
}

/// Return an explicitly supported integer schema version or fail (booleans are not integers).
pub fn require_schema_version(
    value: &Value,
    contract: &str,
    supported: &[i64],
) -> Result<i64, UnsupportedSchema> {
    if let Some(version) = value.as_i64().filter(|_| value.is_i64() || value.is_u64())
        && supported.contains(&version)
    {
        return Ok(version);
    }
    let mut sorted = supported.to_vec();
    sorted.sort_unstable();
    let versions = sorted
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    Err(UnsupportedSchema {
        contract: contract.into(),
        value: py_repr(value),
        versions,
    })
}

fn py_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::String(text) => format!("'{text}'"),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn accepts_supported_integers_only() {
        assert_eq!(require_schema_version(&json!(2), "record", &[1, 2]).unwrap(), 2);
        let err = require_schema_version(&json!(true), "record", &[2, 1]).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Unsupported record schema: True; supported versions: 1, 2"
        );
        assert!(require_schema_version(&json!(2.0), "record", &[2]).is_err());
    }
}
