//! `SourceClassification`, serialized exactly like Python's `SourceClassification.to_dict()`.

use indexmap::IndexMap;
use serde::Serialize;

use crate::consts::{CLAIM, MODEL_SCHEMA, PROVIDERS, UNKNOWN};

/// Provider name to score, in `ALL_PROVIDER_CLASSES` order.
pub type Scores = IndexMap<String, f64>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceReason {
    Classified,
    Abstained,
    FeatureUnavailable,
}

impl SourceReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Classified => "classified",
            Self::Abstained => "abstained",
            Self::FeatureUnavailable => "feature_unavailable",
        }
    }
}

/// Result of abstaining 10-provider source-pipeline classification.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SourceClassification {
    pub label: &'static str,
    pub version_subfamily: Option<String>,
    pub fine_subfamily: Option<String>,
    pub reason: SourceReason,
    pub confidence: f64,
    pub pooled_scores: Scores,
    pub margins: Scores,
    pub thresholds: Scores,
    pub watermark_truth: &'static str,
    pub claim: &'static str,
    pub model_revision: &'static str,
    pub synthetic_detected: bool,
    pub synthetic_probability: f64,
}

impl SourceClassification {
    /// The `feature_unavailable` result for small images or degenerate feature vectors.
    pub fn unavailable() -> Self {
        Self {
            label: UNKNOWN,
            version_subfamily: None,
            fine_subfamily: None,
            reason: SourceReason::FeatureUnavailable,
            confidence: 0.0,
            pooled_scores: Scores::new(),
            margins: Scores::new(),
            thresholds: Scores::new(),
            watermark_truth: UNKNOWN,
            claim: CLAIM,
            model_revision: MODEL_SCHEMA,
            synthetic_detected: false,
            synthetic_probability: 0.0,
        }
    }

    /// Provider pooled scores (Python's `scores` property).
    pub fn scores(&self) -> &Scores {
        &self.pooled_scores
    }

    pub fn is_classified(&self) -> bool {
        self.label != UNKNOWN
    }

    /// `json.dumps(result.to_dict(), indent=2)`, the `classify-source --json` output.
    pub fn to_json_pretty(&self) -> String {
        raiw_core::to_py_json(self, raiw_core::JsonStyle::PRETTY)
    }

    pub fn to_value(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

/// Provider-keyed map from a per-class value slice.
pub(crate) fn provider_scores(values: impl IntoIterator<Item = f64>) -> Scores {
    PROVIDERS.iter().map(|p| (*p).to_owned()).zip(values).collect()
}

/// Map a provider index back to its static label.
pub(crate) fn provider_label(index: usize) -> &'static str {
    PROVIDERS.get(index).copied().unwrap_or(UNKNOWN)
}
