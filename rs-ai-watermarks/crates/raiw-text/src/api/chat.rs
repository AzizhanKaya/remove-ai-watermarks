//! The single-request Chat Completions transport (`build_chat_completion_client`).

use std::time::Duration;

use raiw_core::{JsonStyle, to_py_json};
use serde_json::{Map, Value, json};

use crate::api::url::split_url;
use crate::error::TextError;
use crate::generator::ChatMessage;
use crate::unicode;

/// Which request field carries the completion budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompletionTokenField {
    #[default]
    MaxCompletionTokens,
    MaxTokens,
}

impl CompletionTokenField {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MaxCompletionTokens => "max_completion_tokens",
            Self::MaxTokens => "max_tokens",
        }
    }
}

/// Explicit endpoint configuration; nothing is read from the environment.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatCompletionConfig {
    pub base_url: String,
    pub model: String,
    pub api_key: Option<String>,
    pub max_completion_tokens: u32,
    /// Seconds.
    pub timeout: f64,
    pub completion_token_field: CompletionTokenField,
    pub temperature: Option<f64>,
    pub seed: Option<i64>,
    pub disable_reasoning: bool,
}

impl ChatCompletionConfig {
    /// The Python defaults: 1024 completion tokens, a 60 s timeout, no sampling overrides.
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            model: model.into(),
            api_key: None,
            max_completion_tokens: 1024,
            timeout: 60.0,
            completion_token_field: CompletionTokenField::MaxCompletionTokens,
            temperature: None,
            seed: None,
            disable_reasoning: false,
        }
    }
}

/// One POST per call, no retries or redirects; credentials are redacted from every log line.
#[derive(Debug, Clone)]
pub struct ChatCompletionClient {
    config: ChatCompletionConfig,
    endpoint: String,
    agent: ureq::Agent,
}

fn invalid_url() -> TextError {
    TextError::invalid("base_url must be an HTTP(S) URL without credentials, query or fragment.")
}

impl ChatCompletionClient {
    pub fn new(config: ChatCompletionConfig) -> Result<Self, TextError> {
        let parsed = split_url(&config.base_url).map_err(|()| invalid_url())?;
        if !matches!(parsed.scheme.as_str(), "http" | "https")
            || parsed.hostname.is_none()
            || parsed.has_userinfo
            || !parsed.query.is_empty()
            || !parsed.fragment.is_empty()
        {
            return Err(invalid_url());
        }
        if unicode::strip(&config.model).is_empty() {
            return Err(TextError::invalid("model must be a nonempty string."));
        }
        if let Some(key) = &config.api_key
            && (key.is_empty()
                || !key.is_ascii()
                || !unicode::str_is_printable(key)
                || key.chars().any(unicode::is_space))
        {
            return Err(TextError::invalid(
                "api_key must be a nonempty ASCII token without whitespace, or None.",
            ));
        }
        if config.max_completion_tokens == 0 {
            return Err(TextError::invalid(
                "max_completion_tokens must be a positive integer.",
            ));
        }
        if !config.timeout.is_finite() || config.timeout <= 0.0 {
            return Err(TextError::invalid("timeout must be a positive finite number."));
        }
        if config
            .temperature
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(TextError::invalid(
                "temperature must be a nonnegative finite number.",
            ));
        }
        if !cfg!(feature = "langid") {
            return Err(TextError::runtime(
                "API text rewriting requires 'remove-ai-watermarks[text-api]'.",
            ));
        }
        let endpoint = format!("{}/chat/completions", parsed.base().trim_end_matches('/'));
        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs_f64(config.timeout)))
            .max_redirects(0)
            .http_status_as_error(false)
            .proxy(None)
            .build()
            .into();
        Ok(Self {
            config,
            endpoint,
            agent,
        })
    }

    pub fn config(&self) -> &ChatCompletionConfig {
        &self.config
    }

    fn secrets(&self) -> Vec<String> {
        let Some(key) = &self.config.api_key else {
            return Vec::new();
        };
        let escaped = to_py_json(key, JsonStyle::COMPACT);
        let escaped = escaped[1..escaped.len() - 1].to_owned();
        let mut secrets = vec![key.clone()];
        if escaped != *key {
            secrets.push(escaped);
        }
        secrets.sort_by_key(|secret| std::cmp::Reverse(secret.len()));
        secrets
    }

    /// Replace the configured credential (raw or JSON-escaped) with `[redacted]`.
    pub(crate) fn redact(&self, value: &str) -> String {
        let secrets = self.secrets();
        if secrets.is_empty() {
            return value.to_owned();
        }
        let mut value = value.to_owned();
        if !secrets.iter().any(|secret| value.contains(secret.as_str()))
            && let Ok(decoded) = serde_json::from_str::<Value>(&value)
        {
            let normalized = to_py_json(&decoded, JsonStyle::UNICODE);
            if secrets.iter().any(|secret| normalized.contains(secret.as_str())) {
                value = normalized;
            }
        }
        for secret in &secrets {
            value = value.replace(secret.as_str(), "[redacted]");
        }
        value
    }

    fn warn_failure(&self, payload: &str, reason: &str, status: Option<u16>, body: &str) {
        let status = status.map_or_else(|| "None".to_owned(), |status| status.to_string());
        tracing::warn!(
            "Text API failure endpoint={} payload={} status={} body={} reason={}",
            self.redact(&self.endpoint),
            self.redact(payload),
            status,
            self.redact(body),
            self.redact(reason)
        );
    }

    /// The standard request object for `messages`.
    pub(crate) fn request(
        &self,
        messages: &[ChatMessage],
        response_format: Option<&Value>,
    ) -> Map<String, Value> {
        let mut request = Map::new();
        request.insert("model".into(), json!(self.config.model));
        request.insert("messages".into(), json!(messages));
        request.insert(
            self.config.completion_token_field.as_str().into(),
            json!(self.config.max_completion_tokens),
        );
        request.insert("stream".into(), json!(false));
        if let Some(temperature) = self.config.temperature {
            request.insert("temperature".into(), json!(temperature));
        }
        if let Some(seed) = self.config.seed {
            request.insert("seed".into(), json!(seed));
        }
        if self.config.disable_reasoning {
            request.insert("reasoning_effort".into(), json!("none"));
        }
        if let Some(format) = response_format {
            request.insert("response_format".into(), format.clone());
        }
        request
    }

    /// POST `request` and return the UTF-8 body of a 2xx response that does not echo the key.
    pub(crate) fn post(&self, request: &Map<String, Value>) -> Result<(String, String), TextError> {
        let payload = to_py_json(request, JsonStyle::UNICODE);
        tracing::info!(
            "Text API request endpoint={} payload={}",
            self.redact(&self.endpoint),
            self.redact(&payload)
        );
        let mut call = self
            .agent
            .post(&self.endpoint)
            .header("Content-Type", "application/json");
        if let Some(key) = &self.config.api_key {
            call = call.header("Authorization", format!("Bearer {key}"));
        }
        let response = call.send(payload.as_bytes());
        let mut response = match response {
            Ok(response) => response,
            Err(error) => {
                self.warn_failure(&payload, &format!("{error}"), None, "");
                return Err(TextError::runtime("Text API request failed."));
            }
        };
        let status = response.status().as_u16();
        let raw = match response
            .body_mut()
            .with_config()
            .limit(64 * 1024 * 1024)
            .read_to_vec()
        {
            Ok(raw) => raw,
            Err(error) => {
                self.warn_failure(&payload, &format!("{error}"), None, "");
                return Err(TextError::runtime("Text API request failed."));
            }
        };
        let body = String::from_utf8_lossy(&raw).into_owned();
        tracing::info!("Text API response status={} body={}", status, self.redact(&body));
        if !(200..300).contains(&status) {
            self.warn_failure(&payload, "HTTP error", Some(status), &body);
            return Err(TextError::runtime(format!("Text API returned HTTP {status}.")));
        }
        if std::str::from_utf8(&raw).is_err() {
            self.warn_failure(&payload, "Invalid UTF-8", Some(status), &body);
            return Err(TextError::runtime("Text API returned invalid UTF-8."));
        }
        if self.redact(&body) != body {
            self.warn_failure(&payload, "Credential reflected in response", Some(status), &body);
            return Err(TextError::runtime(
                "Text API response contained the configured credential.",
            ));
        }
        Ok((payload, body))
    }

    /// Send `messages` and return the single naturally completed assistant text.
    pub fn complete(
        &self,
        messages: &[ChatMessage],
        response_format: Option<&Value>,
    ) -> Result<String, TextError> {
        let request = self.request(messages, response_format);
        let (payload, body) = self.post(&request)?;
        completion_text(&body)
            .inspect_err(|error| self.warn_failure(&payload, &error.to_string(), Some(200), &body))
    }
}

fn truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(flag)) => *flag,
        Some(Value::Number(number)) => number.as_f64().is_some_and(|value| value != 0.0),
        Some(Value::String(text)) => !text.is_empty(),
        Some(Value::Array(items)) => !items.is_empty(),
        Some(Value::Object(fields)) => !fields.is_empty(),
    }
}

fn single_choice(body: &str) -> Result<Map<String, Value>, TextError> {
    let decoded: Value =
        serde_json::from_str(body).map_err(|_| TextError::runtime("Text API returned invalid JSON."))?;
    let Value::Object(decoded) = decoded else {
        return Err(TextError::runtime(
            "Text API returned an invalid completion object.",
        ));
    };
    let Some(Value::Array(choices)) = decoded.get("choices") else {
        return Err(TextError::runtime(
            "Text API must return exactly one completion choice.",
        ));
    };
    if choices.len() != 1 {
        return Err(TextError::runtime(
            "Text API must return exactly one completion choice.",
        ));
    }
    match &choices[0] {
        Value::Object(choice) => Ok(choice.clone()),
        _ => Err(TextError::runtime(
            "Text API returned an invalid completion object.",
        )),
    }
}

fn assistant_content(choice: &Map<String, Value>) -> Result<String, TextError> {
    let Some(Value::Object(message)) = choice.get("message") else {
        return Err(TextError::runtime(
            "Text API returned an invalid completion object.",
        ));
    };
    if message.get("role").and_then(Value::as_str) != Some("assistant")
        || truthy(message.get("refusal"))
        || truthy(message.get("tool_calls"))
        || truthy(message.get("function_call"))
    {
        return Err(TextError::runtime(
            "Text API returned a refusal, tool call or unsupported message.",
        ));
    }
    match message.get("content") {
        Some(Value::String(content)) if !unicode::strip(content).is_empty() => Ok(content.clone()),
        _ => Err(TextError::runtime(
            "Text API did not return a nonempty text passage.",
        )),
    }
}

/// `_completion_text`: the one naturally stopped assistant message.
pub(crate) fn completion_text(body: &str) -> Result<String, TextError> {
    let choice = single_choice(body)?;
    if choice.get("finish_reason").and_then(Value::as_str) != Some("stop") {
        return Err(TextError::runtime(
            "Text API did not return a naturally completed passage.",
        ));
    }
    assistant_content(&choice)
}

/// Like [`completion_text`], but report a token-budget stop instead of failing on it.
pub(crate) fn completion_with_reason(body: &str) -> Result<(String, bool), TextError> {
    let choice = single_choice(body)?;
    let truncated = match choice.get("finish_reason").and_then(Value::as_str) {
        Some("stop") => false,
        Some("length") => true,
        _ => {
            return Err(TextError::runtime(
                "Text API did not return a naturally completed passage.",
            ));
        }
    };
    Ok((assistant_content(&choice)?, truncated))
}
