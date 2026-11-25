//! Local Ollama backends: a [`TextGenerator`] for passage rewriting and the CLI document route.

use serde_json::json;

use crate::api::chat::{
    ChatCompletionClient, ChatCompletionConfig, CompletionTokenField, completion_with_reason,
};
use crate::api::document::{
    DocumentApiRewriter, FactValidator, RewriteStyle, StrictResidualConfig, StrictResidualRewriter,
};
use crate::api::url::split_url;
use crate::error::TextError;
use crate::generator::{ChatMessage, FinishReason, Generation, GenerationRequest, TextGenerator};
use crate::rewrite::{DocumentRewriter, InvisibleOptions};
use crate::unicode;

/// Ollama's OpenAI-compatible endpoint on the loopback interface.
pub const DEFAULT_OLLAMA_URL: &str = "http://127.0.0.1:11434/v1";

/// Require an `http://` URL whose host is `127.0.0.1` or `::1`.
pub fn validate_loopback_url(base_url: &str) -> Result<(), TextError> {
    let error = || TextError::invalid("--ollama-url must be an HTTP URL on a loopback address");
    let parsed = split_url(base_url).map_err(|()| error())?;
    if parsed.scheme != "http" || !matches!(parsed.hostname.as_deref(), Some("127.0.0.1" | "::1")) {
        return Err(error());
    }
    Ok(())
}

/// A local Ollama model as a [`TextGenerator`] (seeded nucleus sampling, no remote fallback).
#[derive(Debug, Clone)]
pub struct OllamaGenerator {
    client: ChatCompletionClient,
}

impl OllamaGenerator {
    /// `base_url` defaults to [`DEFAULT_OLLAMA_URL`]; it must be a loopback HTTP URL.
    pub fn new(base_url: Option<&str>, model: &str) -> Result<Self, TextError> {
        let base_url = base_url.unwrap_or(DEFAULT_OLLAMA_URL);
        validate_loopback_url(base_url)?;
        let config = ChatCompletionConfig {
            completion_token_field: CompletionTokenField::MaxTokens,
            timeout: 185.0,
            ..ChatCompletionConfig::new(base_url, model)
        };
        Ok(Self {
            client: ChatCompletionClient::new(config)?,
        })
    }
}

impl TextGenerator for OllamaGenerator {
    fn model_id(&self) -> &str {
        &self.client.config().model
    }

    fn prompt_token_count(
        &mut self,
        _messages: &[ChatMessage],
        _plain_prompt: &str,
    ) -> Result<Option<usize>, TextError> {
        Ok(None)
    }

    fn generate(&mut self, request: &GenerationRequest<'_>) -> Result<Generation, TextError> {
        let mut payload = self.client.request(request.messages, None);
        payload.insert("max_tokens".into(), json!(request.max_new_tokens));
        payload.insert("temperature".into(), json!(request.temperature));
        payload.insert("top_p".into(), json!(request.top_p));
        payload.insert("seed".into(), json!(request.seed));
        let (_, body) = self.client.post(&payload)?;
        let (text, truncated) = completion_with_reason(&body)?;
        Ok(Generation {
            text,
            finish_reason: if truncated {
                FinishReason::Length
            } else {
                FinishReason::Stop
            },
        })
    }
}

/// The CLI's default Ollama document rewriter: refuse unchanged slots unless partial output is allowed.
#[derive(Debug, Clone)]
pub struct OllamaDocumentRewriter {
    inner: DocumentApiRewriter,
    allow_partial: bool,
}

impl DocumentRewriter for OllamaDocumentRewriter {
    fn rewrite_document(&mut self, prose: &[String]) -> Result<Vec<String>, TextError> {
        let candidate = self.inner.rewrite(prose, None)?;
        if !self.allow_partial && prose.iter().zip(&candidate).any(|(left, right)| left == right) {
            return Err(TextError::runtime(
                "Local Ollama returned an unchanged prose span. Use --allow-partial to retain it.",
            ));
        }
        Ok(candidate)
    }
}

/// The `--rewriter-backend ollama` route of `text invisible` / `text all --invisible`.
#[derive(Debug, Clone)]
pub enum OllamaRoute {
    Document {
        rewriter: OllamaDocumentRewriter,
        validator: FactValidator,
    },
    StrictResidual(StrictResidualRewriter),
}

impl OllamaRoute {
    /// Validate the CLI settings and build the callbacks (`_text_rewrite_options`).
    pub fn new(
        base_url: Option<&str>,
        model: Option<&str>,
        seed: i64,
        allow_partial: bool,
        strict_residual: bool,
    ) -> Result<Self, TextError> {
        let model = model
            .filter(|model| !unicode::strip(model).is_empty())
            .ok_or_else(|| {
                TextError::invalid("--rewriter-backend ollama requires an explicit --rewriter-model")
            })?;
        if strict_residual && allow_partial {
            return Err(TextError::invalid(
                "--strict-residual cannot be combined with --allow-partial",
            ));
        }
        let base_url = base_url.unwrap_or(DEFAULT_OLLAMA_URL);
        validate_loopback_url(base_url)?;
        if strict_residual {
            let config = StrictResidualConfig {
                seed,
                completion_token_field: CompletionTokenField::MaxTokens,
                ..StrictResidualConfig::new(base_url, model)
            };
            return Ok(Self::StrictResidual(StrictResidualRewriter::new(&config)?));
        }
        let config = ChatCompletionConfig {
            max_completion_tokens: 2048,
            completion_token_field: CompletionTokenField::MaxTokens,
            timeout: 185.0,
            temperature: Some(0.0),
            seed: Some(seed),
            disable_reasoning: true,
            ..ChatCompletionConfig::new(base_url, model)
        };
        let inner = DocumentApiRewriter::new(config, RewriteStyle::Sentence, true)?;
        Ok(Self::Document {
            rewriter: OllamaDocumentRewriter { inner, allow_partial },
            validator: FactValidator,
        })
    }

    /// Wire the callbacks into regeneration options.
    pub fn options(&mut self) -> InvisibleOptions<'_> {
        match self {
            Self::Document { rewriter, validator } => InvisibleOptions {
                document_rewriter: Some(rewriter),
                document_validator: Some(validator),
                ..InvisibleOptions::default()
            },
            Self::StrictResidual(strict) => InvisibleOptions {
                document_rewriter: Some(&mut strict.rewriter),
                document_validator: Some(&mut strict.validator),
                document_repairer: Some(&mut strict.repairer),
                document_refiner: Some(&mut strict.refiner),
                ..InvisibleOptions::default()
            },
        }
    }
}
