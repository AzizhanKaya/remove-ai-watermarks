//! Local generation backends behind a trait, and the guarded local passage rewriter.
//!
//! The Python package drove Transformers/MLX directly. Here a [`TextGenerator`] supplies
//! prompt rendering and sampling (burn on GPU, or a local Ollama server); the retry,
//! masking and acceptance policy stays in [`LocalRewriter`].

use serde::Serialize;

use crate::error::TextError;
use crate::guard::{
    LanguageClassifier, TEXT_REWRITE_SYSTEM_PROMPT, guard_local_rewrite_candidate, mask_local_rewrite_facts,
};

/// The default local rewriter checkpoint.
pub const DEFAULT_TEXT_REWRITER_MODEL: &str = "Qwen/Qwen2.5-1.5B-Instruct";

/// A chat role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

/// One chat message, serialized as `{"role": ..., "content": ...}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }
}

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// A configured end-of-sequence token or natural stop.
    Stop,
    /// The token budget ran out: the text may be truncated.
    Length,
}

/// One sampled continuation, special tokens removed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Generation {
    pub text: String,
    pub finish_reason: FinishReason,
}

/// A sampling request for one attempt.
#[derive(Debug, Clone, Copy)]
pub struct GenerationRequest<'a> {
    /// System and user messages; render them with the model's chat template when it has one.
    pub messages: &'a [ChatMessage],
    /// The prompt to use when the model has no chat template (special tokens added by the tokenizer).
    pub plain_prompt: &'a str,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,
}

/// A local causal LM: renders prompts, counts their tokens and samples continuations.
///
/// Implementations must render a configured chat template once without adding special tokens a
/// second time, report [`FinishReason::Length`] when the budget ends without an EOS token, and
/// never route through a watermarking remote service.
pub trait TextGenerator {
    /// The checkpoint label reported in results.
    fn model_id(&self) -> &str;

    /// Prompt length in tokens, or `None` when the backend cannot tokenize locally.
    fn prompt_token_count(
        &mut self,
        messages: &[ChatMessage],
        plain_prompt: &str,
    ) -> Result<Option<usize>, TextError>;

    /// Sample one continuation for `request`.
    fn generate(&mut self, request: &GenerationRequest<'_>) -> Result<Generation, TextError>;
}

/// Rewrites one prose passage (with `[[RAIW_SPAN_k]]` placeholders for inline locked spans).
pub trait PassageRewriter {
    fn rewrite(&mut self, passage: &str) -> Result<String, TextError>;
}

impl<F: FnMut(&str) -> Result<String, TextError>> PassageRewriter for F {
    fn rewrite(&mut self, passage: &str) -> Result<String, TextError> {
        self(passage)
    }
}

/// A `(source, candidate) -> accept` acceptance callback.
pub type RewriteValidator<'a> = Box<dyn FnMut(&str, &str) -> Result<bool, TextError> + 'a>;

/// The guarded local rewriter (`_build_local_lm_rewriter`): mask facts, sample up to three
/// seeds, reject truncated or unacceptable candidates, restore the facts.
pub struct LocalRewriter<'a> {
    generator: &'a mut dyn TextGenerator,
    seed: u64,
    validator: Option<RewriteValidator<'a>>,
    language: &'a dyn LanguageClassifier,
}

impl std::fmt::Debug for LocalRewriter<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalRewriter")
            .field("model_id", &self.generator.model_id())
            .field("seed", &self.seed)
            .finish_non_exhaustive()
    }
}

impl<'a> LocalRewriter<'a> {
    pub fn new(
        generator: &'a mut dyn TextGenerator,
        seed: u64,
        validator: Option<RewriteValidator<'a>>,
        language: &'a dyn LanguageClassifier,
    ) -> Self {
        Self {
            generator,
            seed,
            validator,
            language,
        }
    }

    pub fn model_id(&self) -> &str {
        self.generator.model_id()
    }
}

impl PassageRewriter for LocalRewriter<'_> {
    fn rewrite(&mut self, passage: &str) -> Result<String, TextError> {
        let (masked, facts) = mask_local_rewrite_facts(passage);
        let messages = [
            ChatMessage::system(TEXT_REWRITE_SYSTEM_PROMPT),
            ChatMessage::user(masked.clone()),
        ];
        let plain_prompt =
            format!("{TEXT_REWRITE_SYSTEM_PROMPT}\n\nPassage:\n{masked}\n\nRewritten passage:\n");
        let prompt_tokens = self.generator.prompt_token_count(&messages, &plain_prompt)?;
        let max_new_tokens = prompt_tokens.map_or(1024, |tokens| (tokens * 2).clamp(64, 1024));
        for attempt in 0..3 {
            let request = GenerationRequest {
                messages: &messages,
                plain_prompt: &plain_prompt,
                max_new_tokens,
                temperature: 0.7,
                top_p: 0.9,
                seed: self.seed + attempt,
            };
            let generation = self.generator.generate(&request)?;
            if generation.finish_reason != FinishReason::Stop {
                continue;
            }
            let validator = self
                .validator
                .as_mut()
                .map(|validator| validator.as_mut() as &mut dyn FnMut(&str, &str) -> Result<bool, TextError>);
            if let Some(checked) = guard_local_rewrite_candidate(
                passage,
                &masked,
                &facts,
                &generation.text,
                validator,
                self.language,
            )? {
                return Ok(checked);
            }
        }
        Err(TextError::runtime(
            "Text rewrite failed completeness, extent, fact, script, language, placeholder, non-identity or validator checks after 3 attempts.",
        ))
    }
}
