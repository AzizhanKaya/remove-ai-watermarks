//! Explicitly configured Chat Completions rewriting (port of `text_api.py`).
//!
//! Construction never touches the network; each call makes exactly one POST. There are no
//! default endpoints, credentials or remote fallbacks.

pub mod chat;
pub mod document;
pub mod ollama;
mod url;

use crate::error::TextError;
use crate::generator::{ChatMessage, PassageRewriter};
use crate::guard::{
    LanguageClassifier, TEXT_REWRITE_SYSTEM_PROMPT, guard_local_rewrite_candidate, mask_local_rewrite_facts,
};

pub use chat::{ChatCompletionClient, ChatCompletionConfig, CompletionTokenField};
pub use document::{
    DocumentApiRewriter, FactValidator, RetainedSentenceRefiner, RewriteStyle, StrictResidualConfig,
    StrictResidualRewriter, VetoedSentenceRestorer, replace_selected_sentences, sentence_slots,
};
pub use ollama::{
    DEFAULT_OLLAMA_URL, OllamaDocumentRewriter, OllamaGenerator, OllamaRoute, validate_loopback_url,
};

/// A guarded passage rewriter over an explicit endpoint (`build_text_api_rewriter`).
///
/// Masking, the shared system prompt, restoration and the local acceptance guard all apply;
/// a guard failure raises instead of returning the candidate.
pub struct ApiPassageRewriter<'a> {
    client: ChatCompletionClient,
    language: &'a dyn LanguageClassifier,
}

impl std::fmt::Debug for ApiPassageRewriter<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApiPassageRewriter")
            .field("client", &self.client)
            .finish_non_exhaustive()
    }
}

impl<'a> ApiPassageRewriter<'a> {
    pub fn new(
        config: ChatCompletionConfig,
        language: &'a dyn LanguageClassifier,
    ) -> Result<Self, TextError> {
        Ok(Self {
            client: ChatCompletionClient::new(config)?,
            language,
        })
    }
}

impl PassageRewriter for ApiPassageRewriter<'_> {
    fn rewrite(&mut self, passage: &str) -> Result<String, TextError> {
        let (masked, facts) = mask_local_rewrite_facts(passage);
        let messages = [
            ChatMessage::system(TEXT_REWRITE_SYSTEM_PROMPT),
            ChatMessage::user(masked.clone()),
        ];
        let candidate = self.client.complete(&messages, None)?;
        if let Some(checked) =
            guard_local_rewrite_candidate(passage, &masked, &facts, &candidate, None, self.language)?
        {
            return Ok(checked);
        }
        tracing::warn!("Text API candidate failed the shared local acceptance guard.");
        Err(TextError::runtime(
            "Text API rewrite failed the shared local acceptance checks.",
        ))
    }
}
