//! Text provenance identification and removal (port of `remove_ai_watermarks.text`).
//!
//! Two layers:
//! - deterministic provenance and character steganography: C2PATXT, C2PA HTML/ASCII armor,
//!   VSRMark, Unicode Tags, zero-width/VS/PUA payloads, homoglyphs ([`identify_text`],
//!   [`remove_text_metadata`]);
//! - span-protected prose regeneration for token-level watermarks ([`remove_text_invisible`],
//!   [`remove_text_all`]) with the shared fact, script, extent and language guards.

#[cfg(feature = "api")]
pub mod api;
mod c2pa;
pub mod error;
pub mod facts;
pub mod generator;
pub mod guard;
pub mod io;
#[cfg(feature = "langid")]
pub mod langid;
pub mod provenance;
pub mod pyre;
pub mod pyrepr;
pub mod report;
pub mod rewrite;
pub mod roles;
pub mod scores;
pub mod selectors;
pub mod spans;
pub mod stego;
pub mod tables;
pub mod unicode;

pub use error::TextError;
pub use facts::verify_text_rewrite_facts;
pub use generator::{
    ChatMessage, DEFAULT_TEXT_REWRITER_MODEL, FinishReason, Generation, GenerationRequest, LocalRewriter,
    PassageRewriter, RewriteValidator, Role, TextGenerator,
};
pub use guard::{
    LanguageClassifier, TEXT_REWRITE_SYSTEM_PROMPT, guard_local_rewrite_candidate, mask_local_rewrite_facts,
};
pub use io::TextInput;
pub use provenance::armor::{
    C2PA_ASCII_ARMOR_BEGIN, C2PA_ASCII_ARMOR_END, CommentStyle, ManifestSource, encode_c2pa_ascii_armor,
};
pub use provenance::c2patxt::{C2PATXT_MAGIC, C2PATXT_VERSION, encode_c2patxt};
pub use provenance::vsrmark::{VSRMARK_ALGORITHM, VSRMARK_MAGIC, VSRMARK_VERSION, encode_vsrmark};
pub use report::{
    Confidence, TextMetadataResult, TextProvenanceReport, TextSignal, identify_text, remove_text_metadata,
};
pub use rewrite::{
    AllOptions, DocumentRepairer, DocumentRewriter, DocumentValidator, InvisibleOptions, TextAllResult,
    TextInvisibleResult, TokenWatermarkAction, remove_text_all, remove_text_invisible,
};
pub use roles::{DependencyParser, ParsedDoc, ParsedToken, verify_text_rewrite_loan_roles};
pub use scores::{
    DEFAULT_ALPHA_001_Z, SynthIDThreefryScore, TextWatermarkScore, ThreefryOptions, WatermarkScheme,
    score_gumbel_text, score_kgw_text, score_synthid_threefry_text, score_textgrain_text,
    score_textseal2_text, score_unigram_text,
};
pub use selectors::{
    byte_to_variation_selector, decode_variation_selectors, encode_variation_selectors,
    variation_selector_to_byte,
};
pub use spans::{SpanKind, TextSpan, split_protected_text_spans};
pub use stego::decode::{decode_stego_payloads, encode_innamark, encode_unicode_tags};

/// File extensions handled as text documents.
pub const TEXT_EXTENSIONS: &[&str] = &[
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".html",
    ".htm",
    ".xml",
    ".svg",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
    ".tex",
];

/// Whether the token-watermark stage can run with the compiled features and `generator` supplied.
pub const fn is_text_invisible_available(has_generator: bool) -> bool {
    has_generator && cfg!(feature = "langid")
}
