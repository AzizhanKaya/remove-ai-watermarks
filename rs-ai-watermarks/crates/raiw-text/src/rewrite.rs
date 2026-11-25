//! Span-protected prose regeneration (`remove_text_invisible`) and the combined cleaner.

use std::cell::RefCell;
use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use serde::Serialize;

use crate::error::TextError;
use crate::generator::{LocalRewriter, PassageRewriter, RewriteValidator, TextGenerator};
use crate::guard::{LanguageClassifier, unused_span_tokens};
use crate::io::{TextInput, prepare_text_io, write_output_text};
use crate::pyrepr::repr_str;
use crate::report::{TextSignal, identify_text, remove_text_metadata};
use crate::spans::{SpanKind, TextSpan, split_protected_text_spans};
use crate::unicode;

const DOCUMENT_VETO: &str = "Text rewrite rejected by document_validator; no output was written.";
/// Raised when prose needs regeneration but no rewriter or local generator was configured.
pub const MISSING_GENERATOR: &str = "Token-level text watermark removal requires a local text generator (enable a generation backend or pass a custom rewriter).";

/// One ordered batch of unlocked prose spans in, one replacement per span out.
pub trait DocumentRewriter {
    fn rewrite_document(&mut self, prose: &[String]) -> Result<Vec<String>, TextError>;
}

impl<F: FnMut(&[String]) -> Result<Vec<String>, TextError>> DocumentRewriter for F {
    fn rewrite_document(&mut self, prose: &[String]) -> Result<Vec<String>, TextError> {
        self(prose)
    }
}

/// Reviews the original and the complete visible result before anything is written.
pub trait DocumentValidator {
    fn validate_document(&mut self, source: &str, candidate: &str) -> Result<bool, TextError>;
}

impl<F: FnMut(&str, &str) -> Result<bool, TextError>> DocumentValidator for F {
    fn validate_document(&mut self, source: &str, candidate: &str) -> Result<bool, TextError> {
        self(source, candidate)
    }
}

/// Offers replacement slots for `(original, draft)`, or `None` to refuse/keep.
pub trait DocumentRepairer {
    fn repair_document(
        &mut self,
        original: &[String],
        draft: &[String],
    ) -> Result<Option<Vec<String>>, TextError>;
}

impl<F: FnMut(&[String], &[String]) -> Result<Option<Vec<String>>, TextError>> DocumentRepairer for F {
    fn repair_document(
        &mut self,
        original: &[String],
        draft: &[String],
    ) -> Result<Option<Vec<String>>, TextError> {
        self(original, draft)
    }
}

/// Options for [`remove_text_invisible`]; the defaults mirror the Python keyword defaults.
#[derive(Default)]
pub struct InvisibleOptions<'a> {
    /// A caller-supplied passage rewriter; its factual policy is caller-owned.
    pub rewriter: Option<&'a mut dyn PassageRewriter>,
    /// The local generation backend used when no custom rewriter is given.
    pub generator: Option<&'a mut dyn TextGenerator>,
    /// Strict-boolean acceptance check over restored visible text; three attempts per cluster.
    pub rewrite_validator: Option<RewriteValidator<'a>>,
    pub document_rewriter: Option<&'a mut dyn DocumentRewriter>,
    pub document_validator: Option<&'a mut dyn DocumentValidator>,
    pub document_repairer: Option<&'a mut dyn DocumentRepairer>,
    pub document_refiner: Option<&'a mut dyn DocumentRepairer>,
    /// Language guard for the local generator; defaults to the packaged py3langid model.
    pub language: Option<&'a dyn LanguageClassifier>,
    pub seed: u64,
    /// Retain clusters whose rewrite fails with a [`TextError::Runtime`] instead of aborting.
    pub allow_partial: bool,
}

impl std::fmt::Debug for InvisibleOptions<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InvisibleOptions")
            .field("rewriter", &self.rewriter.is_some())
            .field(
                "generator",
                &self
                    .generator
                    .as_ref()
                    .map(|generator| generator.model_id().to_owned()),
            )
            .field("rewrite_validator", &self.rewrite_validator.is_some())
            .field("document_rewriter", &self.document_rewriter.is_some())
            .field("seed", &self.seed)
            .field("allow_partial", &self.allow_partial)
            .finish_non_exhaustive()
    }
}

/// How the token-watermark stage ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenWatermarkAction {
    NotRun,
    Regenerated,
    PartialRegeneration,
    NoProseSpans,
}

#[allow(clippy::ref_option)]
fn serialize_path<S: serde::Serializer>(path: &Option<PathBuf>, serializer: S) -> Result<S::Ok, S::Error> {
    match path {
        Some(path) => serializer.serialize_str(&path.to_string_lossy()),
        None => serializer.serialize_none(),
    }
}

/// Result of span-protected prose regeneration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TextInvisibleResult {
    #[serde(serialize_with = "serialize_path")]
    pub source: Option<PathBuf>,
    #[serde(serialize_with = "serialize_path")]
    pub output: Option<PathBuf>,
    pub cleaned_text: String,
    pub rewritten_spans: usize,
    pub locked_spans: usize,
    pub locked_characters: usize,
    pub rewritten_characters: usize,
    pub model_id: String,
    pub retained_prose_spans: usize,
}

impl TextInvisibleResult {
    pub const fn token_watermark_action(&self) -> TokenWatermarkAction {
        if self.retained_prose_spans > 0 {
            TokenWatermarkAction::PartialRegeneration
        } else if self.rewritten_spans > 0 {
            TokenWatermarkAction::Regenerated
        } else {
            TokenWatermarkAction::NoProseSpans
        }
    }
}

/// Combined result of metadata scrubbing and optional prose regeneration.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TextAllResult {
    #[serde(serialize_with = "serialize_path")]
    pub source: Option<PathBuf>,
    #[serde(serialize_with = "serialize_path")]
    pub output: Option<PathBuf>,
    pub cleaned_text: String,
    pub removed_signals: Vec<TextSignal>,
    pub detected_markers: IndexMap<String, String>,
    pub remaining_markers: IndexMap<String, String>,
    pub removed_codepoints: usize,
    pub rewritten_spans: usize,
    pub locked_spans: usize,
    pub model_id: Option<String>,
    pub retained_prose_spans: usize,
}

impl TextAllResult {
    pub const fn token_watermark_action(&self) -> TokenWatermarkAction {
        if self.model_id.is_none() {
            TokenWatermarkAction::NotRun
        } else if self.retained_prose_spans > 0 {
            TokenWatermarkAction::PartialRegeneration
        } else if self.rewritten_spans > 0 {
            TokenWatermarkAction::Regenerated
        } else {
            TokenWatermarkAction::NoProseSpans
        }
    }
}

fn char_len(text: &str) -> usize {
    text.chars().count()
}

/// Placeholder tokens for the locked spans of a cluster, absent from its text.
fn protected_cluster_tokens(cluster: &[TextSpan]) -> Vec<(String, String)> {
    let joined: String = cluster.iter().map(|span| span.text.as_str()).collect();
    let mut tokens = unused_span_tokens(&joined);
    cluster
        .iter()
        .filter(|span| span.locked)
        .filter_map(|span| tokens.next().map(|token| (token, span.text.clone())))
        .collect()
}

fn rewrite_preserving_whitespace(
    text: &str,
    rewriter: &mut dyn PassageRewriter,
) -> Result<String, TextError> {
    let core = unicode::strip(text);
    if core.is_empty() {
        return Ok(text.to_owned());
    }
    let lead = &text[..text.len() - unicode::lstrip(text).len()];
    let trail = &text[unicode::rstrip(text).len()..];
    let rewritten = rewriter.rewrite(core)?;
    let rewritten = unicode::strip(&rewritten);
    if rewritten.is_empty() {
        return Err(TextError::runtime("Text rewriter returned empty prose."));
    }
    Ok(format!("{lead}{rewritten}{trail}"))
}

/// Rewrite a paragraph cluster with placeholders; fall back per prose span if tokens break.
fn rewrite_paragraph_cluster(
    cluster: &[TextSpan],
    rewriter: &mut dyn PassageRewriter,
) -> Result<(String, usize, usize), TextError> {
    let prose_count = cluster.iter().filter(|span| !span.locked).count();
    if prose_count == 0 {
        return Ok((cluster.iter().map(|span| span.text.as_str()).collect(), 0, 0));
    }
    let placeholders = protected_cluster_tokens(cluster);
    let mut locked = placeholders.iter();
    let mut templated = String::new();
    let mut rewritten_chars = 0;
    for span in cluster {
        if span.locked {
            if let Some((token, _)) = locked.next() {
                templated.push_str(token);
            }
        } else {
            templated.push_str(&span.text);
            rewritten_chars += char_len(&span.text);
        }
    }
    let leading = &templated[..templated.len() - unicode::lstrip(&templated).len()];
    let trailing = &templated[unicode::rstrip(&templated).len()..];
    let rewritten = rewriter.rewrite(unicode::strip(&templated))?;
    let core = unicode::strip(&rewritten);
    if core.is_empty() {
        return Err(TextError::runtime("Text rewriter returned empty prose."));
    }
    if placeholders
        .iter()
        .all(|(token, _)| core.matches(token.as_str()).count() == 1)
    {
        let mut restored = core.to_owned();
        for (token, original) in &placeholders {
            restored = restored.replace(token.as_str(), original);
        }
        return Ok((
            format!("{leading}{restored}{trailing}"),
            prose_count,
            rewritten_chars,
        ));
    }
    let mut out = String::new();
    for span in cluster {
        if span.locked {
            out.push_str(&span.text);
        } else {
            out.push_str(&rewrite_preserving_whitespace(&span.text, rewriter)?);
        }
    }
    Ok((out, prose_count, rewritten_chars))
}

struct DocumentAssembly {
    result: TextInvisibleResult,
    proposed: Vec<String>,
}

fn assemble_document(
    spans: &[TextSpan],
    prose: &[String],
    replacements: &[String],
    source: Option<&Path>,
    output: Option<&Path>,
) -> Result<DocumentAssembly, TextError> {
    if replacements.len() != prose.len() {
        return Err(TextError::runtime(
            "Document rewriter must return one replacement per prose span.",
        ));
    }
    let mut trimmed = Vec::with_capacity(replacements.len());
    for value in replacements {
        let core = unicode::strip(value);
        if core.is_empty() {
            return Err(TextError::runtime(
                "Document rewriter returned an invalid or empty prose span.",
            ));
        }
        trimmed.push(core.to_owned());
    }
    let mut cores = trimmed.iter();
    let mut cleaned = String::new();
    let (mut rewritten, mut rewritten_characters) = (0, 0);
    for span in spans {
        if span.locked {
            cleaned.push_str(&span.text);
            continue;
        }
        let old_core = unicode::strip(&span.text);
        let core = cores.next().map_or("", String::as_str);
        if core == old_core {
            cleaned.push_str(&span.text);
            continue;
        }
        let start = span.text.len() - unicode::lstrip(&span.text).len();
        cleaned.push_str(&span.text[..start]);
        cleaned.push_str(core);
        cleaned.push_str(&span.text[start + old_core.len()..]);
        rewritten += 1;
        rewritten_characters += char_len(&span.text);
    }
    if !prose.is_empty() && rewritten == 0 {
        return Err(TextError::runtime(
            "Document rewriter returned unchanged prose; no regeneration was performed.",
        ));
    }
    let locked: Vec<(SpanKind, &str)> = spans
        .iter()
        .filter(|span| span.locked)
        .map(|span| (span.kind, span.text.as_str()))
        .collect();
    let resegmented = split_protected_text_spans(&cleaned);
    let relocked: Vec<(SpanKind, &str)> = resegmented
        .iter()
        .filter(|span| span.locked)
        .map(|span| (span.kind, span.text.as_str()))
        .collect();
    if relocked != locked {
        return Err(TextError::runtime(
            "Document rewriter changed the protected text structure.",
        ));
    }
    let protected: Vec<&str> = locked
        .iter()
        .filter(|(kind, _)| *kind != SpanKind::Whitespace)
        .map(|(_, text)| *text)
        .collect();
    Ok(DocumentAssembly {
        result: TextInvisibleResult {
            source: source.map(Path::to_path_buf),
            output: output.map(Path::to_path_buf),
            cleaned_text: cleaned,
            rewritten_spans: rewritten,
            locked_spans: protected.len(),
            locked_characters: protected.iter().map(|text| char_len(text)).sum(),
            rewritten_characters,
            model_id: "document_rewriter".to_owned(),
            retained_prose_spans: prose.len() - rewritten,
        },
        proposed: trimmed,
    })
}

fn passes_document_validator(
    validator: &mut Option<&mut dyn DocumentValidator>,
    source: &str,
    candidate: &str,
) -> Result<bool, TextError> {
    match validator {
        Some(validator) => validator.validate_document(source, candidate),
        None => Ok(true),
    }
}

fn validate_options(options: &InvisibleOptions<'_>) -> Result<(), TextError> {
    if options.rewriter.is_some() && options.generator.is_some() {
        return Err(TextError::invalid(
            "rewriter_backend cannot select a backend for a custom rewriter",
        ));
    }
    let has_document_pair = options.document_rewriter.is_some() && options.document_validator.is_some();
    if options.document_repairer.is_some() && !has_document_pair {
        return Err(TextError::invalid(
            "document_repairer requires document_rewriter and document_validator",
        ));
    }
    if options.document_refiner.is_some() && !has_document_pair {
        return Err(TextError::invalid(
            "document_refiner requires document_rewriter and document_validator",
        ));
    }
    if options.document_rewriter.is_some() {
        if options.rewriter.is_some() || options.rewrite_validator.is_some() {
            return Err(TextError::invalid(
                "document_rewriter cannot be combined with passage rewriter or rewrite_validator",
            ));
        }
        if options.generator.is_some() {
            return Err(TextError::invalid(
                "rewriter_backend cannot select a backend for document_rewriter",
            ));
        }
    }
    Ok(())
}

fn rewrite_documents(
    text: &str,
    spans: &[TextSpan],
    source: Option<&Path>,
    output: Option<&Path>,
    options: &mut InvisibleOptions<'_>,
) -> Result<TextInvisibleResult, TextError> {
    let prose: Vec<String> = spans
        .iter()
        .filter(|span| !span.locked)
        .map(|span| unicode::strip(&span.text).to_owned())
        .collect();
    let replacements = match (&mut options.document_rewriter, prose.is_empty()) {
        (Some(rewriter), false) => rewriter.rewrite_document(&prose)?,
        _ => Vec::new(),
    };
    let mut assembly = assemble_document(spans, &prose, &replacements, source, output)?;
    if assembly.result.rewritten_spans > 0
        && !passes_document_validator(
            &mut options.document_validator,
            text,
            &assembly.result.cleaned_text,
        )?
    {
        let correction = match &mut options.document_repairer {
            Some(repairer) => repairer.repair_document(&prose, &assembly.proposed)?,
            None => None,
        };
        let Some(correction) = correction else {
            return Err(TextError::runtime(DOCUMENT_VETO));
        };
        let repaired = assemble_document(spans, &prose, &correction, source, output)?;
        if repaired.result.cleaned_text == assembly.result.cleaned_text {
            return Err(TextError::runtime(
                "Document repairer returned the rejected draft unchanged.",
            ));
        }
        if !passes_document_validator(
            &mut options.document_validator,
            text,
            &repaired.result.cleaned_text,
        )? {
            return Err(TextError::runtime(DOCUMENT_VETO));
        }
        assembly = repaired;
    }
    if assembly.result.rewritten_spans > 0
        && let Some(refiner) = &mut options.document_refiner
        && let Some(refinement) = refiner.repair_document(&prose, &assembly.proposed)?
    {
        assembly = assemble_document(spans, &prose, &refinement, source, output)?;
        if !passes_document_validator(
            &mut options.document_validator,
            text,
            &assembly.result.cleaned_text,
        )? {
            return Err(TextError::runtime(DOCUMENT_VETO));
        }
    }
    Ok(assembly.result)
}

/// A custom rewriter that retries with the validator, skipping empty or unchanged proposals.
struct ValidatedRewriter<'r, 'v> {
    inner: &'r mut dyn PassageRewriter,
    validator: RewriteValidator<'v>,
}

impl PassageRewriter for ValidatedRewriter<'_, '_> {
    fn rewrite(&mut self, passage: &str) -> Result<String, TextError> {
        for _ in 0..3 {
            let candidate = self.inner.rewrite(passage)?;
            let candidate = unicode::strip(&candidate);
            if candidate.is_empty() || candidate == unicode::strip(passage) {
                continue;
            }
            if (self.validator)(passage, candidate)? {
                return Ok(candidate.to_owned());
            }
        }
        Err(TextError::runtime(
            "Text rewrite validator rejected all 3 candidate attempts, including empty or unchanged prose.",
        ))
    }
}

/// Remove statistical token watermarks by regenerating unlocked prose spans.
///
/// Fenced and inline code, math, URLs, paths, HTML and Markdown structure are locked and
/// verified byte-for-byte. With a document rewriter the ordered prose slots are replaced in one
/// call and reviewed by the document validator/repairer/refiner before any write.
pub fn remove_text_invisible<'a>(
    input: impl Into<TextInput<'a>>,
    output: Option<&Path>,
    mut options: InvisibleOptions<'_>,
) -> Result<TextInvisibleResult, TextError> {
    validate_options(&options)?;
    let input = input.into();
    let (resolved, text) = prepare_text_io(input, output)?;
    let source = input.source();
    let spans = split_protected_text_spans(&text);
    if options.document_rewriter.is_some() {
        let result = rewrite_documents(&text, &spans, source, resolved.as_deref(), &mut options)?;
        write_output_text(resolved.as_deref(), &result.cleaned_text)?;
        return Ok(result);
    }

    let has_prose = spans.iter().any(|span| !span.locked);
    let cluster_tokens: RefCell<Vec<(String, String)>> = RefCell::new(Vec::new());
    let visible_validator: Option<RewriteValidator<'_>> =
        options.rewrite_validator.take().map(|mut validator| {
            let tokens = &cluster_tokens;
            Box::new(move |original: &str, candidate: &str| {
                let (mut original, mut candidate) = (original.to_owned(), candidate.to_owned());
                for (token, literal) in tokens.borrow().iter() {
                    original = original.replace(token.as_str(), literal);
                    candidate = candidate.replace(token.as_str(), literal);
                }
                validator(&original, &candidate)
            }) as RewriteValidator<'_>
        });

    #[cfg(feature = "langid")]
    let default_language = crate::langid::DefaultLanguageClassifier;
    let language: Option<&dyn LanguageClassifier> = options.language.or({
        #[cfg(feature = "langid")]
        {
            Some(&default_language as &dyn LanguageClassifier)
        }
        #[cfg(not(feature = "langid"))]
        {
            None
        }
    });

    let mut local: Option<LocalRewriter<'_>> = None;
    let mut validated: Option<ValidatedRewriter<'_, '_>> = None;
    let mut model_label = "custom_rewriter".to_owned();
    let mut active: Option<&mut dyn PassageRewriter> = None;
    match (options.rewriter, options.generator) {
        (Some(custom), _) => match visible_validator {
            Some(validator) => {
                active = Some(validated.insert(ValidatedRewriter {
                    inner: custom,
                    validator,
                }));
            }
            None => active = Some(custom),
        },
        (None, Some(generator)) => {
            generator.model_id().clone_into(&mut model_label);
            if has_prose {
                let language = language.ok_or_else(|| TextError::runtime(MISSING_GENERATOR))?;
                active = Some(local.insert(LocalRewriter::new(
                    generator,
                    options.seed,
                    visible_validator,
                    language,
                )));
            }
        }
        (None, None) => {
            crate::generator::DEFAULT_TEXT_REWRITER_MODEL.clone_into(&mut model_label);
            if has_prose {
                return Err(TextError::runtime(MISSING_GENERATOR));
            }
        }
    }

    let mut out = String::with_capacity(text.len());
    let mut counts = ClusterCounts::default();
    let mut cluster: Vec<TextSpan> = Vec::new();
    for span in &spans {
        if span.kind.is_block() {
            flush_cluster(
                &mut cluster,
                &mut active,
                &cluster_tokens,
                options.allow_partial,
                &mut out,
                &mut counts,
            )?;
            out.push_str(&span.text);
            if span.kind != SpanKind::Whitespace {
                counts.locked_spans += 1;
                counts.locked_characters += char_len(&span.text);
            }
        } else {
            cluster.push(span.clone());
        }
    }
    flush_cluster(
        &mut cluster,
        &mut active,
        &cluster_tokens,
        options.allow_partial,
        &mut out,
        &mut counts,
    )?;

    if options.allow_partial && has_prose && counts.rewritten_spans == 0 {
        return Err(TextError::runtime(
            "Text regeneration rejected every prose cluster; no output was written.",
        ));
    }
    let mut cursor = 0;
    for span in spans
        .iter()
        .filter(|span| span.locked && span.kind != SpanKind::Whitespace)
    {
        let Some(found) = out[cursor..].find(&span.text) else {
            return Err(TextError::runtime(format!(
                "Locked {} span failed byte-identical verification: {}",
                span.kind.as_str(),
                repr_str(&span.text)
            )));
        };
        cursor += found + span.text.len();
    }
    if counts.rewritten_spans > 0 && !passes_document_validator(&mut options.document_validator, &text, &out)?
    {
        return Err(TextError::runtime(DOCUMENT_VETO));
    }
    write_output_text(resolved.as_deref(), &out)?;
    Ok(TextInvisibleResult {
        source: source.map(Path::to_path_buf),
        output: resolved,
        cleaned_text: out,
        rewritten_spans: counts.rewritten_spans,
        locked_spans: counts.locked_spans,
        locked_characters: counts.locked_characters,
        rewritten_characters: counts.rewritten_characters,
        model_id: model_label,
        retained_prose_spans: counts.retained_prose_spans,
    })
}

#[derive(Debug, Default)]
struct ClusterCounts {
    rewritten_spans: usize,
    retained_prose_spans: usize,
    locked_spans: usize,
    locked_characters: usize,
    rewritten_characters: usize,
}

fn flush_cluster(
    cluster: &mut Vec<TextSpan>,
    active: &mut Option<&mut dyn PassageRewriter>,
    cluster_tokens: &RefCell<Vec<(String, String)>>,
    allow_partial: bool,
    out: &mut String,
    counts: &mut ClusterCounts,
) -> Result<(), TextError> {
    if cluster.is_empty() {
        return Ok(());
    }
    for span in cluster.iter().filter(|span| span.locked) {
        counts.locked_spans += 1;
        counts.locked_characters += char_len(&span.text);
    }
    let original: String = cluster.iter().map(|span| span.text.as_str()).collect();
    match active {
        Some(rewriter) if cluster.iter().any(|span| !span.locked) => {
            *cluster_tokens.borrow_mut() = protected_cluster_tokens(cluster);
            let attempt =
                rewrite_paragraph_cluster(cluster, &mut **rewriter).and_then(|(rebuilt, spans, chars)| {
                    if rebuilt == original {
                        Err(TextError::runtime(
                            "Text rewriter returned unchanged prose; no regeneration was performed.",
                        ))
                    } else {
                        Ok((rebuilt, spans, chars))
                    }
                });
            let (rebuilt, spans, chars) = match attempt {
                Ok(done) => done,
                Err(error) if allow_partial && error.is_runtime() => {
                    counts.retained_prose_spans += cluster.iter().filter(|span| !span.locked).count();
                    (original, 0, 0)
                }
                Err(error) => return Err(error),
            };
            out.push_str(&rebuilt);
            counts.rewritten_spans += spans;
            counts.rewritten_characters += chars;
        }
        _ => out.push_str(&original),
    }
    cluster.clear();
    Ok(())
}

/// Options for [`remove_text_all`].
#[derive(Debug, Default)]
pub struct AllOptions<'a> {
    /// Opt into span-protected prose regeneration.
    pub include_invisible: bool,
    pub invisible: InvisibleOptions<'a>,
}

/// Strip deterministic text provenance and optionally regenerate prose spans.
pub fn remove_text_all<'a>(
    input: impl Into<TextInput<'a>>,
    output: Option<&Path>,
    options: AllOptions<'_>,
) -> Result<TextAllResult, TextError> {
    let invisible = &options.invisible;
    if !options.include_invisible {
        if invisible.allow_partial {
            return Err(TextError::invalid(
                "allow_partial requires include_invisible=True",
            ));
        }
        if invisible.rewrite_validator.is_some() {
            return Err(TextError::invalid(
                "rewrite_validator requires include_invisible=True",
            ));
        }
        if invisible.document_rewriter.is_some()
            || invisible.document_validator.is_some()
            || invisible.document_repairer.is_some()
        {
            return Err(TextError::invalid(
                "document_rewriter, document_validator and document_repairer require include_invisible=True",
            ));
        }
        if invisible.document_refiner.is_some() {
            return Err(TextError::invalid(
                "document_refiner requires include_invisible=True",
            ));
        }
    }
    if invisible.rewriter.is_some() && invisible.generator.is_some() {
        return Err(TextError::invalid(
            "rewriter_backend cannot select a backend for a custom rewriter",
        ));
    }
    let input = input.into();
    let (resolved, raw) = prepare_text_io(input, output)?;
    let meta = remove_text_metadata(raw.as_str(), None)?;
    let mut result = TextAllResult {
        source: input.source().map(Path::to_path_buf),
        output: resolved.clone(),
        cleaned_text: meta.cleaned_text,
        removed_signals: meta.removed_signals,
        detected_markers: meta.detected_markers,
        remaining_markers: meta.remaining_markers,
        removed_codepoints: meta.removed_codepoints,
        rewritten_spans: 0,
        locked_spans: 0,
        model_id: None,
        retained_prose_spans: 0,
    };
    if options.include_invisible {
        let regenerated = remove_text_invisible(result.cleaned_text.as_str(), None, options.invisible)?;
        result.remaining_markers = identify_text(regenerated.cleaned_text.as_str())?.metadata_markers;
        result.cleaned_text = regenerated.cleaned_text;
        result.rewritten_spans = regenerated.rewritten_spans;
        result.retained_prose_spans = regenerated.retained_prose_spans;
        result.locked_spans = regenerated.locked_spans;
        result.model_id = Some(regenerated.model_id);
    }
    write_output_text(resolved.as_deref(), &result.cleaned_text)?;
    Ok(result)
}
