//! Behavioral ports of the Python regeneration tests (custom, document and local-generator routes).

use std::cell::RefCell;
use std::path::Path;

use raiw_text::generator::{ChatMessage, FinishReason, Generation, GenerationRequest, TextGenerator};
use raiw_text::guard::{self, LanguageClassifier};
use raiw_text::{
    AllOptions, InvisibleOptions, PassageRewriter, SpanKind, TextError, TokenWatermarkAction,
    remove_text_all, remove_text_invisible, remove_text_metadata, split_protected_text_spans,
    verify_text_rewrite_facts,
};

#[allow(clippy::unnecessary_wraps)]
fn english(_: &str) -> Result<(String, f64), TextError> {
    Ok(("en".to_owned(), 0.99))
}

fn invisible(
    text: &str,
    rewriter: &mut dyn PassageRewriter,
) -> Result<raiw_text::TextInvisibleResult, TextError> {
    remove_text_invisible(
        text,
        None,
        InvisibleOptions {
            rewriter: Some(rewriter),
            ..InvisibleOptions::default()
        },
    )
}

fn closure(rewrite: impl Fn(&str) -> String) -> impl FnMut(&str) -> Result<String, TextError> {
    move |passage| Ok(rewrite(passage))
}

#[test]
fn span_protected_regeneration_locks_technical_spans_verbatim() {
    let document = "# Architecture Overview\n\nThe model computes `y = f(x)` using threshold $\\alpha = 0.01$ and publishes results to https://raiw.cc/docs/spec from ./src/runner.py.\n\n```python\ndef compute(x: int) -> int:\n    return x * 2\n```\n\n$$\\sum_{i=1}^{n} x_i = 1$$\n\nA second prose paragraph explains the deployment parameters in detail.\n";
    let kinds: Vec<SpanKind> = split_protected_text_spans(document)
        .iter()
        .map(|span| span.kind)
        .collect();
    for kind in [
        SpanKind::MarkdownStructure,
        SpanKind::Prose,
        SpanKind::InlineCode,
        SpanKind::MathInline,
        SpanKind::Url,
        SpanKind::Path,
        SpanKind::FencedCode,
        SpanKind::MathBlock,
    ] {
        assert!(kinds.contains(&kind), "{kind:?}");
    }
    let mut preserving = closure(|text| {
        text.replace("The model computes", "Our system evaluates")
            .replace(
                "A second prose paragraph explains",
                "The concluding section describes",
            )
    });
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("doc_clean.md");
    let result = remove_text_invisible(
        document,
        Some(&out),
        InvisibleOptions {
            rewriter: Some(&mut preserving),
            ..InvisibleOptions::default()
        },
    )
    .unwrap();
    assert_eq!(result.token_watermark_action(), TokenWatermarkAction::Regenerated);
    assert!(result.rewritten_spans >= 2);
    assert!(
        result
            .cleaned_text
            .contains("Our system evaluates `y = f(x)` using threshold $\\alpha = 0.01$")
    );
    assert!(
        result
            .cleaned_text
            .contains("def compute(x: int) -> int:\n    return x * 2")
    );
    assert_eq!(std::fs::read_to_string(&out).unwrap(), result.cleaned_text);

    let mut dropping = closure(|_| "Rewritten prose segment".to_owned());
    let dropped = invisible(document, &mut dropping).unwrap();
    for literal in [
        "# Architecture Overview",
        "`y = f(x)`",
        "$\\alpha = 0.01$",
        "https://raiw.cc/docs/spec",
        "./src/runner.py",
        "$$\\sum_{i=1}^{n} x_i = 1$$",
    ] {
        assert!(dropped.cleaned_text.contains(literal), "{literal}");
    }
}

#[test]
fn file_operations_preserve_line_endings_and_locked_code() {
    for newline in ["\r\n", "\r", "\n"] {
        let text = [
            "Original prose.",
            "",
            "```python",
            "value = \"source\"",
            "```",
            "",
        ]
        .join(newline);
        let dir = tempfile::tempdir().unwrap();
        let (source, output) = (dir.path().join("source.md"), dir.path().join("output.md"));
        std::fs::write(&source, text.as_bytes()).unwrap();
        assert_eq!(
            remove_text_metadata(source.as_path(), Some(&output))
                .unwrap()
                .cleaned_text,
            text
        );
        assert_eq!(
            remove_text_all(source.as_path(), Some(&output), AllOptions::default())
                .unwrap()
                .cleaned_text,
            text
        );
        let mut rewrite = |passage: &str| {
            assert!(
                !passage.contains("value"),
                "Fenced code reached the prose rewriter"
            );
            Ok(passage.replace("Original", "Rephrased"))
        };
        let result = remove_text_invisible(
            source.as_path(),
            Some(&output),
            InvisibleOptions {
                rewriter: Some(&mut rewrite),
                ..InvisibleOptions::default()
            },
        )
        .unwrap();
        let expected = text.replace("Original", "Rephrased");
        assert_eq!(result.cleaned_text, expected);
        assert_eq!(std::fs::read(&output).unwrap(), expected.as_bytes());
        assert_eq!(std::fs::read(&source).unwrap(), text.as_bytes());
    }
}

#[test]
fn default_output_path_and_same_path_refusal() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("doc.txt");
    std::fs::write(&source, "Plain\u{200b} text.").unwrap();
    let result = remove_text_metadata(source.as_path(), None).unwrap();
    assert_eq!(
        result.output.as_deref(),
        Some(dir.path().join("doc_clean.txt").as_path())
    );
    assert_eq!(
        std::fs::read_to_string(dir.path().join("doc_clean.txt")).unwrap(),
        "Plain text."
    );
    let error = remove_text_metadata(source.as_path(), Some(&source)).unwrap_err();
    assert_eq!(error.to_string(), "Output path must differ from source text file");
    std::fs::write(&source, b"\xff").unwrap();
    assert!(matches!(
        raiw_text::identify_text(source.as_path()),
        Err(TextError::Decode(_))
    ));
}

fn selective(passage: &str) -> Result<String, TextError> {
    if passage.contains("retained") {
        return Err(TextError::runtime("Candidate did not preserve meaning"));
    }
    Ok(passage.replace("rewritten", "rephrased"))
}

#[test]
fn partial_regeneration_retains_rejected_prose() {
    let source = "This paragraph is rewritten.\n\nThis retained paragraph has `original()`.\n\nFinal paragraph is rewritten.";
    let mut rewriter = selective;
    let result = remove_text_invisible(
        source,
        None,
        InvisibleOptions {
            rewriter: Some(&mut rewriter),
            allow_partial: true,
            ..InvisibleOptions::default()
        },
    )
    .unwrap();
    assert_eq!(result.cleaned_text, source.replace("rewritten", "rephrased"));
    assert_eq!((result.rewritten_spans, result.retained_prose_spans), (2, 2));
    assert_eq!(
        result.token_watermark_action(),
        TokenWatermarkAction::PartialRegeneration
    );

    let mut identity_on_second = closure(|passage| passage.replace("rewritten", "rephrased"));
    let result = remove_text_invisible(
        "This paragraph is rewritten.\n\nThis paragraph stays original.",
        None,
        InvisibleOptions {
            rewriter: Some(&mut identity_on_second),
            allow_partial: true,
            ..InvisibleOptions::default()
        },
    )
    .unwrap();
    assert_eq!(result.retained_prose_spans, 1);

    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("result.txt");
    std::fs::write(&output, "Previous reviewed output.").unwrap();
    let mut echo = closure(str::to_owned);
    let error = remove_text_invisible(
        "Retain every word.",
        Some(&output),
        InvisibleOptions {
            rewriter: Some(&mut echo),
            allow_partial: true,
            ..InvisibleOptions::default()
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("regeneration"));
    assert_eq!(
        std::fs::read_to_string(&output).unwrap(),
        "Previous reviewed output."
    );

    let mut broken = |_: &str| -> Result<String, TextError> {
        Err(TextError::invalid("Unexpected invalid configuration"))
    };
    let output = dir.path().join("fresh.txt");
    let error = remove_text_invisible(
        "Some ordinary prose.",
        Some(&output),
        InvisibleOptions {
            rewriter: Some(&mut broken),
            allow_partial: true,
            ..InvisibleOptions::default()
        },
    )
    .unwrap_err();
    assert!(matches!(error, TextError::InvalidArgument(_)));
    assert!(!output.exists());

    let mut rewriter = selective;
    let combined = remove_text_all(
        "This paragraph is rewritten.\n\nThis paragraph is retained.\u{200b}",
        None,
        AllOptions {
            include_invisible: true,
            invisible: InvisibleOptions {
                rewriter: Some(&mut rewriter),
                allow_partial: true,
                ..InvisibleOptions::default()
            },
        },
    )
    .unwrap();
    assert_eq!(
        combined.cleaned_text,
        "This paragraph is rephrased.\n\nThis paragraph is retained."
    );
    assert_eq!(
        combined.token_watermark_action(),
        TokenWatermarkAction::PartialRegeneration
    );
    let error = remove_text_all(
        "Some prose.",
        None,
        AllOptions {
            invisible: InvisibleOptions {
                allow_partial: true,
                ..InvisibleOptions::default()
            },
            ..AllOptions::default()
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("include_invisible"));
}

#[test]
fn validators_participate_in_the_rewrite_transaction() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("out.txt");
    std::fs::write(&output, "Previous result.").unwrap();
    let mut tourists = closure(|_| "Tourists used the room.".to_owned());
    let error = remove_text_invisible(
        "Visitors used the room.",
        Some(&output),
        InvisibleOptions {
            rewriter: Some(&mut tourists),
            rewrite_validator: Some(Box::new(|_: &str, candidate: &str| {
                Ok(!candidate.contains("Tourists"))
            })),
            ..InvisibleOptions::default()
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("validator"));
    assert_eq!(std::fs::read_to_string(&output).unwrap(), "Previous result.");

    let mut swap = closure(|passage| {
        passage
            .replace("Visitors", "Tourists")
            .replace("borrowed", "rented")
    });
    let result = remove_text_invisible(
        "Visitors used the room.\n\nReaders borrowed books.",
        None,
        InvisibleOptions {
            rewriter: Some(&mut swap),
            rewrite_validator: Some(Box::new(|_: &str, candidate: &str| {
                Ok(!candidate.contains("Tourists"))
            })),
            allow_partial: true,
            ..InvisibleOptions::default()
        },
    )
    .unwrap();
    assert_eq!(
        result.cleaned_text,
        "Visitors used the room.\n\nReaders rented books."
    );
    assert_eq!(result.retained_prose_spans, 1);

    let candidates = RefCell::new(vec!["The room was used by visitors.", "Tourists used the room."]);
    let mut sequence = |_: &str| Ok(candidates.borrow_mut().pop().unwrap_or("").to_owned());
    let result = remove_text_invisible(
        "Visitors used the room.",
        None,
        InvisibleOptions {
            rewriter: Some(&mut sequence),
            rewrite_validator: Some(Box::new(|_: &str, candidate: &str| {
                Ok(!candidate.contains("Tourists"))
            })),
            ..InvisibleOptions::default()
        },
    )
    .unwrap();
    assert_eq!(result.cleaned_text, "The room was used by visitors.");

    let seen = RefCell::new(Vec::new());
    let mut describe = closure(|passage| passage.replace("Explain", "Describe"));
    let source = "Explain [[RAIW_SPAN_0]] and `value` precisely.";
    let result = remove_text_invisible(
        source,
        None,
        InvisibleOptions {
            rewriter: Some(&mut describe),
            rewrite_validator: Some(Box::new(|before: &str, after: &str| {
                seen.borrow_mut().push((before.to_owned(), after.to_owned()));
                Ok(true)
            })),
            ..InvisibleOptions::default()
        },
    )
    .unwrap();
    assert_eq!(result.cleaned_text, source.replace("Explain", "Describe"));
    assert_eq!(
        *seen.borrow(),
        vec![(source.to_owned(), result.cleaned_text.clone())]
    );

    let candidates = RefCell::new(vec!["The room was used by visitors.", "Visitors used the room."]);
    let mut echo_first = |_: &str| Ok(candidates.borrow_mut().pop().unwrap_or("").to_owned());
    let approved = RefCell::new(Vec::new());
    let result = remove_text_invisible(
        "Visitors used the room.",
        None,
        InvisibleOptions {
            rewriter: Some(&mut echo_first),
            rewrite_validator: Some(Box::new(|_: &str, candidate: &str| {
                approved.borrow_mut().push(candidate.to_owned());
                Ok(true)
            })),
            ..InvisibleOptions::default()
        },
    )
    .unwrap();
    assert_eq!(*approved.borrow(), vec![result.cleaned_text.clone()]);
}

#[test]
fn protected_content_and_empty_rewrites() {
    let source = "Explain `[[RAIW_SPAN_1]]` and `value` precisely.";
    let mut describe = closure(|passage| passage.replace("Explain", "Describe"));
    assert_eq!(
        invisible(source, &mut describe).unwrap().cleaned_text,
        source.replace("Explain", "Describe")
    );

    for source in ["Original prose.", "Use `safe()` carefully."] {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("out.txt");
        std::fs::write(&output, "Previous result.").unwrap();
        let mut rewrite = closure(|passage| {
            if passage.contains("[[RAIW_SPAN_") {
                "A changed paragraph.".to_owned()
            } else {
                String::new()
            }
        });
        let error = remove_text_invisible(
            source,
            Some(&output),
            InvisibleOptions {
                rewriter: Some(&mut rewrite),
                ..InvisibleOptions::default()
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("empty"), "{error}");
        assert_eq!(std::fs::read_to_string(&output).unwrap(), "Previous result.");
    }

    let mut identity = closure(str::to_owned);
    assert!(
        invisible("Ordinary prose needs actual rewording.", &mut identity)
            .unwrap_err()
            .to_string()
            .contains("unchanged")
    );

    let source = "1. Libraries help readers.\n2) Libraries host workshops.\n- Libraries lend books.";
    let mut archives = closure(|passage| passage.replace("Libraries", "Archives"));
    assert_eq!(
        invisible(source, &mut archives).unwrap().cleaned_text,
        source.replace("Libraries", "Archives")
    );

    let literal = format!("[[RAIW_SPAN_{}]]", "9".repeat(5000));
    let source = format!("We shipped 12 crates. {literal}");
    let (masked, replacements) = guard::mask_local_rewrite_facts(&source);
    let mut restored = masked;
    for (token, value) in &replacements {
        restored = restored.replace(token.as_str(), value);
    }
    assert_eq!(restored, source);
    assert!(!replacements.contains_key(&literal));
}

#[test]
fn document_route_uses_protected_assembly() {
    let source = "Visitors used the room.\r\n\r\n`value = 12`\r\n\r\nVisitors used the room.";
    let draft = "Staff used the room.\r\n\r\n`value = 12`\r\n\r\nGuests used the room.";
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("output.txt");
    let seen = RefCell::new(Vec::new());
    let mut rewrite = |passages: &[String]| {
        assert_eq!(passages, ["Visitors used the room.", "Visitors used the room."]);
        Ok(vec![
            "Staff used the room.".to_owned(),
            "Guests used the room.".to_owned(),
        ])
    };
    let mut validate = |before: &str, after: &str| {
        seen.borrow_mut().push((before.to_owned(), after.to_owned()));
        Ok(true)
    };
    let result = remove_text_invisible(
        source,
        Some(&output),
        InvisibleOptions {
            document_rewriter: Some(&mut rewrite),
            document_validator: Some(&mut validate),
            ..InvisibleOptions::default()
        },
    )
    .unwrap();
    assert_eq!(result.cleaned_text, draft);
    assert_eq!((result.rewritten_spans, result.locked_spans), (2, 1));
    assert_eq!(std::fs::read(&output).unwrap(), draft.as_bytes());
    assert_eq!(*seen.borrow(), vec![(source.to_owned(), draft.to_owned())]);

    for bad in [
        vec![],
        vec!["A new paragraph.".to_owned(), "Extra.".to_owned()],
        vec![" ".to_owned()],
        vec!["Use `different`.".to_owned()],
    ] {
        let output = dir.path().join("refused.txt");
        std::fs::write(&output, "Previous output.").unwrap();
        let mut rewrite = |_: &[String]| Ok(bad.clone());
        let outcome = remove_text_invisible(
            "Old text.",
            Some(&output),
            InvisibleOptions {
                document_rewriter: Some(&mut rewrite),
                ..InvisibleOptions::default()
            },
        );
        assert!(matches!(outcome, Err(TextError::Runtime(_))), "{bad:?}");
        assert_eq!(std::fs::read_to_string(&output).unwrap(), "Previous output.");
    }

    let mut retained = |_: &[String]| {
        Ok(vec![
            "Guests used the room.".to_owned(),
            "Readers borrowed books.".to_owned(),
        ])
    };
    let result = remove_text_invisible(
        "Visitors used the room.\n\nReaders borrowed books.",
        None,
        InvisibleOptions {
            document_rewriter: Some(&mut retained),
            ..InvisibleOptions::default()
        },
    )
    .unwrap();
    assert_eq!((result.rewritten_spans, result.retained_prose_spans), (1, 1));

    let mut unexpected = |_: &[String]| -> Result<Vec<String>, TextError> {
        panic!("protected-only input must not invoke the rewriter")
    };
    let result = remove_text_invisible(
        "`value = 12`",
        None,
        InvisibleOptions {
            document_rewriter: Some(&mut unexpected),
            ..InvisibleOptions::default()
        },
    )
    .unwrap();
    assert_eq!(
        result.token_watermark_action(),
        TokenWatermarkAction::NoProseSpans
    );
}

#[test]
fn document_veto_and_repair() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("output.txt");
    std::fs::write(&output, "Previous output.").unwrap();
    let mut rewrite = |_: &[String]| Ok(vec!["Guests used the room.".to_owned()]);
    let mut veto = |_: &str, _: &str| Ok(false);
    let error = remove_text_all(
        "Visitors used the room.",
        Some(&output),
        AllOptions {
            include_invisible: true,
            invisible: InvisibleOptions {
                document_rewriter: Some(&mut rewrite),
                document_validator: Some(&mut veto),
                ..InvisibleOptions::default()
            },
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("document_validator"));
    assert_eq!(std::fs::read_to_string(&output).unwrap(), "Previous output.");

    for accept_draft in [true, false] {
        let source = "Visitors used the room.\r\n\r\n`value = 12`\r\n\r\nReaders borrowed books.";
        let draft = "Staff used the room.\r\n\r\n`value = 12`\r\n\r\nReaders borrowed books.";
        let repaired = "Guests used the room.\r\n\r\n`value = 12`\r\n\r\nReaders borrowed books.";
        let calls = RefCell::new(Vec::new());
        let mut rewrite = |prose: &[String]| Ok(vec!["Staff used the room.".to_owned(), prose[1].clone()]);
        let mut validate = |before: &str, after: &str| {
            assert_eq!(before, source);
            calls.borrow_mut().push(after.to_owned());
            Ok(accept_draft || after == repaired)
        };
        let mut repair = |original: &[String], rejected: &[String]| {
            assert_eq!(original, ["Visitors used the room.", "Readers borrowed books."]);
            assert_eq!(rejected, ["Staff used the room.", "Readers borrowed books."]);
            Ok(Some(vec![
                "Guests used the room.".to_owned(),
                original[1].clone(),
            ]))
        };
        let result = remove_text_all(
            source,
            None,
            AllOptions {
                include_invisible: true,
                invisible: InvisibleOptions {
                    document_rewriter: Some(&mut rewrite),
                    document_validator: Some(&mut validate),
                    document_repairer: Some(&mut repair),
                    ..InvisibleOptions::default()
                },
            },
        )
        .unwrap();
        let expected = if accept_draft { draft } else { repaired };
        assert_eq!(result.cleaned_text, expected);
        assert_eq!(calls.borrow().len(), if accept_draft { 1 } else { 2 });
    }

    for (answer, reviews, message) in [
        (None, 1, "document_validator"),
        (Some(vec!["Draft text.".to_owned()]), 1, "rejected draft"),
        (Some(vec!["Repaired text.".to_owned()]), 2, "document_validator"),
        (
            Some(vec!["New `code`.".to_owned()]),
            1,
            "protected text structure",
        ),
        (Some(vec![]), 1, "one replacement"),
    ] {
        let reviewed = RefCell::new(0);
        let mut rewrite = |_: &[String]| Ok(vec!["Draft text.".to_owned()]);
        let mut validate = |_: &str, _: &str| {
            *reviewed.borrow_mut() += 1;
            Ok(false)
        };
        let mut repair = |_: &[String], _: &[String]| Ok(answer.clone());
        let error = remove_text_invisible(
            "Old text.",
            None,
            InvisibleOptions {
                document_rewriter: Some(&mut rewrite),
                document_validator: Some(&mut validate),
                document_repairer: Some(&mut repair),
                allow_partial: true,
                ..InvisibleOptions::default()
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains(message), "{error}");
        assert_eq!(*reviewed.borrow(), reviews);
    }

    let mut repair = |_: &[String], _: &[String]| Ok(None);
    let mut rewrite = |_: &[String]| Ok(vec!["Guests used the room.".to_owned()]);
    let error = remove_text_invisible(
        "Visitors used the room.",
        None,
        InvisibleOptions {
            document_rewriter: Some(&mut rewrite),
            document_repairer: Some(&mut repair),
            ..InvisibleOptions::default()
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("document_validator"));
}

#[test]
fn document_veto_applies_to_passage_rewrites() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("output.txt");
    let source = "Visitors used the room.\n\nReaders borrowed books.";
    let seen = RefCell::new(Vec::new());
    let mut yesterday = closure(|text| format!("Yesterday, {}", text.to_lowercase()));
    let mut validate = |before: &str, after: &str| {
        seen.borrow_mut().push((before.to_owned(), after.to_owned()));
        Ok(false)
    };
    let error = remove_text_invisible(
        source,
        Some(&output),
        InvisibleOptions {
            rewriter: Some(&mut yesterday),
            allow_partial: true,
            document_validator: Some(&mut validate),
            ..InvisibleOptions::default()
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("document_validator"));
    assert!(!output.exists());
    assert_eq!(
        *seen.borrow(),
        vec![(
            source.to_owned(),
            "Yesterday, visitors used the room.\n\nYesterday, readers borrowed books.".to_owned()
        )]
    );
}

/// A scripted generator standing in for a local causal LM.
struct ScriptedGenerator {
    candidates: Vec<String>,
    truncated: bool,
    requests: Vec<(u64, usize)>,
}

impl ScriptedGenerator {
    fn new(candidates: &[&str]) -> Self {
        Self {
            candidates: candidates.iter().rev().map(|text| (*text).to_owned()).collect(),
            truncated: false,
            requests: Vec::new(),
        }
    }
}

impl TextGenerator for ScriptedGenerator {
    fn model_id(&self) -> &'static str {
        "scripted"
    }

    fn prompt_token_count(
        &mut self,
        messages: &[ChatMessage],
        _plain: &str,
    ) -> Result<Option<usize>, TextError> {
        assert_eq!(messages[0].content, raiw_text::TEXT_REWRITE_SYSTEM_PROMPT);
        Ok(Some(3))
    }

    fn generate(&mut self, request: &GenerationRequest<'_>) -> Result<Generation, TextError> {
        self.requests.push((request.seed, request.max_new_tokens));
        let text = if self.candidates.len() > 1 {
            self.candidates.pop().unwrap_or_default()
        } else {
            self.candidates[0].clone()
        };
        Ok(Generation {
            text,
            finish_reason: if self.truncated {
                FinishReason::Length
            } else {
                FinishReason::Stop
            },
        })
    }
}

fn local(
    text: &str,
    generator: &mut ScriptedGenerator,
    language: &dyn LanguageClassifier,
) -> Result<raiw_text::TextInvisibleResult, TextError> {
    remove_text_invisible(
        text,
        None,
        InvisibleOptions {
            generator: Some(generator),
            language: Some(language),
            seed: 7,
            ..InvisibleOptions::default()
        },
    )
}

#[test]
fn local_generator_refuses_fact_drift_echo_and_placeholders() {
    for candidate in [
        "The shipment contains 99 units.",
        "The shipment contains 12 units.",
        "",
        "This shipment holds 12 units. [[RAIW_SPAN_N]]",
        "The shipment contains at least [[RAIW_SPAN_0]] units.",
    ] {
        let mut generator = ScriptedGenerator::new(&[candidate]);
        let error = local("The shipment contains 12 units.", &mut generator, &english).unwrap_err();
        assert!(error.to_string().contains("rewrite"), "{candidate:?}: {error}");
        assert_eq!(generator.requests, vec![(7, 64), (8, 64), (9, 64)]);
    }
    let mut generator = ScriptedGenerator::new(&["This shipment holds [[RAIW_SPAN_0]] units."]);
    let result = local("The shipment contains 12 units.", &mut generator, &english).unwrap();
    assert_eq!(result.cleaned_text, "This shipment holds 12 units.");
    assert_eq!(result.model_id, "scripted");
}

#[test]
fn local_generator_rejects_truncation_and_script_changes() {
    let mut truncated = ScriptedGenerator::new(&["The archive lends books to readers."]);
    truncated.truncated = true;
    assert!(
        local(
            "Readers can borrow books from the archive.",
            &mut truncated,
            &english
        )
        .is_err()
    );

    let mut script = ScriptedGenerator::new(&["A mother reads a book to her child."]);
    assert!(local("мама читает книгу своему ребёнку.", &mut script, &english).is_err());

    let passage = "the archive lends books to readers and holds public workshops. its rooms provide a place to study and meet neighbors. ".repeat(4);
    let mut collapse = ScriptedGenerator::new(&["The archive lends books."]);
    assert!(local(&passage, &mut collapse, &english).is_err());
    let kept = passage.replace("lends", "loans").replace("holds", "hosts");
    let mut faithful = ScriptedGenerator::new(&[&kept]);
    assert_eq!(
        local(&passage, &mut faithful, &english).unwrap().cleaned_text,
        format!("{} ", kept.trim())
    );
}

#[test]
fn local_generator_retries_after_validator_rejection() {
    let mut generator = ScriptedGenerator::new(&[
        "Tourists used the reading room.",
        "The reading room was used by visitors.",
    ]);
    let seen = RefCell::new(Vec::new());
    let result = remove_text_invisible(
        "Visitors used the reading room.",
        None,
        InvisibleOptions {
            generator: Some(&mut generator),
            language: Some(&english),
            rewrite_validator: Some(Box::new(|source: &str, candidate: &str| {
                seen.borrow_mut().push((source.to_owned(), candidate.to_owned()));
                Ok(!candidate.contains("Tourists"))
            })),
            ..InvisibleOptions::default()
        },
    )
    .unwrap();
    assert_eq!(result.cleaned_text, "The reading room was used by visitors.");
    assert_eq!(seen.borrow().len(), 2);
    assert!(
        seen.borrow()
            .iter()
            .all(|(source, _)| source == "Visitors used the reading room.")
    );
}

#[test]
fn missing_generator_is_a_runtime_error_and_option_conflicts_are_rejected() {
    let error = remove_text_invisible("Some prose.", None, InvisibleOptions::default()).unwrap_err();
    assert!(error.is_runtime());
    assert_eq!(
        remove_text_invisible("`code`", None, InvisibleOptions::default())
            .unwrap()
            .model_id,
        raiw_text::DEFAULT_TEXT_REWRITER_MODEL
    );
    let mut repair = |_: &[String], _: &[String]| Ok(None);
    let error = remove_text_invisible(
        "Text.",
        None,
        InvisibleOptions {
            document_repairer: Some(&mut repair),
            ..InvisibleOptions::default()
        },
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("document_repairer requires document_rewriter and document_validator")
    );
    let mut rewrite = |_: &[String]| Ok(vec![]);
    let mut custom = closure(str::to_owned);
    let error = remove_text_invisible(
        "Text.",
        None,
        InvisibleOptions {
            document_rewriter: Some(&mut rewrite),
            rewriter: Some(&mut custom),
            ..InvisibleOptions::default()
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("cannot be combined"));
}

#[test]
fn language_guard_matrix() {
    let source =
        "The library hosted an open meeting where volunteers explained how to preserve old documents.";
    let candidate = "La bibliothèque a organisé une réunion ouverte pour expliquer la conservation des documents anciens.";
    for (source_label, source_score, target_label, target_score, expected) in [
        ("en", 0.99, "fr", 0.99, false),
        ("bg", 0.99, "ru", 0.99, false),
        ("en", 0.95, "es", 0.8, false),
        ("en", 0.949, "fr", 0.99, true),
        ("en", 0.99, "fr", 0.799, true),
        ("en", 0.99, "en", 0.8, true),
        ("und", 0.99, "en", 0.99, true),
        ("zxx", 0.99, "en", 0.99, true),
        ("en", 0.99, "zxx", 0.99, false),
    ] {
        let answers = RefCell::new(vec![(target_label, target_score), (source_label, source_score)]);
        let classifier = |_: &str| {
            let (label, score) = answers.borrow_mut().pop().unwrap_or(("en", 0.99));
            Ok((label.to_owned(), score))
        };
        assert_eq!(
            guard::preserves_rewrite_language(source, candidate, &classifier).unwrap(),
            expected
        );
    }
    let unexpected =
        |_: &str| -> Result<(String, f64), TextError> { panic!("short samples must not be classified") };
    assert!(guard::preserves_rewrite_language("Bonjour", "Hello", &unexpected).unwrap());
    assert!(
        guard::preserves_rewrite_language(&"[[RAIW_SPAN_0]] ".repeat(30), candidate, &unexpected).unwrap()
    );
    let seen = RefCell::new(Vec::new());
    let recording = |text: &str| {
        seen.borrow_mut().push(text.to_owned());
        Ok(("en".to_owned(), 0.99))
    };
    assert!(
        guard::preserves_rewrite_language(
            &format!("{} [[RAIW_SPAN_0]]", "e\u{301}".repeat(40)),
            &format!("{} [[RAIW_SPAN_1]]", "a".repeat(40)),
            &recording
        )
        .unwrap()
    );
    assert_eq!(*seen.borrow(), vec!["é".repeat(40), "a".repeat(40)]);

    let french = "La bibliothèque a organisé une rencontre publique. Les bénévoles ont présenté des méthodes pour conserver les vieux documents. Les participants ont reçu une fiche explicative.";
    let english_candidate = "The library held an open meeting. Volunteers shared techniques for preserving old documents. Attendees received an explanation sheet.";
    let answers = RefCell::new(vec![("en", 0.99), ("fr", 1.0)]);
    let classifier = |_: &str| {
        let (label, score) = answers.borrow_mut().pop().unwrap_or(("en", 0.99));
        Ok((label.to_owned(), score))
    };
    let facts = indexmap::IndexMap::new();
    assert_eq!(
        guard::guard_local_rewrite_candidate(french, french, &facts, english_candidate, None, &classifier)
            .unwrap(),
        None
    );
}

#[test]
fn fact_guard_spot_checks() {
    assert!(verify_text_rewrite_facts(
        "Cycling is one of the most efficient options.",
        "Cycling is a highly efficient option."
    ));
    assert!(!verify_text_rewrite_facts(
        "We shipped one crate.",
        "We shipped crates."
    ));
    assert!(!verify_text_rewrite_facts(
        "NASA maintains the archive.",
        "Another organization maintains the archive."
    ));
    assert!(!verify_text_rewrite_facts(
        "Readers can\u{2019}t borrow books.",
        "Readers can borrow books."
    ));
    assert!(verify_text_rewrite_facts(
        "Users mustn\u{2019}t share keys.",
        "Users must not share keys."
    ));
    assert!(!guard::preserves_dominant_script(
        "Libraries support local community development.",
        "Libraries support local community развитие."
    ));
}

#[test]
fn clean_text_passes_through_the_combined_cleaner() {
    let path = Path::new("unused");
    assert!(!path.exists());
    let result = remove_text_all("Keep these two\u{a0}words together.", None, AllOptions::default()).unwrap();
    assert_eq!(result.cleaned_text, "Keep these two\u{a0}words together.");
    assert_eq!(result.token_watermark_action(), TokenWatermarkAction::NotRun);
    let marked = remove_text_all(
        "Keep \u{200b}\u{34f}\u{2060}\u{200b}these two\u{a0}words together.",
        None,
        AllOptions::default(),
    )
    .unwrap();
    assert_eq!(marked.cleaned_text, "Keep these two\u{a0}words together.");
}
