//! Chat Completions transport ports of `tests/test_text_api.py` against a local HTTP server.
#![cfg(all(feature = "api", feature = "langid"))]

use std::collections::HashMap;
use std::fmt::Write as _;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use raiw_text::api::{
    ApiPassageRewriter, ChatCompletionConfig, CompletionTokenField, DocumentApiRewriter, OllamaRoute,
    RewriteStyle, StrictResidualConfig, StrictResidualRewriter,
};
use raiw_text::{
    InvisibleOptions, PassageRewriter as _, TEXT_REWRITE_SYSTEM_PROMPT, TextError, remove_text_invisible,
};
use serde_json::{Value, json};

type Responder = Box<dyn Fn(&Value) -> (u16, Vec<u8>, Vec<(String, String)>) + Send>;

#[derive(Debug, Clone)]
struct Recorded {
    path: String,
    headers: HashMap<String, String>,
    payload: Value,
}

struct Endpoint {
    url: String,
    requests: Arc<Mutex<Vec<Recorded>>>,
    responder: Arc<Mutex<Responder>>,
}

fn completion(content: &str, finish: &str) -> Value {
    json!({"choices": [{"finish_reason": finish, "message": {"role": "assistant", "content": content}}]})
}

fn echo_shipped(payload: &Value) -> (u16, Vec<u8>, Vec<(String, String)>) {
    let content = payload["messages"][1]["content"]
        .as_str()
        .unwrap_or("")
        .replace("sent", "shipped");
    (
        200,
        completion(&content, "stop").to_string().into_bytes(),
        Vec::new(),
    )
}

impl Endpoint {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://127.0.0.1:{}/v1", listener.local_addr().unwrap().port());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let responder: Arc<Mutex<Responder>> = Arc::new(Mutex::new(Box::new(echo_shipped)));
        let (log, answer) = (Arc::clone(&requests), Arc::clone(&responder));
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 {
                    continue;
                }
                let path = line.split_whitespace().nth(1).unwrap_or("").to_owned();
                let mut headers = HashMap::new();
                loop {
                    let mut header = String::new();
                    reader.read_line(&mut header).unwrap();
                    let header = header.trim_end();
                    if header.is_empty() {
                        break;
                    }
                    if let Some((name, value)) = header.split_once(':') {
                        headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_owned());
                    }
                }
                let length: usize = headers
                    .get("content-length")
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(0);
                let mut body = vec![0; length];
                reader.read_exact(&mut body).unwrap();
                let payload: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
                log.lock().unwrap().push(Recorded {
                    path,
                    headers,
                    payload: payload.clone(),
                });
                let (status, raw, extra) = (answer.lock().unwrap())(&payload);
                let mut response = format!(
                    "HTTP/1.1 {status} Status\r\nContent-Length: {}\r\nConnection: close\r\n",
                    raw.len()
                );
                for (name, value) in extra {
                    let _ = write!(response, "{name}: {value}\r\n");
                }
                response.push_str("\r\n");
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.write_all(&raw);
            }
        });
        Self {
            url,
            requests,
            responder,
        }
    }

    fn respond(&self, responder: impl Fn(&Value) -> (u16, Vec<u8>, Vec<(String, String)>) + Send + 'static) {
        *self.responder.lock().unwrap() = Box::new(responder);
    }

    fn body(&self, status: u16, body: &Value) {
        let raw = body.to_string().into_bytes();
        self.respond(move |_| (status, raw.clone(), Vec::new()));
    }

    fn raw(&self, status: u16, raw: &[u8]) {
        let raw = raw.to_vec();
        self.respond(move |_| (status, raw.clone(), Vec::new()));
    }

    fn requests(&self) -> Vec<Recorded> {
        self.requests.lock().unwrap().clone()
    }

    fn config(&self) -> ChatCompletionConfig {
        ChatCompletionConfig::new(self.url.clone(), "test-model")
    }
}

#[allow(clippy::unnecessary_wraps)]
fn english(_: &str) -> Result<(String, f64), TextError> {
    Ok(("en".to_owned(), 0.99))
}

#[test]
fn endpoint_is_opt_in_and_protected_facts_roundtrip() {
    let endpoint = Endpoint::start();
    let config = ChatCompletionConfig {
        api_key: Some("synthetic-secret".into()),
        max_completion_tokens: 75,
        timeout: 5.0,
        ..endpoint.config()
    };
    let mut rewriter = ApiPassageRewriter::new(config, &english).unwrap();
    assert!(endpoint.requests().is_empty());
    let result = remove_text_invisible(
        "The curator Maya sent 12 crates.",
        None,
        InvisibleOptions {
            rewriter: Some(&mut rewriter),
            ..InvisibleOptions::default()
        },
    )
    .unwrap();
    assert_eq!(result.cleaned_text, "The curator Maya shipped 12 crates.");
    let requests = endpoint.requests();
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    assert_eq!(request.path, "/v1/chat/completions");
    assert_eq!(request.headers["authorization"], "Bearer synthetic-secret");
    assert_eq!(request.payload["model"], "test-model");
    assert_eq!(request.payload["max_completion_tokens"], 75);
    assert_eq!(request.payload["stream"], false);
    assert_eq!(
        request.payload["messages"][0]["content"],
        TEXT_REWRITE_SYSTEM_PROMPT
    );
    let masked = request.payload["messages"][1]["content"].as_str().unwrap();
    assert!(!masked.contains("12") && !masked.contains("Maya") && masked.contains("[[RAIW_SPAN_"));
}

#[test]
fn locked_text_and_missing_key() {
    let endpoint = Endpoint::start();
    let mut rewriter = ApiPassageRewriter::new(endpoint.config(), &english).unwrap();
    let result = remove_text_invisible(
        "Maya sent 12 crates, tagged `x=1`.",
        None,
        InvisibleOptions {
            rewriter: Some(&mut rewriter),
            ..InvisibleOptions::default()
        },
    )
    .unwrap();
    assert_eq!(result.cleaned_text, "Maya shipped 12 crates, tagged `x=1`.");
    assert!(!endpoint.requests()[0].headers.contains_key("authorization"));
}

#[test]
fn malformed_or_unacceptable_output_is_not_returned() {
    let bodies: Vec<Vec<u8>> = vec![
        b"not json".to_vec(),
        b"\xff".to_vec(),
        b"[]".to_vec(),
        b"{}".to_vec(),
        br#"{"choices": []}"#.to_vec(),
        br#"{"choices": [{}, {}]}"#.to_vec(),
        completion("Tools require repairs.", "length").to_string().into_bytes(),
        completion("Tools require repairs.", "content_filter").to_string().into_bytes(),
        json!({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Tools require repairs.", "refusal": "Cannot comply"}}]}).to_string().into_bytes(),
        json!({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Tools require repairs.", "tool_calls": [{"id": "call"}]}}]}).to_string().into_bytes(),
        json!({"choices": [{"finish_reason": "stop", "message": {"role": "tool", "content": "Tools require repairs."}}]}).to_string().into_bytes(),
        completion("", "stop").to_string().into_bytes(),
        json!({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": null}}]}).to_string().into_bytes(),
        completion("Tools need repairs.", "stop").to_string().into_bytes(),
        completion("Инструменты нуждаются в ремонте.", "stop").to_string().into_bytes(),
    ];
    for body in bodies {
        let endpoint = Endpoint::start();
        endpoint.raw(200, &body);
        let mut rewriter = ApiPassageRewriter::new(endpoint.config(), &english).unwrap();
        let error = rewriter.rewrite("Tools need repairs.").unwrap_err();
        assert!(error.is_runtime(), "{}", String::from_utf8_lossy(&body));
        assert_eq!(endpoint.requests().len(), 1);
    }
}

#[test]
fn http_failures_do_not_retry_or_expose_credentials() {
    for status in [429, 500, 307] {
        let endpoint = Endpoint::start();
        let key = "synthetic-\"secret";
        endpoint.body(status, &json!({"error": key}));
        let config = ChatCompletionConfig {
            api_key: Some(key.into()),
            ..endpoint.config()
        };
        let mut rewriter = ApiPassageRewriter::new(config, &english).unwrap();
        let error = rewriter.rewrite("Tools need repairs.").unwrap_err().to_string();
        assert_eq!(error, format!("Text API returned HTTP {status}."));
        assert!(!error.contains(key));
        assert_eq!(endpoint.requests().len(), 1);
    }
}

#[test]
fn reflected_credentials_are_never_returned() {
    let endpoint = Endpoint::start();
    let config = ChatCompletionConfig {
        api_key: Some("synthetic-secret".into()),
        ..endpoint.config()
    };
    endpoint.body(
        200,
        &completion("Tools require repairs. synthetic-secret", "stop"),
    );
    let mut rewriter = ApiPassageRewriter::new(config.clone(), &english).unwrap();
    assert!(
        rewriter
            .rewrite("Tools need repairs.")
            .unwrap_err()
            .to_string()
            .contains("configured credential")
    );
    let escaped = completion("Tools require repairs. synthetic-secret", "stop")
        .to_string()
        .replace("synthetic-secret", "synthetic-\\u0073ecret");
    endpoint.raw(200, escaped.as_bytes());
    assert!(
        rewriter
            .rewrite("Tools need repairs.")
            .unwrap_err()
            .to_string()
            .contains("configured credential")
    );
}

#[test]
fn invalid_configuration_is_rejected_before_requests() {
    let endpoint = Endpoint::start();
    let base = endpoint.config();
    let invalid = [
        ChatCompletionConfig {
            base_url: "file:///tmp/model".into(),
            ..base.clone()
        },
        ChatCompletionConfig {
            base_url: "https://user:secret@example.test/v1".into(),
            ..base.clone()
        },
        ChatCompletionConfig {
            base_url: "https://example.test/v1?token=secret".into(),
            ..base.clone()
        },
        ChatCompletionConfig {
            base_url: "https://example.test/v1#secret".into(),
            ..base.clone()
        },
        ChatCompletionConfig {
            model: String::new(),
            ..base.clone()
        },
        ChatCompletionConfig {
            api_key: Some("secret\r\nInjected: true".into()),
            ..base.clone()
        },
        ChatCompletionConfig {
            api_key: Some("secret\0".into()),
            ..base.clone()
        },
        ChatCompletionConfig {
            max_completion_tokens: 0,
            ..base.clone()
        },
        ChatCompletionConfig {
            timeout: 0.0,
            ..base.clone()
        },
        ChatCompletionConfig {
            timeout: f64::NAN,
            ..base.clone()
        },
        ChatCompletionConfig {
            temperature: Some(-0.1),
            ..base.clone()
        },
        ChatCompletionConfig {
            temperature: Some(f64::NAN),
            ..base
        },
    ];
    for config in invalid {
        let error = ApiPassageRewriter::new(config.clone(), &english).unwrap_err();
        assert!(matches!(error, TextError::InvalidArgument(_)), "{config:?}");
    }
    assert!(endpoint.requests().is_empty());
}

#[test]
fn outer_validator_retains_retry_policy() {
    let endpoint = Endpoint::start();
    let mut rewriter = ApiPassageRewriter::new(endpoint.config(), &english).unwrap();
    let error = remove_text_invisible(
        "Tools sent crates.",
        None,
        InvisibleOptions {
            rewriter: Some(&mut rewriter),
            rewrite_validator: Some(Box::new(|_: &str, _: &str| Ok(false))),
            ..InvisibleOptions::default()
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("validator rejected"));
    assert_eq!(endpoint.requests().len(), 3);
}

#[test]
fn document_api_draft_correction_and_review_use_two_posts() {
    let endpoint = Endpoint::start();
    endpoint.respond(|payload| {
        let data: Value = serde_json::from_str(payload["messages"][1]["content"].as_str().unwrap()).unwrap();
        assert_eq!(
            data["original"],
            json!(["Visitors used the room.", "Readers borrowed books."])
        );
        let slots = if data.get("rejected").is_some() {
            assert_eq!(
                data["rejected"],
                json!(["Staff used the room.", "Guests borrowed books."])
            );
            json!(["Guests used the room.", "Readers checked out books."])
        } else {
            json!(["Staff used the room.", "Guests borrowed books."])
        };
        (
            200,
            completion(&slots.to_string(), "stop").to_string().into_bytes(),
            Vec::new(),
        )
    });
    let config = DocumentApiRewriter::config(endpoint.url.clone(), "test-model");
    let mut rewrite = DocumentApiRewriter::new(config.clone(), RewriteStyle::Standard, false).unwrap();
    let mut repair = DocumentApiRewriter::new(config, RewriteStyle::Standard, false).unwrap();
    let seen = std::cell::RefCell::new(Vec::new());
    let mut validate = |_: &str, candidate: &str| {
        seen.borrow_mut().push(candidate.to_owned());
        Ok(candidate.starts_with("Guests used the room."))
    };
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("output.txt");
    let result = remove_text_invisible(
        "Visitors used the room.\r\n\r\n`value = 12`\r\n\r\nReaders borrowed books.",
        Some(&output),
        InvisibleOptions {
            document_rewriter: Some(&mut rewrite),
            document_validator: Some(&mut validate),
            document_repairer: Some(&mut repair),
            ..InvisibleOptions::default()
        },
    )
    .unwrap();
    let requests = endpoint.requests();
    assert_eq!(requests.len(), 2);
    for request in &requests {
        let prompt = request.payload["messages"][0]["content"].as_str().unwrap();
        assert!(prompt.contains("The input contains exactly 2 prose slots. Output exactly 2 JSON strings."));
    }
    assert_eq!(
        result.cleaned_text,
        "Guests used the room.\r\n\r\n`value = 12`\r\n\r\nReaders checked out books."
    );
    assert_eq!(std::fs::read_to_string(&output).unwrap(), result.cleaned_text);
}

#[test]
fn document_api_shapes_and_fences() {
    for response in [
        r#"["Only one."]"#,
        r#"{"0":"Staff used the room.","1":"Guests borrowed books."}"#,
        r#"[1,"Text"]"#,
    ] {
        let endpoint = Endpoint::start();
        endpoint.body(200, &completion(response, "stop"));
        let rewriter = DocumentApiRewriter::new(endpoint.config(), RewriteStyle::Standard, false).unwrap();
        let error = rewriter
            .rewrite(
                &["Visitors used the room.".into(), "Readers borrowed books.".into()],
                None,
            )
            .unwrap_err();
        assert!(error.to_string().contains("one nonempty string per prose span"));
    }
    for newline in ["\n", "\r\n"] {
        let endpoint = Endpoint::start();
        endpoint.body(
            200,
            &completion(
                &[" ```json", r#"["Guests used the room."]"#, "```  "].join(newline),
                "stop",
            ),
        );
        let rewriter = DocumentApiRewriter::new(endpoint.config(), RewriteStyle::Standard, false).unwrap();
        assert_eq!(
            rewriter
                .rewrite(&["Visitors used the room.".into()], None)
                .unwrap(),
            ["Guests used the room."]
        );
    }
    for response in [
        "Here is the result:\n```json\n[\"Text.\"]\n```",
        "```json\n[\"Text.\"]\n```\nMore text",
    ] {
        let endpoint = Endpoint::start();
        endpoint.body(200, &completion(response, "stop"));
        let rewriter = DocumentApiRewriter::new(endpoint.config(), RewriteStyle::Standard, false).unwrap();
        assert!(
            rewriter
                .rewrite(&["Visitors used the room.".into()], None)
                .unwrap_err()
                .to_string()
                .contains("invalid JSON")
        );
    }
}

#[test]
fn document_api_sends_recast_sampling_and_token_field() {
    let endpoint = Endpoint::start();
    endpoint.body(200, &completion(r#"["Guests used the room."]"#, "stop"));
    let config = ChatCompletionConfig {
        temperature: Some(0.0),
        seed: Some(7291),
        max_completion_tokens: 42,
        completion_token_field: CompletionTokenField::MaxTokens,
        ..endpoint.config()
    };
    let rewriter = DocumentApiRewriter::new(config, RewriteStyle::Recast, false).unwrap();
    assert_eq!(
        rewriter
            .rewrite(&["Visitors used the room.".into()], None)
            .unwrap(),
        ["Guests used the room."]
    );
    let payload = &endpoint.requests()[0].payload;
    let prompt = payload["messages"][0]["content"].as_str().unwrap();
    assert!(
        prompt.find("Rewrite every prose slot").unwrap()
            < prompt.find("The input contains exactly 1 prose slots").unwrap()
    );
    assert_eq!(
        (
            payload["temperature"].as_f64(),
            payload["seed"].as_i64(),
            payload["max_tokens"].as_i64()
        ),
        (Some(0.0), Some(7291), Some(42))
    );
    assert!(payload.get("max_completion_tokens").is_none());
}

#[test]
fn sentence_api_preserves_separators_and_fixed_shape() {
    let endpoint = Endpoint::start();
    let drafts = [
        "Dr. Reed measures with a ruler.",
        "The block measures 1.5 meters across.",
        "They record the height.",
    ];
    endpoint.respond(move |payload| {
        let data: Value = serde_json::from_str(payload["messages"][1]["content"].as_str().unwrap()).unwrap();
        assert_eq!(data["original"], json!(["Dr. Reed uses a ruler.", "The block is 1.5 meters wide.", "They note the height."]));
        assert_eq!(payload["response_format"]["json_schema"]["schema"], json!({"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 3, "maxItems": 3}));
        assert_eq!(payload["reasoning_effort"], "none");
        (200, completion(&json!(drafts).to_string(), "stop").to_string().into_bytes(), Vec::new())
    });
    let config = ChatCompletionConfig {
        disable_reasoning: true,
        ..endpoint.config()
    };
    let rewriter = DocumentApiRewriter::new(config, RewriteStyle::Sentence, true).unwrap();
    let original = vec![
        "Dr. Reed uses a ruler.\tThe block is 1.5 meters wide.".to_owned(),
        "They note the height.".to_owned(),
    ];
    let rejected = vec![format!("{}\t{}", drafts[0], drafts[1]), drafts[2].to_owned()];
    assert_eq!(rewriter.rewrite(&original, None).unwrap(), rejected);
    assert_eq!(rewriter.rewrite(&original, Some(&rejected)).unwrap(), rejected);
    let error = rewriter
        .rewrite(
            &original,
            Some(&["Merged sentence.".into(), "They record the height.".into()]),
        )
        .unwrap_err();
    assert!(error.to_string().contains("same sentence slots"));
    assert_eq!(endpoint.requests().len(), 2);

    for extra in [
        "Two sentences. Here.",
        "Extra\nline",
        "Extra\rline",
        "The room empties",
    ] {
        let endpoint = Endpoint::start();
        endpoint.body(
            200,
            &completion(&json!([extra, "The door swings shut."]).to_string(), "stop"),
        );
        let rewriter = DocumentApiRewriter::new(endpoint.config(), RewriteStyle::Sentence, false).unwrap();
        let error = rewriter
            .rewrite(&["The room emptied. The door closed.".into()], None)
            .unwrap_err();
        assert!(error.to_string().contains("sentence boundary"), "{extra:?}");
        assert!(endpoint.requests()[0].payload.get("response_format").is_none());
    }
}

#[test]
fn strict_residual_resends_only_restored_sentences() {
    for refuse in [false, true] {
        let endpoint = Endpoint::start();
        endpoint.respond(move |payload| {
            let slots = if payload["seed"] == 7291 {
                json!([
                    "The crew counted 13 crates.",
                    "Staff will enter the seal number in the log."
                ])
            } else if refuse {
                json!(["The crew tallied 13 crates."])
            } else {
                json!(["The crew tallied 12 crates."])
            };
            (
                200,
                completion(&slots.to_string(), "stop").to_string().into_bytes(),
                Vec::new(),
            )
        });
        let config = StrictResidualConfig {
            seed: 7291,
            ..StrictResidualConfig::new(endpoint.url.clone(), "local")
        };
        let mut strict = StrictResidualRewriter::new(&config).unwrap();
        let outcome = remove_text_invisible(
            "The crew counted 12 crates.\n\nStaff will record the seal number in the log.",
            None,
            InvisibleOptions {
                document_rewriter: Some(&mut strict.rewriter),
                document_validator: Some(&mut strict.validator),
                document_repairer: Some(&mut strict.repairer),
                document_refiner: Some(&mut strict.refiner),
                ..InvisibleOptions::default()
            },
        );
        if refuse {
            assert!(
                outcome
                    .unwrap_err()
                    .to_string()
                    .contains("Strict residual rewrite")
            );
        } else {
            assert_eq!(
                outcome.unwrap().cleaned_text,
                "The crew tallied 12 crates.\n\nStaff will enter the seal number in the log."
            );
        }
        let requests = endpoint.requests();
        assert_eq!(
            requests
                .iter()
                .map(|request| request.payload["seed"].as_i64().unwrap())
                .collect::<Vec<_>>(),
            [7291, 84237]
        );
        let data: Value =
            serde_json::from_str(requests[1].payload["messages"][1]["content"].as_str().unwrap()).unwrap();
        assert_eq!(data["original"], json!(["The crew counted 12 crates."]));
    }
}

#[test]
fn ollama_generator_drives_the_guarded_local_rewriter() {
    use raiw_text::api::OllamaGenerator;
    let endpoint = Endpoint::start();
    let calls = Arc::new(Mutex::new(0));
    let counter = Arc::clone(&calls);
    endpoint.respond(move |payload| {
        let mut count = counter.lock().unwrap();
        *count += 1;
        let content = payload["messages"][1]["content"]
            .as_str()
            .unwrap()
            .replace("sent", "shipped");
        let finish = if *count == 1 { "length" } else { "stop" };
        (
            200,
            completion(&content, finish).to_string().into_bytes(),
            Vec::new(),
        )
    });
    let mut generator = OllamaGenerator::new(Some(&endpoint.url), "local-model").unwrap();
    let result = remove_text_invisible(
        "The curator Maya sent 12 crates.",
        None,
        InvisibleOptions {
            generator: Some(&mut generator),
            language: Some(&english),
            seed: 5,
            ..InvisibleOptions::default()
        },
    )
    .unwrap();
    assert_eq!(result.cleaned_text, "The curator Maya shipped 12 crates.");
    assert_eq!(result.model_id, "local-model");
    let requests = endpoint.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(
        requests
            .iter()
            .map(|request| request.payload["seed"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        [5, 6]
    );
    let first = &requests[0].payload;
    assert_eq!(
        (
            first["max_tokens"].as_u64(),
            first["top_p"].as_f64().map(|value| (value * 10.0).round())
        ),
        (Some(1024), Some(9.0))
    );
    assert!(OllamaGenerator::new(Some("http://10.0.0.1:11434/v1"), "m").is_err());
}

#[test]
fn ollama_route_validates_loopback_and_unchanged_slots() {
    assert!(
        OllamaRoute::new(Some("http://example.com/v1"), Some("local"), 0, false, false)
            .unwrap_err()
            .to_string()
            .contains("loopback")
    );
    assert!(
        OllamaRoute::new(None, None, 0, false, false)
            .unwrap_err()
            .to_string()
            .contains("explicit --rewriter-model")
    );
    assert!(
        OllamaRoute::new(None, Some("m"), 0, true, true)
            .unwrap_err()
            .to_string()
            .contains("--allow-partial")
    );

    let endpoint = Endpoint::start();
    endpoint.body(
        200,
        &completion(r#"["People visit the library.", "It closes at noon."]"#, "stop"),
    );
    let mut route = OllamaRoute::new(Some(&endpoint.url), Some("local-test"), 7291, false, false).unwrap();
    let error = remove_text_invisible(
        "Readers visit the library.\n\nIt closes at noon.",
        None,
        route.options(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("unchanged prose span"));
    let mut partial = OllamaRoute::new(Some(&endpoint.url), Some("local-test"), 7291, true, false).unwrap();
    let result = remove_text_invisible(
        "Readers visit the library.\n\nIt closes at noon.",
        None,
        partial.options(),
    )
    .unwrap();
    assert_eq!(
        result.cleaned_text,
        "People visit the library.\n\nIt closes at noon."
    );
    assert_eq!(result.retained_prose_spans, 1);

    let endpoint = Endpoint::start();
    endpoint.body(
        200,
        &completion(r#"["People visit the library.", "It shuts at noon."]"#, "stop"),
    );
    let mut route = OllamaRoute::new(Some(&endpoint.url), Some("local-test"), 7291, false, false).unwrap();
    let result = remove_text_invisible(
        "Readers visit the library.\tIt closes at noon.\r\n",
        None,
        route.options(),
    )
    .unwrap();
    assert_eq!(
        result.cleaned_text,
        "People visit the library.\tIt shuts at noon.\r\n"
    );
    let payload = &endpoint.requests()[0].payload;
    assert_eq!(
        (
            payload["seed"].as_i64(),
            payload["response_format"]["type"].as_str()
        ),
        (Some(7291), Some("json_schema"))
    );
    assert!(!endpoint.requests()[0].headers.contains_key("authorization"));

    let endpoint = Endpoint::start();
    endpoint.body(200, &completion(r#"["Readers checked out 19 books."]"#, "stop"));
    let mut route = OllamaRoute::new(Some(&endpoint.url), Some("local-test"), 0, false, false).unwrap();
    let error = remove_text_invisible("Readers borrowed 12 books.", None, route.options()).unwrap_err();
    assert!(error.to_string().contains("document_validator"));
}
