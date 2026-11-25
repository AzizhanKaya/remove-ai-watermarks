//! `rs-ai-watermarks mcp`: protocol layer, CLI mapping and the catalog (port of `tests/test_mcp_server.py`).

use std::io;
use std::sync::{Arc, Mutex};

use clap::CommandFactory;
use raiw_mcp::{Completed, Runner, Server};
use rs_ai_watermarks::cli::{Cli, mcp_server, operation_catalog};
use serde_json::{Value, json};

#[derive(Debug, Default)]
struct Recorder {
    calls: Mutex<Vec<Vec<String>>>,
    result: Option<Completed>,
}

#[derive(Debug, Clone)]
struct Shared(Arc<Recorder>);

impl Runner for Shared {
    fn run(&self, args: &[String], _stdin: Option<&str>) -> io::Result<Completed> {
        self.0.calls.lock().unwrap().push(args.to_vec());
        Ok(self.0.result.clone().unwrap_or(Completed {
            code: 0,
            stdout: String::new(),
            stderr: String::new(),
        }))
    }
}

fn recorder(code: i32, stdout: &str, stderr: &str) -> (Server, Arc<Recorder>) {
    let recorder = Arc::new(Recorder {
        calls: Mutex::default(),
        result: Some(Completed {
            code,
            stdout: stdout.into(),
            stderr: stderr.into(),
        }),
    });
    (mcp_server(Box::new(Shared(recorder.clone()))), recorder)
}

fn calls(recorder: &Recorder) -> Vec<Vec<String>> {
    recorder.calls.lock().unwrap().clone()
}

fn args(value: Value) -> serde_json::Map<String, Value> {
    match value {
        Value::Object(map) => map,
        other => panic!("arguments must be an object: {other}"),
    }
}

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

/// Every subcommand and flag a tool builds must exist in the real clap tree.
fn assert_the_cli_accepts(args: &[String]) {
    let mut command = Cli::command();
    command.build();
    let mut position = 0;
    let mut current = &command;
    while current.has_subcommands() {
        current = current
            .find_subcommand(&args[position])
            .unwrap_or_else(|| panic!("no subcommand {}", args[position]));
        position += 1;
    }
    let mut options = Vec::new();
    for arg in current.get_arguments() {
        options.extend(arg.get_long().map(|l| format!("--{l}")));
        options.extend(arg.get_short().map(|s| format!("-{s}")));
    }
    let end = args.iter().position(|a| a == "--").unwrap();
    for flag in args[position..end].iter().filter(|a| a.starts_with('-')) {
        assert!(options.contains(flag), "{args:?}: unknown flag {flag}");
    }
}

#[test]
fn audio_tools_map_tracks_and_explicit_verification_options() {
    let (server, rec) = recorder(0, r#"{"verified_removed": null}"#, "");
    let result = server.call_tool(
        "audio_invisible",
        &args(json!({"path": "speech.wav", "stream": 1, "force": true, "verify": false})),
    );
    assert_eq!(
        calls(&rec),
        vec![strings(&[
            "audio",
            "invisible",
            "--json",
            "--stream",
            "1",
            "-o",
            "speech_processed.wav",
            "--force",
            "--no-verify",
            "--",
            "speech.wav",
        ])]
    );
    assert_eq!(
        result["structuredContent"]["report"]["verified_removed"],
        Value::Null
    );
    assert_the_cli_accepts(&calls(&rec)[0]);

    let (server, rec) = recorder(0, "{}", "");
    server.call_tool(
        "audio_identify",
        &args(json!({"path": "speech.wav", "check_invisible": false})),
    );
    assert_eq!(
        calls(&rec),
        vec![strings(&[
            "audio",
            "identify",
            "--json",
            "--stream",
            "0",
            "--no-invisible",
            "--",
            "speech.wav"
        ])]
    );
}

#[test]
fn audio_tools_reject_invalid_track_ordinals() {
    for stream in [json!(-1), json!(true), json!("0"), json!(0.5)] {
        let (server, rec) = recorder(0, "", "");
        let result = server.call_tool(
            "audio_identify",
            &args(json!({"path": "a.wav", "stream": stream})),
        );
        assert_eq!(result["isError"], true);
        assert!(calls(&rec).is_empty());
    }
}

#[test]
fn audio_tools_preserve_explicit_silentcipher_scope() {
    for tool in ["audio_identify", "audio_invisible", "video_audio"] {
        let (server, rec) = recorder(0, "{}", "");
        let mut arguments = json!({"path": if tool == "video_audio" { "clip.mkv" } else { "speech.wav" }, "scheme": "silentcipher"});
        if tool != "audio_identify" {
            arguments["output"] = json!(if tool == "video_audio" {
                "processed.mkv"
            } else {
                "processed.wav"
            });
        }
        let result = server.call_tool(tool, &args(arguments));
        assert_ne!(result["isError"], true, "{tool}");
        let command = &calls(&rec)[0];
        let at = command.iter().position(|a| a == "--scheme").unwrap();
        assert_eq!(command[at + 1], "silentcipher");
        assert_the_cli_accepts(command);
    }
}

#[test]
fn audio_tool_refuses_source_output_and_skips_report_no_output() {
    let (server, rec) = recorder(0, "", "");
    let result = server.call_tool(
        "audio_invisible",
        &args(json!({"path": "a.wav", "output": "a.wav"})),
    );
    assert_eq!(result["isError"], true);
    assert!(calls(&rec).is_empty());

    let (server, _) = recorder(2, r#"{"skipped": true, "output": null}"#, "");
    let result = server.call_tool("audio_invisible", &args(json!({"path": "a.wav"})));
    assert_eq!(result["isError"], false);
    assert_eq!(result["structuredContent"]["output"], Value::Null);
    assert_eq!(result["structuredContent"]["report"]["skipped"], true);
}

#[test]
fn video_audio_tool_maps_to_the_mux_command_and_catalog() {
    let (server, rec) = recorder(0, "{}", "");
    server.call_tool("video_audio", &args(json!({"path": "input.mp4", "stream": 1})));
    assert_eq!(
        calls(&rec),
        vec![strings(&[
            "video",
            "audio",
            "--json",
            "--stream",
            "1",
            "-o",
            "input_audio_processed.mkv",
            "--",
            "input.mp4"
        ])]
    );
    let catalog = operation_catalog();
    let operation = catalog.iter().find(|op| op.id == "cli_video_audio").unwrap();
    let (planned, output) = operation
        .plan(&json!({"source": "input.mov", "output": "processed.mp4", "stream": 1}))
        .unwrap();
    assert_eq!(output.as_deref(), Some("processed.mp4"));
    let at = planned.iter().position(|a| a == "-o").unwrap();
    assert_eq!(planned[at + 1], "processed.mp4");
    assert_the_cli_accepts(&planned);
    let parameter = operation.parameters.iter().find(|p| p.name == "output").unwrap();
    assert!(
        parameter.schema["description"]
            .as_str()
            .unwrap()
            .contains("MP4 with AAC")
    );
}

#[test]
fn neural_audio_options_are_forwarded_and_invalid_bandwidth_refused() {
    let (server, rec) = recorder(0, "{}", "");
    let result = server.call_tool(
        "audio_invisible",
        &args(json!({"path": "input.wav", "method": "encodec", "bandwidth": 6})),
    );
    assert_eq!(result["isError"], false);
    let call = &calls(&rec)[0];
    assert_eq!(
        &call[call.len() - 6..],
        strings(&["--method", "encodec", "--bandwidth", "6", "--", "input.wav"])
    );

    let (server, rec) = recorder(0, "{}", "");
    server.call_tool(
        "audio_invisible",
        &args(json!({"path": "input.wav", "method": "dac", "vocoder_device": "mps"})),
    );
    let call = &calls(&rec)[0];
    assert_eq!(
        &call[call.len() - 6..],
        strings(&["--method", "dac", "--vocoder-device", "mps", "--", "input.wav"])
    );

    let (server, rec) = recorder(0, "", "");
    let result = server.call_tool(
        "audio_invisible",
        &args(json!({"path": "input.wav", "bandwidth": 7})),
    );
    assert_eq!(result["isError"], true);
    assert!(calls(&rec).is_empty());
}

fn request(method: &str, params: Option<Value>) -> Value {
    let mut message = json!({"jsonrpc": "2.0", "id": 1, "method": method});
    if let Some(params) = params {
        message["params"] = params;
    }
    message
}

#[test]
fn initialize_negotiates_a_supported_protocol_version() {
    let (server, _) = recorder(0, "", "");
    for (requested, negotiated) in [
        ("2025-06-18", "2025-06-18"),
        ("2024-11-05", "2024-11-05"),
        ("2099-01-01", "2025-11-25"),
    ] {
        let response = server
            .handle(&request(
                "initialize",
                Some(json!({"protocolVersion": requested, "capabilities": {}})),
            ))
            .unwrap();
        assert_eq!(response["result"]["protocolVersion"], negotiated);
        assert_eq!(
            response["result"]["capabilities"],
            json!({"tools": {"listChanged": false}})
        );
        assert!(
            response["result"]["instructions"]
                .as_str()
                .unwrap()
                .contains("third-party")
        );
    }
}

#[test]
fn notifications_get_no_reply_and_unknown_methods_get_method_not_found() {
    let (server, _) = recorder(0, "", "");
    assert!(
        server
            .handle(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"}))
            .is_none()
    );
    assert_eq!(
        server.handle(&request("server/discover", None)).unwrap()["error"]["code"],
        -32601
    );
}

#[test]
fn tools_list_exposes_convenience_and_all_catalog_tools() {
    let (server, _) = recorder(0, "", "");
    let response = server.handle(&request("tools/list", None)).unwrap();
    let tools = response["result"]["tools"].as_array().unwrap();
    let legacy = raiw_mcp::tools::convenience_tools();
    let names: Vec<_> = tools.iter().map(|t| t["name"].clone()).collect();
    assert_eq!(
        names[..legacy.len()],
        legacy.iter().map(|t| t["name"].clone()).collect::<Vec<_>>()[..]
    );
    assert_eq!(tools.len(), legacy.len() + operation_catalog().len());
    assert_eq!(tools[0]["annotations"]["readOnlyHint"], true);
    for tool in tools {
        assert_eq!(tool["inputSchema"]["type"], "object");
        assert_eq!(tool["inputSchema"]["additionalProperties"], false);
    }
}

#[test]
fn identify_maps_to_the_cli() {
    for (arguments, expected) in [
        (
            json!({"path": "a.png"}),
            vec!["identify", "--json", "--", "a.png"],
        ),
        (
            json!({"path": "a.png", "check_visible": false}),
            vec!["identify", "--json", "--no-visible", "--", "a.png"],
        ),
        (
            json!({"path": "clip.MP4"}),
            vec!["video", "identify", "--json", "--", "clip.MP4"],
        ),
        (
            json!({"path": "note.md"}),
            vec!["text", "identify", "--json", "--", "note.md"],
        ),
    ] {
        let (server, rec) = recorder(0, r#"{"is_ai_generated": true}"#, "");
        let result = server.call_tool("identify", &args(arguments));
        assert_eq!(calls(&rec), vec![strings(&expected)]);
        assert_the_cli_accepts(&strings(&expected));
        assert_eq!(result["isError"], false);
        assert_eq!(
            result["structuredContent"]["report"],
            json!({"is_ai_generated": true})
        );
    }
}

#[test]
fn remove_maps_to_the_cli_and_always_names_an_output() {
    for (arguments, expected) in [
        (
            json!({"path": "d/a.png"}),
            vec!["metadata", "--remove", "-o", "d/a_clean.png", "--", "d/a.png"],
        ),
        (
            json!({"path": "a.jpg", "mode": "visible"}),
            vec!["visible", "-o", "a_clean.jpg", "--", "a.jpg"],
        ),
        (
            json!({"path": "a.png", "mode": "all", "output": "o.png"}),
            vec!["all", "-o", "o.png", "--", "a.png"],
        ),
        (
            json!({"path": "c.mov"}),
            vec![
                "video",
                "metadata",
                "--remove",
                "-o",
                "c_clean.mov",
                "--",
                "c.mov",
            ],
        ),
        (
            json!({"path": "c.mp4", "mode": "visible"}),
            vec!["video", "visible", "-o", "c_clean.mp4", "--", "c.mp4"],
        ),
        (
            json!({"path": "doc.md"}),
            vec![
                "text",
                "metadata",
                "--remove",
                "-o",
                "doc_clean.md",
                "--",
                "doc.md",
            ],
        ),
        (
            json!({"path": "doc.md", "mode": "all"}),
            vec!["text", "all", "-o", "doc_clean.md", "--", "doc.md"],
        ),
    ] {
        let (server, rec) = recorder(0, "", "");
        let result = server.call_tool("remove", &args(arguments));
        assert_eq!(calls(&rec), vec![strings(&expected)]);
        assert_the_cli_accepts(&strings(&expected));
        let at = expected.iter().position(|a| *a == "-o").unwrap();
        assert_eq!(result["structuredContent"]["output"], expected[at + 1]);
    }
}

#[test]
fn batch_maps_to_the_cli() {
    let default_out = raiw_core::resolve_lenient(std::path::Path::new("in_clean"))
        .to_string_lossy()
        .into_owned();
    for (arguments, expected) in [
        (
            json!({"directory": "in"}),
            vec![
                "batch",
                "--mode",
                "metadata",
                "-o",
                default_out.as_str(),
                "--",
                "in",
            ],
        ),
        (
            json!({"directory": "in", "media": "video", "mode": "visible", "output_dir": "out"}),
            vec!["video", "batch", "--mode", "visible", "-o", "out", "--", "in"],
        ),
    ] {
        let (server, rec) = recorder(0, "", "");
        let result = server.call_tool("batch", &args(arguments));
        assert_eq!(calls(&rec), vec![strings(&expected)]);
        let at = expected.iter().position(|a| *a == "-o").unwrap();
        assert_eq!(result["structuredContent"]["output"], expected[at + 1]);
        assert_the_cli_accepts(&strings(&expected));
    }
}

#[test]
fn invalid_arguments_are_refused_before_anything_runs() {
    for (tool, arguments) in [
        ("remove", json!({"path": "a.png", "output": "./a.png"})),
        ("remove", json!({"path": "a.png", "mode": "invisible"})),
        ("remove", json!({"path": ""})),
        ("remove", json!({})),
        ("identify", json!({"path": "a.png", "check_visible": "no"})),
        ("identify", json!({"path": "a.png", "shell": "rm -rf /"})),
        ("batch", json!({"directory": "in", "output_dir": "in"})),
    ] {
        let (server, rec) = recorder(0, "", "");
        let result = server.call_tool(tool, &args(arguments.clone()));
        assert!(calls(&rec).is_empty(), "{tool} {arguments}");
        assert_eq!(result["isError"], true, "{tool} {arguments}");
    }
}

#[test]
fn only_failures_are_tool_errors() {
    for (code, stderr, is_error) in [
        (0, "", false),
        (2, "", false),
        (1, "Error: boom", true),
        (
            2,
            "Usage: cli metadata [OPTIONS] SOURCE\nError: Invalid value for 'SOURCE'",
            true,
        ),
    ] {
        let (server, _) = recorder(code, "table", stderr);
        let result = server.call_tool("remove", &args(json!({"path": "a.png"})));
        assert_eq!(result["isError"], is_error);
        assert_eq!(result["structuredContent"]["exit_code"], code);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert_eq!(text.contains("not proof of a clean file"), code == 2 && !is_error);
        if is_error {
            assert!(text.contains(stderr.lines().last().unwrap()));
        }
    }
}

#[test]
fn serve_answers_line_by_line_and_reports_parse_errors() {
    let (server, _) = recorder(0, "", "");
    let input = format!(
        "{}\n\nnot json\n{}\n",
        json!({"jsonrpc": "2.0", "id": 7, "method": "ping"}),
        json!({"jsonrpc": "2.0", "method": "notifications/initialized"})
    );
    let mut output = Vec::new();
    server.serve(input.as_bytes(), &mut output).unwrap();
    let replies: Vec<Value> = String::from_utf8(output)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(
        replies,
        vec![
            json!({"jsonrpc": "2.0", "id": 7, "result": {}}),
            json!({"jsonrpc": "2.0", "id": null, "error": {"code": -32700, "message": "Parse error"}}),
        ]
    );
}

#[test]
fn video_message_tools_preserve_known_reference() {
    let reference = "a".repeat(64);
    let (server, rec) = recorder(0, "{}", "");
    server.call_tool(
        "video_message_remove",
        &args(json!({"path": "in.mp4", "reference": reference, "method": "vae", "noise_std": 0.25})),
    );
    let call = &calls(&rec)[0];
    assert_the_cli_accepts(call);
    assert!(call.windows(2).any(|w| w == ["--noise-std", "0.25"]));
    assert!(
        call.windows(2)
            .any(|w| w[0] == "-o" && w[1] == "in_message_processed.mp4")
    );

    let (server, rec) = recorder(0, "", "");
    let result = server.call_tool(
        "video_message_remove",
        &args(json!({"path": "in.mp4", "reference": reference, "noise_std": 0.25})),
    );
    assert_eq!(result["isError"], true);
    assert!(calls(&rec).is_empty());
}

#[test]
fn catalog_covers_every_cli_leaf_with_typed_schemas() {
    let catalog = operation_catalog();
    let ids: Vec<&str> = catalog.iter().map(|op| op.id.as_str()).collect();
    for id in [
        "cli_visible",
        "cli_audio_invisible",
        "cli_video_message_remove",
        "cli_text_all",
        "cli_batch",
        "api_draft_text",
    ] {
        assert!(ids.contains(&id), "{id}");
    }
    assert!(!ids.contains(&"cli_mcp") && !ids.iter().any(|id| id.contains("integration")));
    let visible = catalog.iter().find(|op| op.id == "cli_visible").unwrap();
    let strip = visible
        .parameters
        .iter()
        .find(|p| p.name == "strip_metadata")
        .unwrap();
    let cli = strip.cli.as_ref().unwrap();
    assert_eq!(cli.flags, ["--strip-metadata"]);
    assert_eq!(cli.negative_flags, ["--keep-metadata"]);
    assert_eq!(strip.schema["default"], true);
    let output = visible.parameters.iter().find(|p| p.name == "output").unwrap();
    assert_eq!(output.cli.as_ref().unwrap().flags, ["-o", "--output"]);
    let message = catalog
        .iter()
        .find(|op| op.id == "cli_video_message_remove")
        .unwrap();
    let long_side = message.parameters.iter().find(|p| p.name == "long_side").unwrap();
    assert_eq!(long_side.schema["type"], "integer");
    assert_eq!(long_side.schema["minimum"], 8);
    assert_eq!(long_side.schema["default"], 512);
    let (planned, output) = visible
        .plan(&json!({"source": "a.png", "detect": false, "strip_metadata": false}))
        .unwrap();
    assert_eq!(output.as_deref().map(|o| o.ends_with("a_clean.png")), Some(true));
    assert!(planned.contains(&"--no-detect".to_owned()) && planned.contains(&"--keep-metadata".to_owned()));
    assert_the_cli_accepts(&planned);
}
