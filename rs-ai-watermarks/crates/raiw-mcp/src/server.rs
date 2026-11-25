//! Newline-delimited JSON-RPC 2.0 over stdio with the MCP initialize handshake.

use std::io::{self, BufRead, Write};

use raiw_core::{JsonStyle, to_py_json};
use serde_json::{Map, Value, json};

use crate::catalog::Operation;
use crate::exec::{Completed, Runner, run_operation, secret};
use crate::tools::{self, convenience_tools, validated};
use crate::{Arguments, InvalidArguments};

/// Newest first; a client asking for an unlisted version gets the newest.
pub const PROTOCOL_VERSIONS: [&str; 4] = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];

pub const INSTRUCTIONS: &str = "Identify and remove AI provenance marks from images, video, audio, and text the user generated \
or edited themselves: C2PA, IPTC, EXIF and TC260 metadata, visible AI-generator \
labels, and supported invisible watermarks. Do not use it on stock, agency, \
marketplace or other third-party marks that protect someone else's asset. A \
result of 'unknown' or exit code 2 means no supported signal was found, not \
that the file is clean. Paths are read on the machine running the server.";

/// Program name reported in each tool result's `command`.
pub const PROGRAM: &str = "rs-ai-watermarks";

/// The MCP server: convenience tools plus every catalog operation.
pub struct Server {
    convenience: Vec<Value>,
    operations: Vec<Operation>,
    runner: Box<dyn Runner>,
}

impl std::fmt::Debug for Server {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Server")
            .field("operations", &self.operations.len())
            .finish_non_exhaustive()
    }
}

impl Server {
    pub fn new(operations: Vec<Operation>, runner: Box<dyn Runner>) -> Self {
        Self {
            convenience: convenience_tools(),
            operations,
            runner,
        }
    }

    fn convenience(&self, name: &str) -> Option<&Value> {
        self.convenience.iter().find(|tool| tool["name"] == name)
    }

    fn operation(&self, name: &str) -> Option<&Operation> {
        self.operations.iter().find(|op| op.id == name)
    }

    /// Every tool, convenience tools first.
    pub fn tools(&self) -> Vec<Value> {
        let catalog = self.operations.iter().map(|op| {
            json!({
                "name": op.id,
                "title": op.title,
                "description": op.description,
                "inputSchema": op.input_schema(),
                "annotations": {"openWorldHint": false},
            })
        });
        self.convenience.iter().cloned().chain(catalog).collect()
    }

    /// Run one tool and return an MCP `tools/call` result.
    pub fn call_tool(&self, name: &str, arguments: &Arguments) -> Value {
        if let Some(operation) = self.operation(name) {
            return self.call_operation(operation, arguments);
        }
        let Some(tool) = self.convenience(name) else {
            return invalid(&InvalidArguments(format!("Unknown tool: {name}")));
        };
        let (args, output) = match validated(tool, arguments).and_then(|values| tools::plan(name, &values)) {
            Ok(plan) => plan,
            Err(err) => return invalid(&err),
        };
        match self.runner.run(&args, None) {
            Ok(completed) => tool_result(&completed, &args, output),
            Err(err) => {
                tracing::warn!(?args, %err, "CLI command could not start");
                error_content(&format!("The command could not start: {err}"))
            }
        }
    }

    fn call_operation(&self, operation: &Operation, arguments: &Arguments) -> Value {
        let value = Value::Object(arguments.clone());
        let planned = operation.plan(&value).map_err(|e| e.0).and_then(|plan| {
            run_operation(operation, &value, self.runner.as_ref())
                .map(|result| (plan, result))
                .map_err(|e| e.to_string())
        });
        let ((mut args, _), result) = match planned {
            Ok(done) => done,
            Err(message) => return error_content(&format!("Invalid arguments: {message}")),
        };
        if let Some(token) = secret(arguments) {
            for arg in &mut args {
                if arg == token {
                    *arg = "[redacted]".into();
                }
            }
        }
        let completed = Completed {
            code: result.exit_code,
            stdout: result.stdout,
            stderr: result.stderr,
        };
        let mut formatted = tool_result(&completed, &args, result.output);
        let structured = &mut formatted["structuredContent"];
        structured["operation"] = operation.id.clone().into();
        if args.is_empty() {
            let mut command = vec![PROGRAM.to_owned()];
            command.extend(crate::exec::EXECUTE_COMMAND.iter().map(|s| (*s).to_owned()));
            command.push(operation.id.clone());
            structured["command"] = json!(command);
        }
        formatted
    }

    /// Answer one JSON-RPC message; `None` for notifications and client responses.
    pub fn handle(&self, raw: &Value) -> Option<Value> {
        let Some(message) = raw
            .as_object()
            .filter(|m| m.get("jsonrpc") == Some(&json!("2.0")))
        else {
            return Some(rpc_error(&Value::Null, -32600, "Invalid Request"));
        };
        let request_id = message.get("id")?;
        let Some(method) = message.get("method").and_then(Value::as_str) else {
            return Some(rpc_error(request_id, -32600, "Invalid Request"));
        };
        let params = match message.get("params") {
            None | Some(Value::Null) => Map::new(),
            Some(Value::Object(params)) => params.clone(),
            Some(_) => return Some(rpc_error(request_id, -32602, "params must be an object")),
        };
        let result = match method {
            "initialize" => {
                let requested = params.get("protocolVersion").and_then(Value::as_str);
                let version = requested
                    .filter(|v| PROTOCOL_VERSIONS.contains(v))
                    .unwrap_or(PROTOCOL_VERSIONS[0]);
                json!({
                    "protocolVersion": version,
                    "capabilities": {"tools": {"listChanged": false}},
                    "serverInfo": {"name": PROGRAM, "title": "Remove AI Watermarks", "version": env!("CARGO_PKG_VERSION")},
                    "instructions": INSTRUCTIONS,
                })
            }
            "ping" => json!({}),
            "tools/list" => json!({"tools": self.tools()}),
            "tools/call" => {
                let name = params.get("name").and_then(Value::as_str);
                let known = name.filter(|n| self.convenience(n).is_some() || self.operation(n).is_some());
                let Some(name) = known else {
                    let shown = params
                        .get("name")
                        .map_or_else(|| "None".into(), raiw_core::pyjson::py_str);
                    return Some(rpc_error(request_id, -32602, &format!("Unknown tool: {shown}")));
                };
                let arguments = match params.get("arguments") {
                    None | Some(Value::Null) => Map::new(),
                    Some(Value::Object(arguments)) => arguments.clone(),
                    Some(_) => return Some(rpc_error(request_id, -32602, "arguments must be an object")),
                };
                self.call_tool(name, &arguments)
            }
            other => {
                return Some(rpc_error(
                    request_id,
                    -32601,
                    &format!("Method not found: {other}"),
                ));
            }
        };
        Some(json!({"jsonrpc": "2.0", "id": request_id, "result": result}))
    }

    /// Serve newline-delimited JSON-RPC until `input` closes.
    pub fn serve(&self, input: impl BufRead, mut output: impl Write) -> io::Result<()> {
        for line in input.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let response = match serde_json::from_str::<Value>(&line) {
                Ok(message) => self.handle(&message),
                Err(_) => Some(rpc_error(&Value::Null, -32700, "Parse error")),
            };
            if let Some(response) = response {
                writeln!(output, "{}", to_py_json(&response, JsonStyle::UNICODE))?;
                output.flush()?;
            }
        }
        Ok(())
    }
}

fn rpc_error(id: &Value, code: i64, message: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

fn error_content(text: &str) -> Value {
    json!({"content": [{"type": "text", "text": text}], "isError": true})
}

fn invalid(err: &InvalidArguments) -> Value {
    error_content(&format!("Invalid arguments: {err}"))
}

/// Preserve CLI exit semantics: exit 2 without a usage line is "no signal", not an error.
pub fn tool_result(completed: &Completed, args: &[String], output: Option<String>) -> Value {
    let code = completed.code;
    let text = completed.stdout.trim();
    let stderr_full = completed.stderr.trim();
    let stderr = tail_chars(stderr_full, 4000);
    let mut command = vec![PROGRAM.to_owned()];
    command.extend(args.iter().cloned());
    let mut structured = Map::new();
    structured.insert("command".into(), json!(command));
    structured.insert("exit_code".into(), code.into());
    structured.insert("output".into(), output.map_or(Value::Null, Value::String));
    if (args.iter().any(|a| a == "--json") || args.is_empty())
        && !text.is_empty()
        && let Ok(report) = serde_json::from_str::<Value>(text)
    {
        if let Some(reported) = report.as_object().and_then(|r| r.get("output")) {
            structured.insert("output".into(), reported.clone());
        }
        structured.insert("report".into(), report);
    }
    if !structured.contains_key("report") {
        structured.insert("stdout".into(), text.into());
    }
    if !stderr.is_empty() {
        structured.insert("stderr".into(), stderr.into());
    }
    let no_signal = code == 2 && !stderr.contains("Usage:");
    let is_error = code != 0 && !no_signal;
    let mut summary = format!("exit code {code}");
    if no_signal {
        summary.push_str(
            " (no supported signal found, or nothing to process; this is unknown, not proof of a clean file)",
        );
    }
    let body: Vec<&str> = [summary.as_str(), text, if is_error { stderr } else { "" }]
        .into_iter()
        .filter(|p| !p.is_empty())
        .collect();
    json!({"content": [{"type": "text", "text": body.join("\n\n")}], "structuredContent": structured, "isError": is_error})
}

fn tail_chars(text: &str, count: usize) -> &str {
    let total = text.chars().count();
    if total <= count {
        return text;
    }
    let start = text.char_indices().nth(total - count).map_or(0, |(i, _)| i);
    &text[start..]
}
