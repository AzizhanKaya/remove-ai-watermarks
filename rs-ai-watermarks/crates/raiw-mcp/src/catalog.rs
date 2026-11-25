//! Operation catalog derived from the clap tree, with validation and CLI planning.

use std::any::TypeId;
use std::path::{Path, PathBuf};

use clap::{Arg, ArgAction, Command};
use raiw_core::fs::{resolve_lenient, with_name_suffix, with_stem_replaced, with_stem_suffix};
use raiw_core::pyjson::{py_float_repr, py_repr, py_str};
use serde::Serialize;
use serde_json::{Map, Number, Value, json};

use crate::spec::{BoundSpec, ToggleSpec};
use crate::{Arguments, InvalidArguments};

/// One CLI leaf (or API-only backend) exposed to integrations.
#[derive(Debug, Clone, Serialize)]
pub struct Operation {
    pub id: String,
    pub title: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<Vec<String>>,
    pub parameters: Vec<Parameter>,
}

/// One operation parameter with its JSON Schema and, for CLI leaves, its spelling.
#[derive(Debug, Clone, Serialize)]
pub struct Parameter {
    pub name: String,
    pub schema: Map<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<bool>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub cli: Option<CliBinding>,
}

/// How a parameter is spelled on the command line.
#[derive(Debug, Clone, Serialize)]
pub struct CliBinding {
    pub positional: bool,
    pub flags: Vec<String>,
    pub negative_flags: Vec<String>,
    pub is_flag: bool,
    pub secret: bool,
}

impl Parameter {
    fn is_required(&self) -> bool {
        self.required == Some(true)
    }

    fn kind(&self) -> &str {
        self.schema
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("string")
    }
}

// Catalog construction

/// Describe every non-server CLI leaf of `root` plus the API-only draft-text backend.
pub fn operation_catalog(root: &Command, toggles: &[ToggleSpec], bounds: &[BoundSpec]) -> Vec<Operation> {
    let mut operations = Vec::new();
    walk(root, &mut Vec::new(), toggles, bounds, &mut operations);
    operations.push(draft_text_operation());
    operations
}

fn walk(
    command: &Command,
    path: &mut Vec<String>,
    toggles: &[ToggleSpec],
    bounds: &[BoundSpec],
    out: &mut Vec<Operation>,
) {
    let children: Vec<_> = command
        .get_subcommands()
        .filter(|c| c.get_name() != "help" && !c.is_hide_set())
        .collect();
    if !children.is_empty() {
        for child in children {
            path.push(child.get_name().to_owned());
            walk(child, path, toggles, bounds, out);
            path.pop();
        }
        return;
    }
    if path.as_slice() == ["mcp"] || path.is_empty() {
        return;
    }
    let skipped: Vec<&str> = toggles
        .iter()
        .map(|t| t.off_id)
        .chain(["help", "version", "verbose"])
        .collect();
    let parameters = command
        .get_arguments()
        .filter(|arg| !skipped.contains(&arg.get_id().as_str()))
        .map(|arg| describe(arg, path, toggles, bounds))
        .collect();
    let about = command
        .get_about()
        .map_or_else(|| path.join(" "), ToString::to_string);
    out.push(Operation {
        id: format!("cli_{}", path.join("_").replace('-', "_")),
        title: path.join(" "),
        description: about,
        command: Some(path.clone()),
        parameters,
    });
}

fn describe(arg: &Arg, path: &[String], toggles: &[ToggleSpec], bounds: &[BoundSpec]) -> Parameter {
    let name = arg.get_id().as_str().to_owned();
    let toggle = toggles.iter().find(|t| t.id == name);
    let is_flag = matches!(arg.get_action(), ArgAction::SetTrue);
    let multiple = matches!(arg.get_action(), ArgAction::Append);

    let mut schema = Map::new();
    let kind = value_kind(arg, is_flag);
    schema.insert("type".into(), kind.into());
    let choices: Vec<Value> = arg
        .get_possible_values()
        .iter()
        .filter(|v| !v.is_hide_set())
        .map(|v| v.get_name().into())
        .collect();
    if !is_flag && !choices.is_empty() {
        schema.insert("enum".into(), Value::Array(choices));
    }
    if let Some(bound) = bounds.iter().find(|b| b.applies(&name, path)) {
        if let Some(minimum) = bound.minimum() {
            schema.insert("minimum".into(), minimum);
        }
        if let Some(maximum) = bound.maximum() {
            schema.insert("maximum".into(), maximum);
        }
    }
    if multiple {
        schema = Map::from_iter([
            ("type".to_owned(), "array".into()),
            ("items".to_owned(), Value::Object(schema)),
        ]);
    }
    let default = match toggle {
        Some(toggle) => toggle.default.map(Value::Bool),
        None if is_flag || multiple => None,
        None => arg
            .get_default_values()
            .first()
            .map(|raw| typed_default(&raw.to_string_lossy(), kind)),
    };
    if let Some(default) = default {
        schema.insert("default".into(), default);
    }
    let help = arg
        .get_help()
        .map_or_else(|| name.replace('_', " "), ToString::to_string);
    schema.insert("description".into(), help.into());

    let flags = if arg.is_positional() {
        Vec::new()
    } else {
        arg.get_short()
            .map(|s| format!("-{s}"))
            .into_iter()
            .chain(arg.get_long().map(|l| format!("--{l}")))
            .collect()
    };
    Parameter {
        schema,
        required: Some(arg.is_required_set()),
        cli: Some(CliBinding {
            positional: arg.is_positional(),
            flags,
            negative_flags: toggle
                .map(|t| vec![t.negative_flag.to_owned()])
                .unwrap_or_default(),
            is_flag,
            secret: name == "hf_token",
        }),
        name,
    }
}

fn value_kind(arg: &Arg, is_flag: bool) -> &'static str {
    if is_flag {
        return "boolean";
    }
    let id = arg.get_value_parser().type_id();
    let integers = [
        TypeId::of::<i64>(),
        TypeId::of::<u64>(),
        TypeId::of::<i32>(),
        TypeId::of::<u32>(),
        TypeId::of::<usize>(),
    ];
    if integers.iter().any(|t| id == *t) {
        "integer"
    } else if id == TypeId::of::<f64>() || id == TypeId::of::<f32>() {
        "number"
    } else {
        "string"
    }
}

fn typed_default(raw: &str, kind: &str) -> Value {
    match kind {
        "integer" => raw.parse::<i64>().map_or_else(|_| raw.into(), Value::from),
        "number" => raw
            .parse::<f64>()
            .ok()
            .and_then(Number::from_f64)
            .map_or_else(|| raw.into(), Value::Number),
        _ => raw.into(),
    }
}

fn draft_text_operation() -> Operation {
    let param = |name: &str, schema: Value, required: Option<bool>| Parameter {
        name: name.into(),
        schema: schema.as_object().cloned().unwrap_or_default(),
        required,
        cli: None,
    };
    Operation {
        id: "api_draft_text".into(),
        title: "draft text for verification".into(),
        description: "Propose OCR manifest lines. Every accepted proposal still requires human verification."
            .into(),
        command: None,
        parameters: vec![
            param(
                "source",
                json!({"type": "string", "description": "Source image path"}),
                Some(true),
            ),
            param(
                "min_score",
                json!({"type": "number", "minimum": 0, "maximum": 1, "default": 0.85}),
                None,
            ),
            param("stable", json!({"type": "boolean", "default": true}), None),
        ],
    }
}

// Validation and planning

impl Operation {
    /// JSON Schema shared by the protocol and visual controls.
    pub fn input_schema(&self) -> Value {
        let properties: Map<String, Value> = self
            .parameters
            .iter()
            .map(|p| (p.name.clone(), Value::Object(p.schema.clone())))
            .collect();
        let required: Vec<Value> = self
            .parameters
            .iter()
            .filter(|p| p.is_required())
            .map(|p| p.name.clone().into())
            .collect();
        json!({"type": "object", "properties": properties, "required": required, "additionalProperties": false})
    }

    fn parameter(&self, name: &str) -> Option<&Parameter> {
        self.parameters.iter().find(|p| p.name == name)
    }

    /// Validate values before execution, retaining omitted CLI defaults.
    pub fn validate(&self, arguments: &Value) -> Result<Arguments, InvalidArguments> {
        let Some(arguments) = arguments.as_object() else {
            return Err(InvalidArguments::new("Arguments must be an object"));
        };
        let mut unknown: Vec<&str> = arguments
            .keys()
            .filter(|k| self.parameter(k).is_none())
            .map(String::as_str)
            .collect();
        if !unknown.is_empty() {
            unknown.sort_unstable();
            return Err(InvalidArguments(format!(
                "Unknown parameters: {}",
                unknown.join(", ")
            )));
        }
        let mut missing: Vec<&str> = self
            .parameters
            .iter()
            .filter(|p| p.is_required() && !arguments.contains_key(&p.name))
            .map(|p| p.name.as_str())
            .collect();
        if !missing.is_empty() {
            missing.sort_unstable();
            return Err(InvalidArguments(format!(
                "Missing parameters: {}",
                missing.join(", ")
            )));
        }
        for (name, value) in arguments {
            let spec = &self.parameter(name).expect("checked above").schema;
            check_value(name, value, spec)?;
        }
        Ok(arguments.clone())
    }

    /// Translate optional visual controls into the same typed operation payload.
    pub fn widget_arguments(&self, raw: &Arguments) -> Result<Arguments, InvalidArguments> {
        let mut values = Map::new();
        for parameter in &self.parameters {
            let Some(mut value) = raw.get(&parameter.name).cloned() else {
                continue;
            };
            if value.is_null() || value == "" {
                continue;
            }
            if let Value::String(text) = &value {
                value = match parameter.kind() {
                    "array" => text
                        .lines()
                        .map(str::trim)
                        .filter(|l| !l.is_empty())
                        .map(Value::from)
                        .collect(),
                    "boolean" => match text.as_str() {
                        "default" => continue,
                        "enabled" => Value::Bool(true),
                        "disabled" => Value::Bool(false),
                        _ => {
                            return Err(InvalidArguments(format!(
                                "Invalid boolean control: {}",
                                parameter.name
                            )));
                        }
                    },
                    "integer" => text.trim().parse::<i64>().map(Value::from).map_err(|_| {
                        InvalidArguments(format!("invalid literal for int(): {}", py_repr(&value)))
                    })?,
                    "number" => text
                        .trim()
                        .parse::<f64>()
                        .ok()
                        .and_then(Number::from_f64)
                        .map(Value::Number)
                        .ok_or_else(|| {
                            InvalidArguments(format!(
                                "could not convert string to float: {}",
                                py_repr(&value)
                            ))
                        })?,
                    _ => value,
                };
            }
            if !parameter.is_required() && parameter.schema.get("default") == Some(&value) {
                continue;
            }
            values.insert(parameter.name.clone(), value);
        }
        self.validate(&Value::Object(values))
    }

    /// Build the CLI invocation (no shell), always preserving the source, plus the output it writes.
    pub fn plan(&self, arguments: &Value) -> Result<(Vec<String>, Option<String>), InvalidArguments> {
        let mut values = self.validate(arguments)?;
        let Some(command) = &self.command else {
            return Ok((Vec::new(), None));
        };
        let output = self.plan_output(command, &mut values)?;
        let mut flags = Vec::new();
        let mut positional = Vec::new();
        for parameter in &self.parameters {
            let (Some(value), Some(cli)) = (values.get(&parameter.name), &parameter.cli) else {
                continue;
            };
            if cli.positional {
                positional.push(py_str(value));
            } else if cli.is_flag {
                if value.as_bool() == Some(true) {
                    flags.push(cli.flags[0].clone());
                } else if let Some(negative) = cli.negative_flags.first() {
                    flags.push(negative.clone());
                } else if parameter.schema.get("default") == Some(&Value::Bool(true)) {
                    return Err(InvalidArguments(format!(
                        "{} has no false option in the CLI",
                        parameter.name
                    )));
                }
            } else {
                let items = match value {
                    Value::Array(items) if parameter.kind() == "array" => items.clone(),
                    other => vec![other.clone()],
                };
                for item in items {
                    flags.push(cli.flags[0].clone());
                    flags.push(py_str(&item));
                }
            }
        }
        let mut args = command.clone();
        args.extend(flags);
        args.push("--".into());
        args.extend(positional);
        Ok((args, output))
    }

    fn plan_output(
        &self,
        command: &[String],
        values: &mut Arguments,
    ) -> Result<Option<String>, InvalidArguments> {
        let mut output_parameter = self
            .parameters
            .iter()
            .find(|p| p.name == "output" || p.name == "output_dir")
            .map(|p| p.name.clone());
        let is_true = |values: &Arguments, key: &str| values.get(key).and_then(Value::as_bool) == Some(true);
        if command.last().is_some_and(|c| c == "metadata") && !is_true(values, "remove") {
            output_parameter = None;
        }
        if command == ["visible"] && is_true(values, "detect") {
            output_parameter = None;
        }
        let Some(key) = output_parameter else {
            return Ok(None);
        };
        let source_text = values
            .get("source")
            .or_else(|| values.get("directory"))
            .map(py_str)
            .unwrap_or_default();
        let source = resolve_lenient(Path::new(&source_text));
        let media_suffix = match command.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
            ["audio", "invisible"] => Some("_processed.wav"),
            ["video", "audio"] => Some("_audio_processed.mkv"),
            ["video", "message", "remove"] => Some("_message_processed.mp4"),
            _ => None,
        };
        let default: PathBuf = match media_suffix {
            Some(suffix) => with_stem_replaced(&source, suffix),
            None if key == "output_dir" => with_name_suffix(&source, "_clean"),
            None => with_stem_suffix(&source, "_clean"),
        };
        let output = values
            .get(&key)
            .map(py_str)
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| default.to_string_lossy().into_owned());
        if resolve_lenient(Path::new(&output)) == source {
            return Err(InvalidArguments::new(
                "Output must differ from the source; integrations never overwrite an input",
            ));
        }
        values.insert(key, output.clone().into());
        Ok(Some(output))
    }
}

fn check_value(name: &str, value: &Value, spec: &Map<String, Value>) -> Result<(), InvalidArguments> {
    let kind = spec.get("type").and_then(Value::as_str).unwrap_or("string");
    let valid = match kind {
        "string" => value
            .as_str()
            .is_some_and(|s| !s.trim().is_empty() && !s.contains('\0')),
        "boolean" => value.is_boolean(),
        "integer" => value.is_i64() || value.is_u64(),
        "number" => value.is_number(),
        "array" => value.is_array(),
        _ => false,
    };
    if !valid {
        return Err(InvalidArguments(format!("{name} must be a {kind}")));
    }
    if kind == "number" && !value.as_f64().is_some_and(f64::is_finite) {
        return Err(InvalidArguments(format!("{name} must be finite")));
    }
    if let Some(Value::Array(choices)) = spec.get("enum")
        && !choices.contains(value)
    {
        return Err(InvalidArguments(format!(
            "{name} must be one of {}",
            py_repr(&Value::Array(choices.clone()))
        )));
    }
    let number = value.as_f64();
    if let (Some(minimum), Some(number)) = (spec.get("minimum"), number)
        && number < minimum.as_f64().unwrap_or(f64::NEG_INFINITY)
    {
        return Err(InvalidArguments(format!(
            "{name} must be at least {}",
            bound_text(minimum)
        )));
    }
    if let (Some(maximum), Some(number)) = (spec.get("maximum"), number)
        && number > maximum.as_f64().unwrap_or(f64::INFINITY)
    {
        return Err(InvalidArguments(format!(
            "{name} must be at most {}",
            bound_text(maximum)
        )));
    }
    if let (Value::Array(items), Some(Value::Object(item_spec))) = (value, spec.get("items")) {
        for item in items {
            check_value(name, item, item_spec)?;
        }
    }
    Ok(())
}

fn bound_text(bound: &Value) -> String {
    match bound.as_number() {
        Some(n) if n.is_f64() => n.as_f64().map_or_else(|| n.to_string(), py_float_repr),
        Some(n) => n.to_string(),
        None => py_str(bound),
    }
}
