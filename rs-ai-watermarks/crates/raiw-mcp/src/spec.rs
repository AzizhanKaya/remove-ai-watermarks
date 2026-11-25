//! Facts clap cannot introspect: Click-style toggle pairs and numeric ranges.

use serde_json::{Number, Value};

/// One `--on/--off` boolean pair declared by the CLI.
#[derive(Debug, Clone, Copy)]
pub struct ToggleSpec {
    /// Python parameter name, also the clap id of the positive spelling.
    pub id: &'static str,
    /// Clap id of the hidden negative spelling.
    pub off_id: &'static str,
    pub negative_flag: &'static str,
    /// `None` when the CLI deliberately leaves the choice to the library.
    pub default: Option<bool>,
}

#[derive(Debug, Clone, Copy)]
enum Bound {
    Int(i64),
    Float(f64),
}

impl Bound {
    fn to_json(self) -> Value {
        match self {
            Self::Int(value) => Value::from(value),
            Self::Float(value) => Number::from_f64(value).map_or(Value::Null, Value::Number),
        }
    }
}

/// An `IntRange`/`FloatRange` bound on one parameter, optionally scoped to one command.
#[derive(Debug, Clone, Copy)]
pub struct BoundSpec {
    pub id: &'static str,
    minimum: Option<Bound>,
    maximum: Option<Bound>,
    command: Option<&'static [&'static str]>,
}

impl BoundSpec {
    pub const fn int(id: &'static str, minimum: Option<i64>, maximum: Option<i64>) -> Self {
        Self {
            id,
            minimum: match minimum {
                Some(value) => Some(Bound::Int(value)),
                None => None,
            },
            maximum: match maximum {
                Some(value) => Some(Bound::Int(value)),
                None => None,
            },
            command: None,
        }
    }

    pub const fn float(id: &'static str, minimum: Option<f64>, maximum: Option<f64>) -> Self {
        Self {
            id,
            minimum: match minimum {
                Some(value) => Some(Bound::Float(value)),
                None => None,
            },
            maximum: match maximum {
                Some(value) => Some(Bound::Float(value)),
                None => None,
            },
            command: None,
        }
    }

    /// Restrict the bound to one command path.
    #[must_use]
    pub const fn only_in(mut self, command: &'static [&'static str]) -> Self {
        self.command = Some(command);
        self
    }

    pub fn applies(&self, id: &str, path: &[String]) -> bool {
        self.id == id && self.command.is_none_or(|command| command.iter().eq(path.iter()))
    }

    pub fn minimum(&self) -> Option<Value> {
        self.minimum.map(Bound::to_json)
    }

    pub fn maximum(&self) -> Option<Value> {
        self.maximum.map(Bound::to_json)
    }
}
