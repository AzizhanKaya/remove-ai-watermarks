//! Values of Pillow's `Image.info` mapping and their Python `str` / `repr`.

use indexmap::IndexMap;

use crate::py::{bytes_repr, float_str, str_repr};

/// One `info` value, typed as Pillow stores it.
#[derive(Debug, Clone, PartialEq)]
pub enum InfoValue {
    None,
    Str(String),
    /// A PNG `iTXt` value: a `str` subclass that remembers its language and keyword.
    IText {
        text: String,
        lang: String,
        tkey: String,
    },
    Bytes(Vec<u8>),
    Int(i64),
    Float(f64),
    Bool(bool),
    Tuple(Vec<InfoValue>),
    Dict(Vec<(InfoValue, InfoValue)>),
}

impl InfoValue {
    /// The text of a `str` (or `iTXt`) value.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(text) | Self::IText { text, .. } => Some(text),
            _ => None,
        }
    }

    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Bytes(bytes) => Some(bytes),
            _ => None,
        }
    }

    /// `isinstance(value, (dict, list, tuple))`.
    pub const fn is_container(&self) -> bool {
        matches!(self, Self::Tuple(_) | Self::Dict(_))
    }

    /// Python `str(value)`.
    pub fn py_str(&self) -> String {
        match self {
            Self::Str(text) | Self::IText { text, .. } => text.clone(),
            other => other.py_repr(),
        }
    }

    /// Python `repr(value)`.
    pub fn py_repr(&self) -> String {
        match self {
            Self::None => "None".into(),
            Self::Str(text) | Self::IText { text, .. } => str_repr(text),
            Self::Bytes(bytes) => bytes_repr(bytes),
            Self::Int(value) => value.to_string(),
            Self::Float(value) => float_str(*value),
            Self::Bool(true) => "True".into(),
            Self::Bool(false) => "False".into(),
            Self::Tuple(items) => {
                let inner: Vec<String> = items.iter().map(Self::py_repr).collect();
                if inner.len() == 1 {
                    format!("({},)", inner[0])
                } else {
                    format!("({})", inner.join(", "))
                }
            }
            Self::Dict(items) => {
                let inner: Vec<String> = items
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k.py_repr(), v.py_repr()))
                    .collect();
                format!("{{{}}}", inner.join(", "))
            }
        }
    }
}

/// Pillow's ordered `info` dict.
pub type Info = IndexMap<String, InfoValue>;

pub fn int_pair(a: i64, b: i64) -> InfoValue {
    InfoValue::Tuple(vec![InfoValue::Int(a), InfoValue::Int(b)])
}

pub fn float_pair(a: f64, b: f64) -> InfoValue {
    InfoValue::Tuple(vec![InfoValue::Float(a), InfoValue::Float(b)])
}
