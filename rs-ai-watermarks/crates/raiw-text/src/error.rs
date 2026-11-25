//! Error type mirroring the Python exception classes the text pipeline raises.

use std::path::PathBuf;

/// Errors from text inspection, cleaning and regeneration.
///
/// `Runtime` corresponds to Python's `RuntimeError`: a refused or failed rewrite that
/// `allow_partial` may retain. Every other variant always propagates.
#[derive(Debug, thiserror::Error)]
pub enum TextError {
    /// A rewrite, transport or acceptance failure (`RuntimeError`).
    #[error("{0}")]
    Runtime(String),
    /// An invalid argument or configuration (`ValueError`).
    #[error("{0}")]
    InvalidArgument(String),
    /// A strict UTF-8 decoding failure (`UnicodeDecodeError`).
    #[error("{0}")]
    Decode(String),
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

impl TextError {
    pub fn runtime(message: impl Into<String>) -> Self {
        Self::Runtime(message.into())
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        Self::InvalidArgument(message.into())
    }

    pub const fn is_runtime(&self) -> bool {
        matches!(self, Self::Runtime(_))
    }

    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}

/// Decode UTF-8 strictly, describing failures as CPython does.
pub fn decode_utf8(bytes: Vec<u8>) -> Result<String, TextError> {
    String::from_utf8(bytes).map_err(|err| {
        let bytes = err.as_bytes();
        let error = err.utf8_error();
        TextError::decode_message(bytes, error)
    })
}

impl TextError {
    fn decode_message(bytes: &[u8], error: std::str::Utf8Error) -> Self {
        let start = error.valid_up_to();
        let lead = bytes.get(start).copied().unwrap_or(0);
        let message = match error.error_len() {
            None if bytes.len() - start > 1 => format!(
                "'utf-8' codec can't decode bytes in position {start}-{}: unexpected end of data",
                bytes.len() - 1
            ),
            None => format!(
                "'utf-8' codec can't decode byte 0x{lead:02x} in position {start}: unexpected end of data"
            ),
            Some(_) if matches!(lead, 0x80..=0xC1 | 0xF5..=0xFF) => {
                format!(
                    "'utf-8' codec can't decode byte 0x{lead:02x} in position {start}: invalid start byte"
                )
            }
            Some(1) => {
                format!(
                    "'utf-8' codec can't decode byte 0x{lead:02x} in position {start}: invalid continuation byte"
                )
            }
            Some(length) => format!(
                "'utf-8' codec can't decode bytes in position {start}-{}: invalid continuation byte",
                start + length - 1
            ),
        };
        Self::Decode(message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_messages_match_cpython() {
        let message = |bytes: &[u8]| decode_utf8(bytes.to_vec()).unwrap_err().to_string();
        assert_eq!(
            message(b"\xff"),
            "'utf-8' codec can't decode byte 0xff in position 0: invalid start byte"
        );
        assert_eq!(
            message(b"ab\xe2(\xa1"),
            "'utf-8' codec can't decode byte 0xe2 in position 2: invalid continuation byte"
        );
        assert_eq!(
            message(b"ab\xe2\x82"),
            "'utf-8' codec can't decode bytes in position 2-3: unexpected end of data"
        );
        assert_eq!(
            message(b"\xc3"),
            "'utf-8' codec can't decode byte 0xc3 in position 0: unexpected end of data"
        );
        assert_eq!(
            message(b"\xed\xa0\x80"),
            "'utf-8' codec can't decode byte 0xed in position 0: invalid continuation byte"
        );
        assert_eq!(
            message(b"\xf0\x9f\x98A"),
            "'utf-8' codec can't decode bytes in position 0-2: invalid continuation byte"
        );
    }
}
