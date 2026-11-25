//! C2PA 2.3/2.4 Appendix A.8: the trailing `U+FEFF` + Variation Selector `C2PATXT\0` wrapper.

use crate::error::TextError;
use crate::selectors::{encode_variation_selectors, variation_selector_to_byte};
use crate::unicode;

pub const C2PATXT_MAGIC: &[u8; 8] = b"C2PATXT\x00";
pub const C2PATXT_VERSION: u8 = 1;
const HEADER_LEN: usize = 13;

/// A recognized wrapper and where it sits in the text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C2paTxtMatch {
    pub manifest_bytes: Vec<u8>,
    pub padding_bytes: usize,
    pub had_bom_prefix: bool,
    pub removed_codepoints: usize,
}

/// Detect and strip a trailing wrapper, keeping any final line endings.
pub fn strip_c2patxt_suffix(text: &str) -> (String, Option<C2paTxtMatch>) {
    let end = text.trim_end_matches(['\r', '\n']).len();
    let body = &text[..end];
    let cursor = body
        .trim_end_matches(|ch| variation_selector_to_byte(ch).is_some())
        .len();
    let raw: Vec<u8> = body[cursor..]
        .chars()
        .filter_map(variation_selector_to_byte)
        .collect();
    if raw.len() < HEADER_LEN || &raw[..8] != C2PATXT_MAGIC || raw[8] != C2PATXT_VERSION {
        return (text.to_owned(), None);
    }
    let manifest_len = u32::from_be_bytes([raw[9], raw[10], raw[11], raw[12]]) as usize;
    if HEADER_LEN + manifest_len > raw.len() {
        return (text.to_owned(), None);
    }
    let had_bom = body[..cursor].ends_with('\u{FEFF}');
    let prefix_end = if had_bom {
        cursor - '\u{FEFF}'.len_utf8()
    } else {
        cursor
    };
    let found = C2paTxtMatch {
        manifest_bytes: raw[HEADER_LEN..HEADER_LEN + manifest_len].to_vec(),
        padding_bytes: raw.len() - HEADER_LEN - manifest_len,
        had_bom_prefix: had_bom,
        removed_codepoints: raw.len() + usize::from(had_bom),
    };
    (format!("{}{}", &text[..prefix_end], &text[end..]), Some(found))
}

/// Embed `manifest_bytes` at the end of `text` per Appendix A.8.
pub fn encode_c2patxt(text: &str, manifest_bytes: &[u8], padding_bytes: usize) -> Result<String, TextError> {
    let length =
        u32::try_from(manifest_bytes.len()).map_err(|_| TextError::invalid("manifest exceeds 4 GiB"))?;
    let normalized = unicode::nfc(&strip_c2patxt_suffix(text).0);
    let mut wrapper = Vec::with_capacity(HEADER_LEN + manifest_bytes.len() + padding_bytes);
    wrapper.extend_from_slice(C2PATXT_MAGIC);
    wrapper.push(C2PATXT_VERSION);
    wrapper.extend_from_slice(&length.to_be_bytes());
    wrapper.extend_from_slice(manifest_bytes);
    wrapper.resize(wrapper.len() + padding_bytes, 0);
    Ok(format!(
        "{normalized}\u{FEFF}{}",
        encode_variation_selectors(&wrapper)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_keeps_trailing_newlines() {
        let marked = encode_c2patxt("Body\n", b"jumb", 3).unwrap();
        let (clean, found) = strip_c2patxt_suffix(&format!("{marked}\r\n"));
        assert_eq!(clean, "Body\n\r\n");
        let found = found.unwrap();
        assert_eq!(found.manifest_bytes, b"jumb");
        assert_eq!(found.padding_bytes, 3);
        assert_eq!(found.removed_codepoints, 1 + 13 + 4 + 3);
    }
}
