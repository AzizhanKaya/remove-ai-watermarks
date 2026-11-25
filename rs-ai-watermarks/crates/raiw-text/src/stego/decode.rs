//! Covert payload decoders (Innamark, StegCloak, zero-width binary, VS encoders) and test encoders.

use std::collections::BTreeSet;
use std::io::Write as _;

use flate2::{Decompress, FlushDecompress, Status};
use indexmap::IndexMap;
use sha3::{Digest, Sha3_256};

use crate::error::TextError;
use crate::provenance::c2patxt::strip_c2patxt_suffix;
use crate::provenance::vsrmark::detect_vsrmark;
use crate::selectors::variation_selector_to_byte;
use crate::unicode;

const INNAMARK_SEPARATOR: char = '\u{2004}';
const INNAMARK_ALPHABET: [char; 4] = ['\u{2008}', '\u{2009}', '\u{202F}', '\u{205F}'];

fn innamark_digit(ch: char) -> Option<u8> {
    INNAMARK_ALPHABET
        .iter()
        .position(|&digit| digit == ch)
        .map(|index| index as u8)
}

/// Inflate a complete zlib (`header`) or raw deflate stream; trailing bytes are ignored.
pub(crate) fn inflate(data: &[u8], header: bool) -> Option<Vec<u8>> {
    let mut decoder = Decompress::new(header);
    let mut out = Vec::with_capacity(data.len() * 4 + 64);
    loop {
        let consumed = decoder.total_in() as usize;
        if out.len() == out.capacity() {
            out.reserve(out.capacity().max(64));
        }
        let before_out = decoder.total_out();
        let status = decoder
            .decompress_vec(&data[consumed..], &mut out, FlushDecompress::Finish)
            .ok()?;
        match status {
            Status::StreamEnd => return Some(out),
            Status::Ok | Status::BufError => {
                let stalled = decoder.total_in() as usize == consumed && decoder.total_out() == before_out;
                if stalled && out.len() < out.capacity() {
                    return None;
                }
            }
        }
    }
}

/// `_decode_printable_utf8`: strict UTF-8 whose characters are mostly printable.
pub(crate) fn decode_printable_utf8(raw: &[u8], min_printable_ratio: f64) -> Option<String> {
    if raw.is_empty() {
        return None;
    }
    let decoded = std::str::from_utf8(raw).ok()?;
    let total = decoded.chars().count();
    let printable = decoded
        .chars()
        .filter(|&ch| unicode::is_printable(ch) || matches!(ch, '\n' | '\r' | '\t'))
        .count();
    (printable as f64 / total as f64 >= min_printable_ratio).then(|| decoded.to_owned())
}

fn parse_innamark_tag(raw: &[u8]) -> Option<String> {
    if raw.len() < 2 {
        return None;
    }
    let type_tag = raw[0];
    if type_tag & 0x80 != 0 || type_tag & 0x07 != 0 {
        return None;
    }
    let compressed = type_tag & 0x40 != 0;
    let sized = type_tag & 0x20 != 0;
    let hash_bits = type_tag & 0x18;
    if hash_bits == 0x18 {
        return None;
    }
    let mut offset = 1;
    let mut tag_end = raw.len();
    if sized {
        let field = raw.get(offset..offset + 4)?;
        let sized_len = u32::from_le_bytes([field[0], field[1], field[2], field[3]]) as usize;
        if sized_len < offset + 4 || sized_len > raw.len() {
            return None;
        }
        tag_end = sized_len;
        offset += 4;
    }
    let tag = &raw[..tag_end];
    if hash_bits == 0x10 {
        let field = tag.get(offset..offset + 4)?;
        let expected = u32::from_le_bytes([field[0], field[1], field[2], field[3]]);
        let mut zeroed = tag.to_vec();
        zeroed[offset..offset + 4].fill(0);
        if crc32fast::hash(&zeroed) != expected {
            return None;
        }
        offset += 4;
    } else if hash_bits == 0x08 {
        let expected = tag.get(offset..offset + 32)?;
        let mut masked = tag.to_vec();
        masked[offset..offset + 32].fill(0xFF);
        if Sha3_256::digest(&masked).as_slice() != expected {
            return None;
        }
        offset += 32;
    }
    let body = &tag[offset..];
    let body = if compressed {
        inflate(body, false)?
    } else {
        body.to_vec()
    };
    decode_printable_utf8(&body, 0.75)
}

/// Detect and decode an `Innamark` (`arXiv:2502.12710`) whitespace watermark.
pub fn detect_innamark_payload(text: &str) -> Option<String> {
    let stream: Vec<char> = text
        .chars()
        .filter(|&ch| ch == INNAMARK_SEPARATOR || innamark_digit(ch).is_some())
        .collect();
    if stream.len() < 8 {
        return None;
    }
    let segments: Vec<&[char]> = stream
        .split(|&ch| ch == INNAMARK_SEPARATOR)
        .filter(|segment| segment.len() >= 4)
        .collect();
    let mut counts: IndexMap<String, usize> = IndexMap::new();
    for segment in &segments {
        let raw: Vec<u8> = segment
            .chunks_exact(4)
            .map(|quad| {
                quad.iter().enumerate().fold(0u8, |byte, (shift, &ch)| {
                    byte | (innamark_digit(ch).unwrap_or(0) << (2 * shift))
                })
            })
            .collect();
        let decoded = match parse_innamark_tag(&raw) {
            Some(tag) => Some(tag),
            None if segments.len() >= 2 => decode_printable_utf8(&raw, 0.8),
            None => None,
        };
        if let Some(decoded) = decoded {
            *counts.entry(decoded).or_insert(0) += 1;
        }
    }
    let best = counts.values().copied().max()?;
    counts
        .into_iter()
        .find(|&(_, count)| count == best)
        .map(|(payload, _)| payload)
}

const STEGCLOAK_ZWC: [char; 6] = [
    '\u{200C}', '\u{200D}', '\u{2061}', '\u{2062}', '\u{2063}', '\u{2064}',
];
const STEGCLOAK_PAIRS: [(char, char); 6] = [
    ('\u{200C}', '\u{200D}'),
    ('\u{200C}', '\u{2061}'),
    ('\u{200C}', '\u{2062}'),
    ('\u{200D}', '\u{2061}'),
    ('\u{200D}', '\u{2062}'),
    ('\u{2061}', '\u{2062}'),
];

fn stegcloak_dibits(ch: char) -> Option<u8> {
    match ch {
        '\u{200C}' => Some(0),
        '\u{200D}' => Some(1),
        '\u{2061}' => Some(2),
        '\u{2062}' => Some(3),
        _ => None,
    }
}

/// Maximal runs of characters satisfying `member` with at least `min_len` characters.
fn runs(chars: &[char], min_len: usize, member: impl Fn(char) -> bool) -> Vec<&[char]> {
    chars
        .split(|&ch| !member(ch))
        .filter(|run| run.len() >= min_len)
        .collect()
}

/// Detect a `KuroLabs/stegcloak` zero-width stream and describe its framing or payload.
pub fn detect_stegcloak_stream(chars: &[char]) -> Option<String> {
    for run in runs(chars, 6, |ch| STEGCLOAK_ZWC.contains(&ch)) {
        if !run.iter().any(|ch| matches!(ch, '\u{2061}'..='\u{2064}')) {
            continue;
        }
        let flag = STEGCLOAK_ZWC.iter().position(|&zwc| zwc == run[0]).unwrap_or(0);
        let (zwc_a, zwc_b) = STEGCLOAK_PAIRS[flag];
        let mut expanded = Vec::with_capacity(run.len() * 2);
        for &ch in &run[1..] {
            match ch {
                '\u{2064}' => expanded.extend([zwc_b, zwc_b]),
                '\u{2063}' => expanded.extend([zwc_a, zwc_a]),
                _ => expanded.push(ch),
            }
        }
        let mode = expanded[0];
        if !matches!(mode, '\u{200C}' | '\u{200D}' | '\u{2061}') {
            continue;
        }
        let body = &expanded[1..];
        if body.len() % 4 != 0 {
            continue;
        }
        let Some(dibits) = body
            .iter()
            .map(|&ch| stegcloak_dibits(ch))
            .collect::<Option<Vec<u8>>>()
        else {
            continue;
        };
        let raw: Vec<u8> = dibits
            .chunks_exact(4)
            .map(|quad| quad.iter().fold(0u8, |byte, &dibit| (byte << 2) | dibit) ^ 0xFF)
            .collect();
        if mode == '\u{200C}' && raw.len() > 40 {
            return Some(format!(
                "aes-256-ctr+hmac-sha256 ({} bytes ciphertext)",
                raw.len() - 40
            ));
        }
        if mode == '\u{200D}' && raw.len() > 8 {
            return Some(format!("aes-256-ctr ({} bytes ciphertext)", raw.len() - 8));
        }
        if mode == '\u{2061}' {
            return Some(match decode_printable_utf8(&raw, 0.7) {
                Some(plain) => format!("unencrypted:{plain}"),
                None => format!("unencrypted ({} bytes lzutf8)", raw.len()),
            });
        }
    }
    None
}

fn bits_to_byte(bits: &[char], zero: char) -> u8 {
    bits.iter()
        .fold(0u8, |byte, &ch| (byte << 1) | u8::from(ch != zero))
}

/// Decode `GroundTruth-MCP`, `zengkan0703`, `BitJacker` and `phantomoji` zero-width payloads.
pub fn detect_zero_width_binary_payload(chars: &[char]) -> IndexMap<&'static str, String> {
    let mut found: IndexMap<&'static str, String> = IndexMap::new();
    if let Some(run) = runs(chars, 64, |ch| matches!(ch, '\u{2061}' | '\u{2062}')).first() {
        let bits = &run[..64];
        let value = |slice: &[char]| {
            slice
                .iter()
                .fold(0u32, |acc, &ch| (acc << 1) | u32::from(ch != '\u{2061}'))
        };
        found.insert(
            "groundtruth_mcp",
            format!(
                "install_id=0x{:08x}, nonce=0x{:08x}",
                value(&bits[..32]),
                value(&bits[32..])
            ),
        );
    }
    for run in runs(chars, 8, |ch| {
        matches!(ch, '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{2060}' | '\u{FEFF}')
    }) {
        let distinct: BTreeSet<char> = run.iter().copied().collect();
        let within = |allowed: &[char]| distinct.iter().all(|ch| allowed.contains(ch));
        if distinct.contains(&'\u{200D}') && within(&['\u{200B}', '\u{200C}', '\u{200D}']) {
            let parts: Vec<&[char]> = run
                .split(|&ch| ch == '\u{200D}')
                .filter(|part| !part.is_empty())
                .collect();
            if !parts.is_empty() && parts.iter().all(|part| part.len() <= 16) {
                for zero in ['\u{200B}', '\u{200C}'] {
                    let codes: Vec<u32> = parts
                        .iter()
                        .map(|part| {
                            part.iter()
                                .fold(0u32, |acc, &ch| (acc << 1) | u32::from(ch != zero))
                        })
                        .collect();
                    let printable = codes
                        .iter()
                        .filter(|&&code| char::from_u32(code).is_some_and(unicode::is_printable))
                        .count();
                    if printable as f64 / codes.len() as f64 >= 0.8 {
                        let candidate: String = codes
                            .iter()
                            .map(|&code| char::from_u32(code).unwrap_or('\u{FFFD}'))
                            .collect();
                        found.entry("zero_width_binary").or_insert(candidate);
                        break;
                    }
                }
            }
        }
        if distinct.len() == 2 {
            let usable = run.len() / 8 * 8;
            for &zero in &distinct {
                let raw: Vec<u8> = run[..usable]
                    .chunks_exact(8)
                    .map(|bits| bits_to_byte(bits, zero))
                    .collect();
                if let Some(decoded) = decode_printable_utf8(&raw, 0.8) {
                    found.entry("zero_width_binary").or_insert(decoded);
                    break;
                }
            }
        }
        if run.len() % 4 == 0 && within(&['\u{200B}', '\u{200C}', '\u{200D}', '\u{2060}']) {
            let dibit = |ch: char| match ch {
                '\u{200B}' => 0u8,
                '\u{200C}' => 1,
                '\u{200D}' => 2,
                _ => 3,
            };
            let raw: Vec<u8> = run
                .chunks_exact(4)
                .map(|quad| quad.iter().fold(0u8, |byte, &ch| (byte << 2) | dibit(ch)))
                .collect();
            if let Some(decoded) = decode_printable_utf8(&raw, 0.8) {
                found.entry("zero_width_binary").or_insert(decoded);
            }
        }
    }
    found
}

/// Decode `paulgb/emoji-encoder`, `prospectorlabs/subtext-spec` and `chinmay29hub/stegmoji` VS runs.
pub fn detect_variation_selector_stego_payload(
    chars: &[char],
    excluded: &BTreeSet<usize>,
) -> IndexMap<&'static str, String> {
    let mut runs: Vec<Vec<u8>> = Vec::new();
    let mut current: Vec<u8> = Vec::new();
    for (index, &ch) in chars.iter().enumerate() {
        if excluded.contains(&index) {
            if !current.is_empty() {
                runs.push(std::mem::take(&mut current));
            }
            continue;
        }
        if let Some(byte) = variation_selector_to_byte(ch) {
            current.push(byte);
        } else if ch == '\u{200A}' && !current.is_empty() {
            // prospectorlabs/subtext-spec inserts U+200A every 30 VS codepoints for LINE compatibility
        } else if !current.is_empty() {
            runs.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        runs.push(current);
    }

    let mut found: IndexMap<&'static str, String> = IndexMap::new();
    for raw in runs.iter().filter(|run| run.len() >= 3) {
        if raw.len() >= 8 && raw[..2] == [0xAB, 0xCD] && raw[6] == 0x01 {
            let body_len = u32::from_be_bytes([raw[2], raw[3], raw[4], raw[5]]) as usize;
            let flags = raw[7];
            let body = &raw[8..raw.len().min(8usize.saturating_add(body_len))];
            if flags == 0
                && body.len() >= 2
                && let Some(decoded) = decode_printable_utf8(&body[1..], 0.8)
            {
                found.insert("subtext", decoded);
                continue;
            }
            found.insert(
                "subtext",
                format!("Subtext v1 frame (flags=0x{flags:02x}, len={body_len})"),
            );
            continue;
        }
        if raw.len() >= 16 && raw.iter().all(|&byte| byte == 14 || byte == 15) {
            let bytes: Vec<u8> = raw[..raw.len() / 8 * 8]
                .chunks_exact(8)
                .map(|bits| {
                    bits.iter()
                        .fold(0u8, |byte, &bit| (byte << 1) | u8::from(bit != 14))
                })
                .collect();
            if bytes.len() >= 2 && bytes[0] <= 1 {
                let payload = if bytes[0] == 1 {
                    inflate(&bytes[1..], true).unwrap_or_default()
                } else {
                    bytes[1..].to_vec()
                };
                if let Some(decoded) = decode_printable_utf8(&payload, 0.8) {
                    found.insert("stegmoji", decoded);
                    continue;
                }
            }
        }
        if let Some(decoded) = decode_printable_utf8(raw, 0.8) {
            found.entry("variation_selector").or_insert(decoded);
        }
    }
    found
}

/// Decode covert payloads across C2PATXT, VSRMark, Innamark, StegCloak, VS encoders and ZW binary.
pub fn decode_stego_payloads(text: &str) -> IndexMap<String, String> {
    let mut payloads: IndexMap<String, String> = IndexMap::new();
    let (after, c2patxt) = strip_c2patxt_suffix(text);
    if let Some(found) = c2patxt {
        payloads.insert(
            "c2patxt".into(),
            format!("{} bytes JUMBF", found.manifest_bytes.len()),
        );
    }
    let chars: Vec<char> = after.chars().collect();
    let vsrmark = detect_vsrmark(&chars);
    let excluded = vsrmark
        .as_ref()
        .map(|found| found.indices.clone())
        .unwrap_or_default();
    if let Some(found) = vsrmark {
        payloads.insert("vsrmark".into(), found.payload);
    }
    if let Some(innamark) = detect_innamark_payload(&after) {
        payloads.insert("innamark".into(), innamark);
    }
    if let Some(stegcloak) = detect_stegcloak_stream(&chars) {
        payloads.insert("stegcloak".into(), stegcloak);
    }
    for (key, value) in detect_zero_width_binary_payload(&chars)
        .into_iter()
        .chain(detect_variation_selector_stego_payload(&chars, &excluded))
    {
        payloads.insert(key.to_owned(), value);
    }
    payloads
}

/// Append hidden printable ASCII encoded in the Unicode Tags block.
pub fn encode_unicode_tags(text: &str, hidden_ascii: &str) -> Result<String, TextError> {
    let mut out = String::with_capacity(text.len() + hidden_ascii.len() * 4 + 8);
    out.push_str(text);
    out.push('\u{E0001}');
    for ch in hidden_ascii.chars() {
        if !(' '..='~').contains(&ch) {
            return Err(TextError::invalid(format!(
                "Unicode Tags payload must be printable ASCII (0x20..0x7E), got {}",
                crate::pyrepr::repr_str(&ch.to_string())
            )));
        }
        out.push(char::from_u32(0xE0000 + ch as u32).unwrap_or('\u{E007F}'));
    }
    out.push('\u{E007F}');
    Ok(out)
}

/// Embed an `Innamark` base-4 whitespace watermark into the ASCII spaces of `text`.
///
/// Compressed bodies use a raw deflate stream that any inflater decodes; its bytes need
/// not equal CPython's `zlib` output.
pub fn encode_innamark(
    text: &str,
    payload: &[u8],
    compressed: bool,
    crc32: bool,
) -> Result<String, TextError> {
    let body = if compressed {
        let mut encoder = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(6));
        encoder
            .write_all(payload)
            .and_then(|()| encoder.try_finish())
            .map_err(|err| TextError::runtime(err.to_string()))?;
        encoder.get_ref().clone()
    } else {
        payload.to_vec()
    };
    let type_tag = 0x20 | if compressed { 0x40 } else { 0 } | if crc32 { 0x10 } else { 0 };
    let total_len = u32::try_from(1 + 4 + if crc32 { 4 } else { 0 } + body.len())
        .map_err(|_| TextError::invalid("Innamark payload is too large"))?;
    let mut tag = vec![type_tag];
    tag.extend_from_slice(&total_len.to_le_bytes());
    if crc32 {
        let mut placeholder = tag.clone();
        placeholder.extend_from_slice(&[0; 4]);
        placeholder.extend_from_slice(&body);
        tag.extend_from_slice(&crc32fast::hash(&placeholder).to_le_bytes());
    }
    tag.extend_from_slice(&body);

    let mut transcoded = vec![INNAMARK_SEPARATOR];
    for &byte in &tag {
        transcoded.extend((0..4).map(|shift| INNAMARK_ALPHABET[usize::from((byte >> (2 * shift)) & 3)]));
    }
    let spaces = text.chars().filter(|&ch| ch == ' ').count();
    if spaces < transcoded.len() {
        return Err(TextError::invalid(format!(
            "Innamark frame requires at least {} ASCII spaces, got {spaces}",
            transcoded.len()
        )));
    }
    let mut position = 0;
    Ok(text
        .chars()
        .map(|ch| {
            if ch != ' ' {
                return ch;
            }
            let replacement = transcoded[position % transcoded.len()];
            position += 1;
            replacement
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn innamark_roundtrip() {
        let cover: Vec<String> = (0..120).map(|i| format!("word{i}")).collect();
        let cover = cover.join(" ");
        for (compressed, crc) in [(true, true), (false, true), (false, false)] {
            let marked = encode_innamark(&cover, b"author:astra-6", compressed, crc).unwrap();
            assert_eq!(
                detect_innamark_payload(&marked).as_deref(),
                Some("author:astra-6")
            );
        }
    }

    #[test]
    fn stegcloak_unencrypted() {
        let chars: Vec<char> =
            "Normal \u{200c}\u{2061}\u{2061}\u{200d}\u{2063}\u{2061}\u{200d}\u{200d}\u{200c} x"
                .chars()
                .collect();
        assert_eq!(detect_stegcloak_stream(&chars).as_deref(), Some("unencrypted:ok"));
    }
}
