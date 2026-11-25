//! `VSRMark` (`com.cognitive-proof.vsrmark.1`): interleaved `VSR1` + CRC32 Variation Selector frames.

use std::collections::BTreeSet;

use crate::error::TextError;
use crate::selectors::{byte_to_variation_selector, variation_selector_to_byte};
use crate::stego::is_zero_width;
use crate::tables::is_registered_variation;
use crate::unicode;

pub const VSRMARK_MAGIC: &[u8; 4] = b"VSR1";
pub const VSRMARK_VERSION: u8 = 1;
pub const VSRMARK_ALGORITHM: &str = "com.cognitive-proof.vsrmark.1";
const HEADER_LEN: usize = 7;
const CRC_LEN: usize = 4;

/// A CRC-validated frame and the codepoint indices of its selectors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VsrMarkMatch {
    pub payload: String,
    pub raw_payload: Vec<u8>,
    pub crc32: u32,
    pub indices: BTreeSet<usize>,
}

/// Locate a CRC-validated frame among the selectors of `chars`.
pub fn detect_vsrmark(chars: &[char]) -> Option<VsrMarkMatch> {
    let mut positions = Vec::new();
    let mut values = Vec::new();
    let mut variant_trailers = BTreeSet::new();
    for (index, &ch) in chars.iter().enumerate() {
        let Some(value) = variation_selector_to_byte(ch) else {
            continue;
        };
        // Interleaving puts a frame byte between a base and its existing selector.
        if index >= 2
            && positions.last() == Some(&(index - 1))
            && is_registered_variation(ch, chars[index - 2])
            && chars
                .get(index + 1)
                .is_none_or(|&next| variation_selector_to_byte(next).is_none())
        {
            variant_trailers.insert(positions.len());
        }
        positions.push(index);
        values.push(value);
    }
    let found = match_frame(&positions, &values);
    if found.is_some() || variant_trailers.is_empty() {
        return found;
    }
    let (positions, values): (Vec<usize>, Vec<u8>) = positions
        .iter()
        .zip(&values)
        .enumerate()
        .filter(|(index, _)| !variant_trailers.contains(index))
        .map(|(_, (&position, &value))| (position, value))
        .unzip();
    match_frame(&positions, &values)
}

fn match_frame(positions: &[usize], stream: &[u8]) -> Option<VsrMarkMatch> {
    let min_frame = HEADER_LEN + CRC_LEN;
    if stream.len() < min_frame {
        return None;
    }
    let mut search = 0;
    loop {
        let offset = search
            + stream[search..]
                .windows(4)
                .position(|window| window == VSRMARK_MAGIC)?;
        if offset + min_frame > stream.len() {
            return None;
        }
        let version = stream[offset + 4];
        let length = usize::from(u16::from_be_bytes([stream[offset + 5], stream[offset + 6]]));
        let body_end = offset + HEADER_LEN + length;
        let frame_end = body_end + CRC_LEN;
        if version == VSRMARK_VERSION && frame_end <= stream.len() {
            let expected = u32::from_be_bytes([
                stream[body_end],
                stream[body_end + 1],
                stream[body_end + 2],
                stream[body_end + 3],
            ]);
            if crc32fast::hash(&stream[offset..body_end]) == expected {
                let raw_payload = stream[offset + HEADER_LEN..body_end].to_vec();
                return Some(VsrMarkMatch {
                    payload: String::from_utf8_lossy(&raw_payload).into_owned(),
                    raw_payload,
                    crc32: expected,
                    indices: positions[offset..frame_end].iter().copied().collect(),
                });
            }
        }
        search = offset + 1;
    }
}

/// Interleave a `VSRMark` frame after the first visible carrier characters of `text`.
pub fn encode_vsrmark(text: &str, payload: &[u8]) -> Result<String, TextError> {
    let length = u16::try_from(payload.len())
        .map_err(|_| TextError::invalid("VSRMark payload cannot exceed 65535 bytes"))?;
    let mut frame = Vec::with_capacity(HEADER_LEN + payload.len() + CRC_LEN);
    frame.extend_from_slice(VSRMARK_MAGIC);
    frame.push(VSRMARK_VERSION);
    frame.extend_from_slice(&length.to_be_bytes());
    frame.extend_from_slice(payload);
    frame.extend_from_slice(&crc32fast::hash(&frame).to_be_bytes());

    let is_carrier = |ch: char| {
        variation_selector_to_byte(ch).is_none() && !unicode::is_space(ch) && !is_zero_width(ch as u32)
    };
    let carriers = text.chars().filter(|&ch| is_carrier(ch)).count();
    if carriers < frame.len() {
        return Err(TextError::invalid(format!(
            "VSRMark frame requires at least {} visible carrier characters, got {carriers}",
            frame.len()
        )));
    }
    let mut out = String::with_capacity(text.len() + frame.len() * 4);
    let mut bytes = frame.iter();
    for ch in text.chars() {
        out.push(ch);
        if is_carrier(ch)
            && let Some(&byte) = bytes.next()
        {
            out.push(byte_to_variation_selector(byte));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let carrier = "This carrier sentence has enough visible characters to hold a VSRMark claim.";
        let marked = encode_vsrmark(carrier, b"{\"id\":1}").unwrap();
        let chars: Vec<char> = marked.chars().collect();
        let found = detect_vsrmark(&chars).unwrap();
        assert_eq!(found.payload, "{\"id\":1}");
        assert_eq!(found.indices.len(), 8 + 11);
    }
}
