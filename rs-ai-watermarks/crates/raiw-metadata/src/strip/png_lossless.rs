//! The chunk-walking PNG strip for sources deeper than 8 bits, and the shared text rules.

use std::path::Path;

use flate2::{Decompress, FlushDecompress, Status};
use raiw_container::png::{ChunkAction, rewrite_chunks};
use raiw_core::constants::C2PA_CHUNK_TYPE;

use super::{StripError, write_output};
use crate::exif;
use crate::exif_ai::scrub_ai_exif;
use crate::py::{latin1, utf8_replace};
use crate::rules::keys::{STANDARD_METADATA_KEYS, is_ai_key, is_ai_value};
use crate::rules::tc260::is_aigc_value_str;
use crate::rules::xai::text_pair_member;

const TEXT_CAP: usize = 1 << 20;

/// `zlib.decompressobj().decompress(data, max_length)`; `None` is `zlib.error`.
pub fn inflate_max(data: &[u8], max_length: usize) -> Option<Vec<u8>> {
    let mut inflater = Decompress::new(true);
    let mut out = Vec::new();
    let mut buf = vec![0u8; 64 * 1024];
    while out.len() < max_length {
        let consumed = inflater.total_in() as usize;
        let want = (max_length - out.len()).min(buf.len());
        let before = inflater.total_out();
        let status = inflater
            .decompress(&data[consumed..], &mut buf[..want], FlushDecompress::None)
            .ok()?;
        let produced = (inflater.total_out() - before) as usize;
        out.extend_from_slice(&buf[..produced]);
        if status == Status::StreamEnd || (produced == 0 && inflater.total_in() as usize == consumed) {
            break;
        }
    }
    Some(out)
}

fn capped(data: &[u8]) -> String {
    utf8_replace(&data[..data.len().min(TEXT_CAP)])
}

/// `forensic_metadata.png_text_decode`: `keyword\0text`, inflated where compressed.
pub fn png_text_decode(ctype: &[u8], body: &[u8]) -> String {
    match ctype {
        b"tEXt" => {
            let mut text = body[..body.len().min(TEXT_CAP)].to_vec();
            if body.len() > TEXT_CAP {
                text.extend_from_slice(b"...TRUNCATED");
            }
            utf8_replace(&text)
        }
        b"zTXt" => {
            let Some(nul) = memchr::memchr(0, body) else {
                return capped(body);
            };
            let keyword = latin1(&body[..nul]);
            let text = match inflate_max(body.get(nul + 2..).unwrap_or_default(), TEXT_CAP + 1) {
                Some(decoded) => {
                    let suffix = if decoded.len() > TEXT_CAP {
                        "...TRUNCATED"
                    } else {
                        ""
                    };
                    format!("{}{suffix}", capped(&decoded))
                }
                None => capped(body),
            };
            format!("{keyword}\0{text}")
        }
        _ => {
            let Some(nul) = memchr::memchr(0, body) else {
                return capped(body);
            };
            let keyword = latin1(&body[..nul]);
            let rest = &body[nul + 1..];
            if rest.len() < 2 {
                return capped(body);
            }
            let compressed = rest[0] != 0;
            let mut tail = &rest[2..];
            for _ in 0..2 {
                let Some(end) = memchr::memchr(0, tail) else {
                    return capped(body);
                };
                tail = &tail[end + 1..];
            }
            let mut tail = tail.to_vec();
            if compressed && let Some(decoded) = inflate_max(&tail, TEXT_CAP + 1) {
                tail = decoded;
            }
            if tail.len() > TEXT_CAP {
                tail.truncate(TEXT_CAP);
                tail.extend_from_slice(b"...TRUNCATED");
            }
            format!("{keyword}\0{}", utf8_replace(&tail))
        }
    }
}

/// `_keep_standard_text_metadata` for a `str` value.
pub fn keep_standard_text(
    key: &str,
    value: Option<&str>,
    keep_standard: bool,
    xai_pair_present: bool,
) -> bool {
    keep_standard
        && STANDARD_METADATA_KEYS.contains(&key)
        && !is_ai_key(key)
        && !value.is_some_and(|v| is_ai_value(v) || is_aigc_value_str(v))
        && !(xai_pair_present && value.is_some_and(|v| text_pair_member(key, v)))
}

/// `_png_chunk_should_be_dropped`.
pub fn png_chunk_should_be_dropped(
    kind: &[u8],
    payload: &[u8],
    keep_standard: bool,
    xai_pair_present: bool,
) -> bool {
    if kind == C2PA_CHUNK_TYPE || kind == b"eXIf" {
        return true;
    }
    if kind == b"iCCP" {
        return !keep_standard;
    }
    if !matches!(kind, b"tEXt" | b"zTXt" | b"iTXt") {
        return false;
    }
    let decoded = png_text_decode(kind, payload);
    let (keyword, text) = decoded.split_once('\0').unwrap_or((&decoded, ""));
    !keep_standard_text(keyword, Some(text), keep_standard, xai_pair_present)
}

/// `_scrubbed_png_exif`: the bare TIFF to keep, `None` to drop an unparseable payload.
pub fn scrubbed_png_exif(payload: &[u8]) -> Option<Vec<u8>> {
    let mut dict = exif::load(payload).ok()?;
    scrub_ai_exif(&mut dict);
    let dumped = exif::dump(&dict).ok()?;
    Some(
        dumped
            .strip_prefix(b"Exif\x00\x00")
            .map_or_else(|| dumped.clone(), <[u8]>::to_vec),
    )
}

/// `_strip_png_metadata_lossless`; `Ok(false)` when the chunk walk fails.
pub fn strip_png_lossless(source: &Path, output: &Path, keep_standard: bool) -> Result<bool, StripError> {
    let data = std::fs::read(source)?;
    if !data.starts_with(raiw_core::constants::PNG_SIGNATURE) {
        return Ok(false);
    }
    let xai_pair = crate::detect::xai_signature(source);
    let rewritten = rewrite_chunks(&data, |kind, payload| {
        if kind == b"eXIf" {
            let scrubbed = if keep_standard {
                scrubbed_png_exif(payload)
            } else {
                None
            };
            return scrubbed.map_or(ChunkAction::Drop, ChunkAction::Replace);
        }
        let text = if matches!(kind, b"tEXt" | b"zTXt" | b"iTXt") {
            payload
        } else {
            &[][..]
        };
        if png_chunk_should_be_dropped(kind, text, keep_standard, xai_pair) {
            ChunkAction::Drop
        } else {
            ChunkAction::Keep
        }
    });
    let Some(rewritten) = rewritten else {
        return Ok(false);
    };
    write_output(output, &rewritten)?;
    Ok(true)
}
