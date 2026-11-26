//! Strict native audio creation declarations (`_internal/audio_creation.py`).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{LazyLock, Mutex};

use regex::Regex;
use serde_json::Value;

use crate::cache::{ContentKey, LruCache};
use crate::hooks::Hooks;

static CREATION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"(?i-u)^made with suno(?P<studio> studio)?; created=",
        r"(?P<created>[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z)",
        r"(?:; id=(?P<id>[0-9a-f-]{36}))?$"
    ))
    .unwrap_or_else(|err| unreachable!("static regex: {err}"))
});

const MAX_RESPONSE: usize = 1024 * 1024;

fn valid_timestamp(created: &str) -> bool {
    let num = |range: std::ops::Range<usize>| created.get(range).and_then(|s| s.parse::<u32>().ok());
    let (Some(year), Some(month), Some(day), Some(hour), Some(minute), Some(second)) = (
        num(0..4),
        num(5..7),
        num(8..10),
        num(11..13),
        num(14..16),
        num(17..19),
    ) else {
        return false;
    };
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return false,
    };
    year >= 1 && (1..=days).contains(&day) && hour < 24 && minute < 60 && second < 60
}

fn canonical_uuid(id: &str) -> bool {
    let lower = id.to_ascii_lowercase();
    lower.len() == 36
        && lower.char_indices().all(|(i, c)| {
            if matches!(i, 8 | 13 | 18 | 23) {
                c == '-'
            } else {
                c.is_ascii_hexdigit()
            }
        })
}

/// `creation_tool(str)`: a complete machine declaration, never an incidental mention.
pub fn creation_tool_str(value: &str) -> Option<&'static str> {
    let caps = CREATION.captures(crate::py::strip_chars(value, &[' ', '\0', '\u{feff}']))?;
    let studio = caps.name("studio").is_some();
    let id = caps.name("id");
    if !studio && id.is_none() {
        return None;
    }
    if !valid_timestamp(&caps["created"]) {
        return None;
    }
    if let Some(id) = id
        && !canonical_uuid(id.as_str())
    {
        return None;
    }
    Some(if studio { "Suno Studio" } else { "Suno" })
}

fn decode_utf16(value: &[u8], little: bool) -> Option<String> {
    if !value.len().is_multiple_of(2) {
        return None;
    }
    let units = value.chunks_exact(2).map(|c| {
        if little {
            u16::from_le_bytes([c[0], c[1]])
        } else {
            u16::from_be_bytes([c[0], c[1]])
        }
    });
    char::decode_utf16(units).collect::<Result<String, _>>().ok()
}

/// `creation_tool(bytes)`: tried as UTF-8 (BOM optional) and the UTF-16 forms.
pub fn creation_tool_bytes(value: &[u8]) -> Option<&'static str> {
    let utf8 = std::str::from_utf8(value.strip_prefix(b"\xef\xbb\xbf").unwrap_or(value)).ok();
    let utf16 = if let Some(rest) = value.strip_prefix(b"\xff\xfe") {
        decode_utf16(rest, true)
    } else if let Some(rest) = value.strip_prefix(b"\xfe\xff") {
        decode_utf16(rest, false)
    } else {
        decode_utf16(value, true)
    };
    let attempts = [
        utf8.map(str::to_owned),
        utf16,
        decode_utf16(value, true),
        decode_utf16(value, false),
    ];
    attempts
        .into_iter()
        .flatten()
        .find_map(|text| creation_tool_str(&text))
}

static PROBES: Mutex<LruCache<Option<&'static str>>> = Mutex::new(LruCache::new(256));

fn ffprobe() -> Option<PathBuf> {
    which_on_path("ffprobe")
}

/// The first executable named `name` on `PATH` (`shutil.which`).
pub fn which_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}

fn probe(path: &Path, ffprobe: &Path) -> Option<&'static str> {
    let output = Command::new(ffprobe)
        .args([
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type:stream_disposition=attached_pic:stream_tags=comment:format_tags=comment",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .ok()?;
    if !output.status.success() || output.stdout.len() > MAX_RESPONSE {
        return None;
    }
    let document: Value = serde_json::from_slice(&output.stdout).ok()?;
    let streams = document
        .get("streams")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let codec = |s: &Value| s.get("codec_type").and_then(Value::as_str).map(str::to_owned);
    if !streams.iter().any(|s| codec(s).as_deref() == Some("audio")) {
        return None;
    }
    let attached = |s: &Value| {
        s.get("disposition")
            .and_then(|d| d.get("attached_pic"))
            .is_some_and(|v| v.as_i64().is_some_and(|n| n != 0) || v.as_bool() == Some(true))
    };
    if streams
        .iter()
        .any(|s| codec(s).as_deref() == Some("video") && !attached(s))
    {
        return None;
    }
    let mut groups = vec![document.get("format").and_then(|f| f.get("tags")).cloned()];
    groups.extend(
        streams
            .iter()
            .filter(|s| codec(s).as_deref() == Some("audio"))
            .map(|s| s.get("tags").cloned()),
    );
    groups.into_iter().flatten().find_map(|tags| {
        tags.as_object()?.iter().find_map(|(key, value)| {
            (raiw_c2pa::text::casefold(key) == "comment")
                .then(|| value.as_str().and_then(creation_tool_str))
                .flatten()
        })
    })
}

/// `audio_creation_tool(path)`: a declaration read only from an audio container.
pub fn audio_creation_tool(path: &Path) -> Option<&'static str> {
    let head = crate::scan::read_head(path, 12).ok()?;
    if head.get(4..8) == Some(b"ftyp") {
        if !raiw_container::isobmff::has_audio_creation_comment(path, &Hooks).ok()? {
            return None;
        }
    } else {
        let audio = head.starts_with(b"ID3")
            || head.starts_with(b"fLaC")
            || head.starts_with(b"OggS")
            || (head.starts_with(b"RIFF") && head.get(8..12) == Some(b"WAVE"))
            || (head.len() >= 2 && head[0] == 0xFF && head[1] & 0xE0 == 0xE0);
        if !audio {
            return None;
        }
    }
    let ffprobe = ffprobe()?;
    let resolved = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let key = ContentKey::of(&resolved)?;
    if let Ok(cache) = PROBES.lock()
        && let Some(hit) = cache.get(&key)
    {
        return hit;
    }
    let tool = probe(&resolved, &ffprobe);
    if let Ok(mut cache) = PROBES.lock() {
        cache.put(key, tool);
    }
    tool
}
