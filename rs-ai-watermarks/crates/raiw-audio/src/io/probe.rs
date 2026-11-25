//! `probe_audio`: one audio-stream ordinal's native geometry through ffprobe.

use std::path::Path;
use std::process::Command;

use serde_json::Value;

use super::{AudioStreamInfo, FfmpegTools, Fraction};
use crate::error::{AudioError, Result};

const ENTRIES: &str = "stream=index,codec_name,sample_fmt,bits_per_raw_sample,sample_rate,channels,channel_layout,\
time_base,start_pts,start_time,duration_ts,duration:stream_tags=:format=format_name:format_tags=";

/// Python `str(value)` for a JSON scalar, as `_fraction` and `int(...)` receive it.
fn text(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(if *b { "True" } else { "False" }.into()),
        Value::Null | Value::Array(_) | Value::Object(_) => None,
    }
}

fn fraction(value: &Value) -> Result<Option<Fraction>, ()> {
    match text(value).as_deref() {
        None | Some("" | "N/A") => Ok(None),
        Some(s) => s.parse().map(Some).map_err(|_| ()),
    }
}

fn int(value: &Value) -> Result<i128, ()> {
    text(value).ok_or(())?.trim().parse().map_err(|_| ())
}

fn present(value: &Value) -> bool {
    !matches!(text(value).as_deref(), None | Some("N/A"))
}

/// Inspect the requested audio-stream ordinal; reject unreadable input.
pub fn probe_audio(tools: &FfmpegTools, source: &Path, stream: usize) -> Result<AudioStreamInfo> {
    let source = std::fs::canonicalize(source).unwrap_or_else(|_| source.to_path_buf());
    if !source.is_file() {
        return Err(AudioError::NotFound(source.display().to_string()));
    }
    let output = Command::new(&tools.ffprobe)
        .args([
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            ENTRIES,
            "-of",
            "json",
        ])
        .arg(&source)
        .output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(AudioError::invalid(format!(
            "Cannot read audio from {}: {}",
            source.display(),
            stderr.trim()
        )));
    }
    parse(&output.stdout, stream).map_err(|()| {
        AudioError::invalid(format!(
            "No readable audio stream {stream} in {}",
            source.display()
        ))
    })
}

fn parse(stdout: &[u8], stream: usize) -> Result<AudioStreamInfo, ()> {
    let payload: Value = serde_json::from_slice(stdout).map_err(|_| ())?;
    let raw = payload.get("streams").and_then(|s| s.get(stream)).ok_or(())?;
    let rate = int(&raw["sample_rate"])?;
    let channels = int(&raw["channels"])?;
    if rate <= 0 || channels <= 0 {
        return Err(());
    }
    let time_base = fraction(&raw["time_base"])?;
    let mut start = fraction(&raw["start_time"])?;
    let mut duration = fraction(&raw["duration"])?;
    if let Some(base) = time_base {
        if present(&raw["start_pts"]) {
            start = Some(base.mul_int(int(&raw["start_pts"])?));
        }
        if present(&raw["duration_ts"]) {
            duration = Some(base.mul_int(int(&raw["duration_ts"])?));
        }
    }
    let bits = text(&raw["bits_per_raw_sample"])
        .filter(|s| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()))
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&b| b > 0);
    let sample_fmt = text(&raw["sample_fmt"]).filter(|s| !s.is_empty() && s != "N/A");
    let layout = text(&raw["channel_layout"])
        .filter(|s| !s.is_empty())
        .or_else(|| match channels {
            1 => Some("mono".into()),
            2 => Some("stereo".into()),
            _ => None,
        });
    let codec = text(raw.get("codec_name").ok_or(())?).ok_or(())?;
    let format_name = payload
        .get("format")
        .and_then(|f| f.get("format_name"))
        .and_then(text)
        .ok_or(())?;
    Ok(AudioStreamInfo {
        index: usize::try_from(int(&raw["index"])?).map_err(|_| ())?,
        sample_rate: u32::try_from(rate).map_err(|_| ())?,
        channels: usize::try_from(channels).map_err(|_| ())?,
        codec,
        start_time: start,
        duration,
        format_name,
        channel_layout: layout,
        sample_fmt,
        bits_per_raw_sample: bits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ffprobe_json() {
        let stdout = br#"{"streams":[{"index":1,"codec_name":"aac","sample_fmt":"fltp","sample_rate":"44100",
            "channels":2,"channel_layout":"stereo","time_base":"1/44100","start_pts":0,"start_time":"0.000000",
            "duration_ts":224384,"duration":"5.088005"}],"format":{"format_name":"mov,mp4,m4a,3gp,3g2,mj2"}}"#;
        let info = parse(stdout, 0).unwrap();
        assert_eq!(
            info.duration.unwrap().to_string(),
            "224384/44100".parse::<Fraction>().unwrap().to_string()
        );
        assert_eq!(info.start_time, Some(Fraction::ZERO));
        assert_eq!(info.bits_per_raw_sample, None);
        assert!(parse(stdout, 1).is_err());
    }
}
