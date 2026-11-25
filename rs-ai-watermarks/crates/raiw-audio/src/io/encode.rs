//! `write_audio`: publish a processed file only after every frame and the encoder succeed.

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use raiw_core::StagedSibling;
use raiw_core::constants::AUDIO_OPUS_EXTENSIONS;
use raiw_core::fs::suffix_lower;

use super::process::Spawned;
use super::{AudioStreamInfo, FfmpegTools, probe_audio};
use crate::error::{AudioError, Result};
use crate::frames::Frames;

const MP3_RATES: [u32; 9] = [8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000];
const SHORT_RIFF_BOUND: usize = 0xFFFF_0000;

/// Output containers by extension, with their fixed codec settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Wav,
    Flac,
    Mka,
    M4a,
    Aac,
    Mp3,
    Opus,
}

impl OutputFormat {
    pub fn from_suffix(suffix: &str) -> Result<Self> {
        Ok(match suffix {
            ".wav" => Self::Wav,
            ".flac" => Self::Flac,
            ".mka" => Self::Mka,
            ".m4a" => Self::M4a,
            ".aac" => Self::Aac,
            ".mp3" => Self::Mp3,
            s if AUDIO_OPUS_EXTENSIONS.contains(&s) => Self::Opus,
            _ => {
                return Err(AudioError::invalid(
                    "Audio sample output must be WAV, FLAC, MP3, M4A, OGG, OGA, OPUS, AAC, or MKA",
                ));
            }
        })
    }

    /// FLAC in its own container or in Matroska.
    pub const fn is_flac(self) -> bool {
        matches!(self, Self::Flac | Self::Mka)
    }
}

/// `_source_pcm_bit_depth`: integer PCM depth worth preserving in lossless output.
fn source_pcm_bit_depth(info: &AudioStreamInfo) -> Option<u32> {
    let fmt16 = matches!(info.sample_fmt.as_deref(), Some("s16" | "s16p"));
    if matches!(info.codec.as_str(), "pcm_s16le" | "pcm_s16be")
        || (fmt16 && matches!(info.bits_per_raw_sample, None | Some(16)))
    {
        return Some(16);
    }
    if matches!(info.codec.as_str(), "pcm_s24le" | "pcm_s24be") || info.bits_per_raw_sample == Some(24) {
        return Some(24);
    }
    None
}

/// `_audio_codec_args(suffix, info, bitexact=...)`.
pub fn codec_args(suffix: &str, info: &AudioStreamInfo, bitexact: bool) -> Result<Vec<String>> {
    let format = OutputFormat::from_suffix(suffix)?;
    let depth = if bitexact {
        None
    } else {
        source_pcm_bit_depth(info)
    };
    let args: &[&str] = match format {
        OutputFormat::Wav => {
            let codec = match depth {
                Some(16) => "pcm_s16le",
                Some(24) => "pcm_s24le",
                _ => "pcm_f32le",
            };
            return Ok(vec![
                "-c:a".into(),
                codec.into(),
                "-rf64".into(),
                if bitexact { "never" } else { "auto" }.into(),
            ]);
        }
        OutputFormat::Flac | OutputFormat::Mka => {
            if depth == Some(16) {
                &["-c:a", "flac", "-sample_fmt", "s16"]
            } else {
                &["-c:a", "flac", "-sample_fmt", "s32"]
            }
        }
        OutputFormat::M4a => &["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"],
        OutputFormat::Aac => &["-c:a", "aac", "-b:a", "192k"],
        OutputFormat::Mp3 => {
            if !matches!(info.channels, 1 | 2) {
                return Err(AudioError::invalid(
                    "MP3 output supports mono or stereo; use WAV, FLAC, MKA, or M4A",
                ));
            }
            if !MP3_RATES.contains(&info.sample_rate) {
                return Err(AudioError::invalid(format!(
                    "MP3 output does not support {} Hz; use WAV, FLAC, MKA, or M4A",
                    info.sample_rate
                )));
            }
            &["-c:a", "libmp3lame", "-b:a", "256k", "-write_xing", "1"]
        }
        OutputFormat::Opus => {
            if !matches!(info.channels, 1 | 2) {
                return Err(AudioError::invalid(
                    "Opus/Ogg output supports mono or stereo; use WAV, FLAC, MKA, or M4A",
                ));
            }
            &["-c:a", "libopus", "-b:a", "192k"]
        }
    };
    Ok(args.iter().map(|&s| s.to_owned()).collect())
}

/// Encode `chunks` into `output`; returns the frame count. The staged file replaces `output`
/// only after the encoder exits cleanly and the delivered geometry matches the source.
pub fn write_audio<I>(
    tools: &FfmpegTools,
    info: &AudioStreamInfo,
    output: &Path,
    chunks: I,
    expected_frames: Option<usize>,
    bitexact: bool,
) -> Result<usize>
where
    I: IntoIterator<Item = Result<Frames>>,
{
    let suffix = suffix_lower(output);
    if bitexact && suffix != ".wav" {
        return Err(AudioError::invalid(
            "Bitexact serialization requires floating-point WAV",
        ));
    }
    let codec = codec_args(&suffix, info, bitexact)?;
    let format = OutputFormat::from_suffix(&suffix)?;
    let staged = StagedSibling::new(output)?;
    let mut command = Command::new(&tools.ffmpeg);
    command
        .args(["-nostdin", "-y", "-v", "error", "-f", "f32le"])
        .args([
            "-ar",
            &info.sample_rate.to_string(),
            "-ac",
            &info.channels.to_string(),
        ]);
    if let Some(layout) = &info.channel_layout {
        command.args(["-channel_layout", layout]);
    }
    command
        .args(["-i", "pipe:0", "-map_metadata", "-1", "-map_chapters", "-1"])
        .args(&codec);
    if bitexact {
        command.args(["-flags:a", "+bitexact", "-fflags", "+bitexact"]);
    }
    command.arg(staged.path());
    let mut process = Spawned::spawn(command, Stdio::piped(), Stdio::null())?;
    let mut stdin = process
        .child
        .stdin
        .take()
        .ok_or_else(|| AudioError::failed("Audio encoder did not open its input pipe"))?;
    let mut count = 0usize;
    for chunk in chunks {
        let frames = chunk?;
        if frames.channels() != info.channels {
            return Err(AudioError::invalid(
                "Audio chunk channel count differs from the source stream",
            ));
        }
        if bitexact && (count + frames.frames()) * info.channels * 4 > SHORT_RIFF_BOUND {
            return Err(AudioError::invalid(
                "Bitexact WAV exceeds the short RIFF size bound; use the default RF64 writer",
            ));
        }
        if !frames.is_finite() {
            return Err(AudioError::invalid("Audio samples must be finite"));
        }
        if format.is_flac() && frames.as_slice().iter().any(|&v| !(-1.0..=1.0).contains(&v)) {
            return Err(AudioError::invalid(
                "FLAC/MKA cannot preserve samples outside [-1, 1]; use floating-point WAV",
            ));
        }
        if let Err(err) = stdin.write_all(&frames.to_le_bytes()) {
            drop(stdin);
            if err.kind() == std::io::ErrorKind::BrokenPipe {
                process.finish()?;
                return Err(AudioError::failed("Audio encoder stopped accepting samples"));
            }
            return Err(err.into());
        }
        count += frames.frames();
    }
    if count == 0 || expected_frames.is_some_and(|e| e != count) {
        let expected = expected_frames.map_or_else(|| "None".to_owned(), |e| e.to_string());
        return Err(AudioError::invalid(format!(
            "Audio frame count mismatch: wrote {count}, expected {expected}"
        )));
    }
    drop(stdin);
    process.finish()?;
    let delivered = probe_audio(tools, staged.path(), 0)?;
    let expected_rate = if format == OutputFormat::Opus {
        48_000
    } else {
        info.sample_rate
    };
    if (delivered.sample_rate, delivered.channels) != (expected_rate, info.channels) {
        return Err(AudioError::failed(
            "Audio encoder changed the sample rate or channel count",
        ));
    }
    if info.channel_layout.is_some()
        && matches!(
            format,
            OutputFormat::Wav | OutputFormat::Flac | OutputFormat::M4a | OutputFormat::Mka
        )
        && delivered.channel_layout != info.channel_layout
    {
        return Err(AudioError::failed("Audio encoder changed the channel layout"));
    }
    staged.commit()?;
    Ok(count)
}
