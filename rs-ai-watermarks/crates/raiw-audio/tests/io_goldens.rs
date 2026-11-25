//! ffmpeg IO, fidelity and perceptual metrics against `tools/goldens/audio_io.py`.

mod common;

use std::path::Path;

use common::{check_f32, encode_raw, golden, synth};
use raiw_audio::io::{AudioReader, FfmpegTools, codec_args, probe_audio, read_all, write_audio};
use raiw_audio::perceptual::compute_perceptual_metrics;
use raiw_audio::quality::{audio_quality_passed, is_lossy_output, measure_fidelity};
use raiw_audio::{AudioError, Frames};
use raiw_core::sha256_file;
use raiw_core::testing::{assert_json_close, fixture};
use serde_json::{Value, json};

fn tools() -> FfmpegTools {
    FfmpegTools::discover().expect("ffmpeg and ffprobe on PATH")
}

fn args(value: &Value) -> Vec<String> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect()
}

fn make_source(dir: &Path, name: &str, case: &Value) -> std::path::PathBuf {
    let folder = dir.join(name);
    std::fs::create_dir_all(&folder).unwrap();
    let path = folder.join(case["filename"].as_str().unwrap());
    let spec = &case["spec"];
    let encode_args = args(&case["args"]);
    let refs: Vec<&str> = encode_args.iter().map(String::as_str).collect();
    encode_raw(
        &synth(spec),
        spec["rate"].as_u64().unwrap() as u32,
        &path,
        &refs,
        case["layout"].as_str(),
    );
    assert_eq!(
        sha256_file(&path).unwrap(),
        case["file_sha256"].as_str().unwrap(),
        "{name}: source file bytes"
    );
    path
}

#[test]
fn probe_decode_encode_and_fidelity_match_python() {
    let golden = golden("io");
    let tools = tools();
    let dir = tempfile::tempdir().unwrap();
    let mut decoded = std::collections::HashMap::new();
    for (name, case) in golden["sources"].as_object().unwrap() {
        let path = make_source(dir.path(), name, case);
        let info = probe_audio(&tools, &path, 0).unwrap();
        assert_eq!(
            serde_json::to_value(&info).unwrap(),
            case["info"],
            "{name}: probe"
        );
        let frames = read_all(&tools, &path, &info).unwrap();
        assert!(
            check_f32(frames.as_slice(), &case["decoded"], 1e-12, name),
            "{name}: decoded samples bit-identical"
        );
        let lengths: Vec<u64> = AudioReader::open(&tools, &path, &info, 12_345)
            .unwrap()
            .map(|c| c.unwrap().frames() as u64)
            .collect();
        assert_eq!(json!(lengths), case["chunk_lengths_12345"], "{name}: chunking");
        decoded.insert(name.clone(), (path, info, frames));
    }
    for case in golden["writes"].as_array().unwrap() {
        let name = case["source"].as_str().unwrap();
        let suffix = case["suffix"].as_str().unwrap();
        let bitexact = case["bitexact"].as_bool().unwrap();
        let context = format!("{name} -> {suffix} bitexact={bitexact}");
        let (path, info, frames) = &decoded[name];
        let output = path.with_file_name(format!("out{}{suffix}", if bitexact { "-bitexact" } else { "" }));
        let pieces = if frames.frames() > 20_000 {
            vec![frames.slice(0..20_000), frames.slice(20_000..frames.frames())]
        } else {
            vec![frames.clone()]
        };
        let written = write_audio(
            &tools,
            info,
            &output,
            pieces.into_iter().map(Ok),
            Some(frames.frames()),
            bitexact,
        );
        if let Some(message) = case["write_error"].as_str() {
            assert_eq!(written.unwrap_err().to_string(), message, "{context}");
            assert!(!output.exists(), "{context}: failed write must not publish");
            continue;
        }
        assert_eq!(
            json!(codec_args(suffix, info, bitexact).unwrap()),
            case["codec_args"],
            "{context}"
        );
        assert_eq!(
            written.unwrap() as u64,
            case["count"].as_u64().unwrap(),
            "{context}"
        );
        // Matroska segment UIDs and Ogg stream serials are random per mux.
        if !matches!(suffix, ".mka" | ".ogg" | ".oga" | ".opus") {
            assert_eq!(
                sha256_file(&output).unwrap(),
                case["file_sha256"].as_str().unwrap(),
                "{context}: output bytes"
            );
        }
        let delivered = probe_audio(&tools, &output, 0).unwrap();
        assert_eq!(
            serde_json::to_value(&delivered).unwrap(),
            case["info"],
            "{context}: delivered probe"
        );
        let samples = read_all(&tools, &output, &delivered).unwrap();
        assert!(
            check_f32(samples.as_slice(), &case["decoded"], 1e-12, &context),
            "{context}: decoded output"
        );
        match measure_fidelity(&tools, path, info, &output, 0) {
            Ok(fidelity) => {
                assert_json_close(
                    &serde_json::to_value(&fidelity).unwrap(),
                    &case["fidelity"],
                    1e-9,
                    &context,
                );
                assert_eq!(
                    audio_quality_passed(&fidelity, is_lossy_output(&output)),
                    case["quality_passed"].as_bool().unwrap(),
                    "{context}"
                );
            }
            Err(err) => assert_eq!(
                Some(err.to_string().as_str()),
                case["fidelity_error"].as_str(),
                "{context}"
            ),
        }
        let perceptual = compute_perceptual_metrics(&tools, path, info, &output, 0, None).unwrap();
        assert_json_close(
            &serde_json::to_value(&perceptual).unwrap(),
            &case["perceptual"],
            1e-7,
            &context,
        );
    }
}

#[test]
fn rejected_writes_report_python_messages() {
    let golden = golden("io");
    let errors = &golden["errors"];
    let tools = tools();
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    let stereo = make_source(&root, "s16_stereo48k", &golden["sources"]["s16_stereo48k"]);
    let odd = make_source(&root, "odd_rate19k", &golden["sources"]["odd_rate19k"]);
    let side = make_source(&root, "side51_16k", &golden["sources"]["side51_16k"]);
    let (stereo_info, odd_info, side_info) = (
        probe_audio(&tools, &stereo, 0).unwrap(),
        probe_audio(&tools, &odd, 0).unwrap(),
        probe_audio(&tools, &side, 0).unwrap(),
    );
    let stereo_frames = read_all(&tools, &stereo, &stereo_info).unwrap();
    let message = |key: &str| {
        errors[key]["message"]
            .as_str()
            .unwrap()
            .replace("<ROOT0>", &root.display().to_string())
    };
    let write = |info, name: &str, frames: Frames, expected, bitexact| {
        write_audio(
            &tools,
            info,
            &root.join(name),
            std::iter::once(Ok(frames)),
            expected,
            bitexact,
        )
        .unwrap_err()
        .to_string()
    };
    assert_eq!(
        write(&odd_info, "odd.mp3", Frames::zeros(19_000, 1), None, false),
        message("mp3_rate")
    );
    assert_eq!(
        codec_args(".mp3", &side_info, false).unwrap_err().to_string(),
        message("mp3_channels")
    );
    assert_eq!(
        codec_args(".opus", &side_info, false).unwrap_err().to_string(),
        message("opus_channels")
    );
    assert_eq!(
        codec_args(".xyz", &stereo_info, false).unwrap_err().to_string(),
        message("bad_suffix")
    );
    assert_eq!(
        write(&stereo_info, "b.flac", stereo_frames.clone(), None, true),
        message("bitexact_flac")
    );
    assert_eq!(
        write(
            &stereo_info,
            "r.flac",
            Frames::zeros(10, 2).map(|_| 1.5),
            None,
            false
        ),
        message("flac_range")
    );
    assert_eq!(
        write(
            &stereo_info,
            "n.wav",
            Frames::zeros(10, 2).map(|_| f32::NAN),
            None,
            false
        ),
        message("nan")
    );
    assert_eq!(
        write(&stereo_info, "c.wav", Frames::zeros(10, 1), None, false),
        message("channels")
    );
    let short = stereo_frames.slice(0..stereo_frames.frames() - 1);
    assert_eq!(
        write(&stereo_info, "k.wav", short, Some(stereo_frames.frames()), false),
        message("count")
    );
    let junk = root.join("not-audio.bin");
    std::fs::write(&junk, b"not a media container").unwrap();
    assert_eq!(
        probe_audio(&tools, &junk, 0).unwrap_err().to_string(),
        message("no_audio")
    );
    assert_eq!(
        probe_audio(&tools, &stereo, 3).unwrap_err().to_string(),
        message("missing_stream")
    );
    assert!(matches!(
        probe_audio(&tools, &root.join("missing.wav"), 0),
        Err(AudioError::NotFound(_))
    ));
    let leftovers: Vec<_> = std::fs::read_dir(&root)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_string_lossy().starts_with('.'))
        .collect();
    assert!(
        leftovers.is_empty(),
        "staged files must be cleaned up: {leftovers:?}"
    );
}

#[test]
fn windowed_gate_catches_mid_track_dropout_and_drift() {
    let golden = golden("io");
    let case = &golden["dropout"];
    let tools = tools();
    let dir = tempfile::tempdir().unwrap();
    let clean = synth(&case["spec"]);
    let mut dropped = clean.clone();
    let channels = dropped.channels();
    dropped.as_mut_slice()[16_000 * channels..40_000 * channels].fill(0.0);
    let swapped = Frames::from_channels(&[clean.channel(1), clean.channel(0)]);
    let (source, output, swap) = (
        dir.path().join("drop-src.wav"),
        dir.path().join("drop-out.wav"),
        dir.path().join("swap.wav"),
    );
    for (frames, path) in [(&clean, &source), (&dropped, &output), (&swapped, &swap)] {
        encode_raw(frames, 16_000, path, &["-c:a", "pcm_s16le"], None);
    }
    assert_eq!(
        sha256_file(&source).unwrap(),
        case["source_sha256"].as_str().unwrap()
    );
    assert_eq!(
        sha256_file(&output).unwrap(),
        case["output_sha256"].as_str().unwrap()
    );
    let info = probe_audio(&tools, &source, 0).unwrap();
    let fidelity = measure_fidelity(&tools, &source, &info, &output, 0).unwrap();
    assert_json_close(
        &serde_json::to_value(&fidelity).unwrap(),
        &case["fidelity"],
        1e-9,
        "dropout",
    );
    assert!(fidelity.worst_window_rms_ratio < 1e-4);
    assert!(!audio_quality_passed(&fidelity, false));
    let swapped_fidelity = measure_fidelity(&tools, &source, &info, &swap, 0).unwrap();
    assert_json_close(
        &serde_json::to_value(&swapped_fidelity).unwrap(),
        &case["swapped_fidelity"],
        1e-9,
        "swap",
    );
}

#[test]
fn tracked_aac_fixtures_probe_and_decode_identically() {
    let golden = golden("io");
    let tools = tools();
    for case in golden["fixtures"].as_array().unwrap() {
        let path = fixture(case["fixture"].as_str().unwrap());
        let info = probe_audio(&tools, &path, 0).unwrap();
        assert_eq!(
            serde_json::to_value(&info).unwrap(),
            case["info"],
            "{}",
            path.display()
        );
        let frames = read_all(&tools, &path, &info).unwrap();
        assert!(check_f32(
            frames.as_slice(),
            &case["decoded"],
            1e-12,
            &path.display().to_string()
        ));
    }
}

#[test]
fn failing_encoder_keeps_existing_output() {
    let tools = tools();
    let dir = tempfile::tempdir().unwrap();
    let golden = golden("io");
    let source = make_source(dir.path(), "s16_stereo48k", &golden["sources"]["s16_stereo48k"]);
    let info = probe_audio(&tools, &source, 0).unwrap();
    let broken = dir.path().join("broken-ffmpeg");
    std::fs::write(
        &broken,
        "#!/bin/sh\necho 'synthetic encoder failure' >&2\nexit 17\n",
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&broken, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    let output = dir.path().join("output.wav");
    std::fs::write(&output, b"existing output").unwrap();
    let failing = FfmpegTools::new(&broken, &tools.ffprobe);
    let err = write_audio(
        &failing,
        &info,
        &output,
        std::iter::once(Ok(Frames::zeros(100_000, 2))),
        None,
        false,
    )
    .unwrap_err();
    assert!(err.to_string().contains("synthetic encoder failure"), "{err}");
    assert!(err.to_string().contains("(17)"), "{err}");
    let late = write_audio(
        &tools,
        &info,
        &output,
        vec![
            Ok(Frames::zeros(256_000, 2)),
            Err(AudioError::failed("late upstream failure")),
        ],
        None,
        false,
    )
    .unwrap_err();
    assert_eq!(late.to_string(), "late upstream failure");
    assert_eq!(std::fs::read(&output).unwrap(), b"existing output");
    assert!(
        std::fs::read_dir(dir.path()).unwrap().all(|e| !e
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".output-"))
    );
}

#[test]
fn bitexact_wav_is_byte_stable_across_chunking() {
    let tools = tools();
    let dir = tempfile::tempdir().unwrap();
    let golden = golden("io");
    let source = make_source(dir.path(), "s16_stereo48k", &golden["sources"]["s16_stereo48k"]);
    let info = probe_audio(&tools, &source, 0).unwrap();
    let samples = read_all(&tools, &source, &info).unwrap();
    let (first, second, third) = (
        dir.path().join("first.wav"),
        dir.path().join("second.wav"),
        dir.path().join("third.wav"),
    );
    write_audio(
        &tools,
        &info,
        &first,
        std::iter::once(Ok(samples.clone())),
        None,
        true,
    )
    .unwrap();
    let pieces = vec![
        samples.slice(0..1),
        samples.slice(1..2000),
        samples.slice(2000..samples.frames()),
    ];
    write_audio(&tools, &info, &second, pieces.into_iter().map(Ok), None, true).unwrap();
    let first_info = probe_audio(&tools, &first, 0).unwrap();
    let reader = AudioReader::open(&tools, &first, &first_info, 777).unwrap();
    write_audio(&tools, &first_info, &third, reader, None, true).unwrap();
    let bytes = std::fs::read(&first).unwrap();
    assert_eq!(bytes, std::fs::read(&second).unwrap());
    assert_eq!(bytes, std::fs::read(&third).unwrap());
    assert!(!bytes.windows(4).any(|w| w == b"Lavf" || w == b"JUNK"));
    assert_eq!(read_all(&tools, &first, &first_info).unwrap(), samples);
}
