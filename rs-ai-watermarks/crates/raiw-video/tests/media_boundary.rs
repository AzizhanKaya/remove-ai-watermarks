//! Phase 1 goldens: probing, decoding, command construction, NUT timing, and fidelity.

mod common;

use std::path::Path;
use std::process::Command;

use common::{clip, num, resolve, sha256_hex};
use raiw_core::testing::{assert_json_close, golden};
use raiw_video::encode::{RawVideoOptions, raw_video_command, video_end_repair};
use raiw_video::nut::{NutPixelFormat, TimestampedNutWriter};
use raiw_video::probe::{
    Rational, VideoEncodeProfile, count_decoded_video_frames, probe_video_duration,
    probe_video_encode_profile, probe_video_keyframes, probe_video_timestamps,
};
use raiw_video::quality::{FidelityOptions, QualityFloors, measure_video_fidelity, video_quality_passed};
use raiw_video::{open_video_capture, open_video_capture_16bit};
use serde_json::{Value, json};

fn py_list_json(values: &[f64]) -> String {
    raiw_core::to_py_json(values, raiw_core::JsonStyle::COMPACT)
}

#[test]
fn probes_match_python_on_clips_and_fixtures() {
    let expected = golden("video", "probe");
    for (key, want) in expected.as_object().expect("probe golden") {
        let path = resolve(key);
        let profile =
            serde_json::to_value(probe_video_encode_profile(&path).expect("profile")).expect("json");
        assert_eq!(profile, want["profile"], "{key} profile");
        let timestamps = probe_video_timestamps(&path).expect("timestamps");
        assert_eq!(
            timestamps.len() as u64,
            want["timestamps_count"].as_u64().expect("count"),
            "{key} count"
        );
        assert_eq!(
            sha256_hex(py_list_json(&timestamps).as_bytes()),
            want["timestamps_sha256"].as_str().expect("sha"),
            "{key} timestamps"
        );
        match probe_video_duration(&path) {
            Ok(duration) => assert_eq!(duration, num(&want["duration"]), "{key} duration"),
            Err(error) => {
                let message = want["duration"].as_str().expect("error string");
                let expected_head = message.split(": /").next().expect("head");
                assert!(
                    format!("error: {error}").starts_with(expected_head),
                    "{key}: {error} vs {message}"
                );
            }
        }
        let keyframes = probe_video_keyframes(&path).expect("keyframes");
        match (&keyframes, want["keyframes"].is_null()) {
            (None, true) => {}
            (Some(spec), false) => {
                let mut value = serde_json::to_value(spec).expect("json");
                value["keyframes"] = json!(spec.keyframes.iter().take(8).collect::<Vec<_>>());
                assert_eq!(value, want["keyframes"], "{key} keyframes");
                assert_eq!(
                    json!(spec.keyframes.len()),
                    want["keyframe_count"],
                    "{key} keyframe count"
                );
            }
            _ => panic!("{key}: keyframe presence differs"),
        }
        let decoded = count_decoded_video_frames(&path).expect("count");
        assert_eq!(json!(decoded), want["decoded_frames"], "{key} decoded frames");
    }
}

#[test]
fn decoded_frames_match_opencv_bytes() {
    let expected = golden("video", "decode");
    for (key, want) in expected.as_object().expect("decode golden") {
        if key.ends_with(":bgr48") {
            continue;
        }
        let path = resolve(key);
        let reader = open_video_capture(&path);
        if !want["opened"].as_bool().unwrap_or(false) {
            assert!(reader.is_err(), "{key} should not open");
            continue;
        }
        let mut reader = reader.unwrap_or_else(|e| panic!("{key}: {e}"));
        assert_eq!(reader.width() as f64, num(&want["width"]), "{key} width");
        assert_eq!(reader.height() as f64, num(&want["height"]), "{key} height");
        assert!(
            (reader.fps() - num(&want["fps"])).abs() < 1e-9,
            "{key} fps {}",
            reader.fps()
        );
        let mut digest = sha2::Sha256::default();
        let mut per_frame = Vec::new();
        while let Some(frame) = reader.read().expect("frame") {
            sha2::Digest::update(&mut digest, frame.as_bytes());
            per_frame.push(sha256_hex(frame.as_bytes()));
        }
        assert_eq!(
            per_frame.len() as u64,
            want["frames"].as_u64().expect("frames"),
            "{key} frame count"
        );
        // cv2 bundles FFmpeg 5.1, whose 10-bit to bgr24 swscale path differs; only 8-bit sources match.
        if key == "clip:hevc_10bit" {
            continue;
        }
        if let Some(frames) = want.get("frame_sha256").and_then(Value::as_array) {
            for (index, (got, exp)) in per_frame.iter().zip(frames).enumerate() {
                assert_eq!(Some(got.as_str()), exp.as_str(), "{key} frame {index}");
            }
        }
        assert_eq!(
            hex::encode(sha2::Digest::finalize(digest)),
            want["sha256"].as_str().expect("sha"),
            "{key}"
        );
    }
}

#[test]
fn sixteen_bit_frames_match_the_python_pipe() {
    let want = &golden("video", "decode")["clip:hevc_10bit:bgr48"];
    let mut reader = open_video_capture_16bit(clip("hevc_10bit")).expect("open");
    let mut digest = sha2::Sha256::default();
    let mut frames = 0;
    while let Some(frame) = reader.read().expect("frame") {
        sha2::Digest::update(&mut digest, frame.as_bytes());
        frames += 1;
    }
    assert_eq!(json!(frames), want["frames"]);
    assert_eq!(
        hex::encode(sha2::Digest::finalize(digest)),
        want["sha256"].as_str().expect("sha")
    );
}

fn profile_named(name: &str) -> VideoEncodeProfile {
    let base = VideoEncodeProfile::default();
    match name {
        "bt709" => VideoEncodeProfile {
            color_range: Some("tv".into()),
            color_space: Some("bt709".into()),
            color_transfer: Some("bt709".into()),
            color_primaries: Some("bt709".into()),
            time_base: Some("1/12800".into()),
            start_pts: Some(0),
            ..base
        },
        "hdr10" => VideoEncodeProfile {
            pixel_format: "yuv420p10le".into(),
            color_range: Some("tv".into()),
            color_space: Some("bt2020nc".into()),
            color_transfer: Some("smpte2084".into()),
            color_primaries: Some("bt2020".into()),
            time_base: Some("1/90000".into()),
            ..base
        },
        "full_range" => VideoEncodeProfile {
            pixel_format: "yuv444p".into(),
            color_range: Some("pc".into()),
            time_base: Some("1001/30000".into()),
            ..base
        },
        _ => base,
    }
}

#[test]
fn raw_encoder_commands_match_python() {
    raiw_video::set_ffmpeg_threads(0);
    for case in golden("video", "commands").as_array().expect("cases") {
        let profile = profile_named(case["profile"].as_str().expect("profile"));
        let crf = case["crf"].as_u64().expect("crf") as u32;
        let suffix = case["suffix"].as_str().expect("suffix");
        let output = if case.get("h264_profile").is_some() {
            format!("seg{suffix}")
        } else {
            format!("out{suffix}")
        };
        let (width, height, fps) = if case.get("h264_profile").is_some() {
            (64, 32, 30.0)
        } else {
            (160, 96, if crf == 0 { 29.970_029_97 } else { 24.0 })
        };
        let mut options = RawVideoOptions::new(width, height, fps, crf, &profile);
        options.timestamped_input = case["timestamped"].as_bool().expect("timestamped");
        options.copy_input_timestamps = case["copyts"].as_bool().expect("copyts");
        options.input_pix_fmt = case["input_pix_fmt"].as_str().expect("pix");
        options.h264_profile = case.get("h264_profile").and_then(Value::as_str);
        let mut command = raw_video_command(Path::new(&output), &options).expect("command");
        command[0] = "ffmpeg".into();
        assert_eq!(json!(command), case["command"], "{case}");
    }
}

#[test]
fn end_repairs_match_python() {
    for case in golden("video", "end_repair").as_array().expect("cases") {
        let result = video_end_repair(&case["source"], &case["encoded"]);
        match (result, &case["result"]) {
            (Ok(repair), Value::Object(want)) => {
                let mut got = serde_json::to_value(&repair).expect("json");
                got["bitstream_filter"] = json!(repair.bitstream_filter());
                assert_eq!(&got, &Value::Object(want.clone()));
            }
            (Err(error), Value::String(want)) => assert_eq!(format!("error: {error}"), *want),
            (got, want) => panic!("{got:?} vs {want}"),
        }
    }
}

#[test]
fn fidelity_reports_match_python() {
    let source = clip("cfr_audio");
    for case in golden("video", "quality").as_array().expect("cases") {
        let name = case["name"].as_str().expect("name");
        let delivered = if name.starts_with("same") {
            clip("delivered_same")
        } else {
            clip("delivered_small")
        };
        let options = FidelityOptions {
            source_geometry: (160, 96, 24.0),
            duration: (name == "small_duration").then_some(1.0),
            preserve_timestamps: name == "same_timestamps",
            source_timestamps: None,
        };
        match (
            measure_video_fidelity(source, delivered, &options),
            &case["report"],
        ) {
            (Ok(report), want @ Value::Object(_)) => {
                let got = serde_json::to_value(&report).expect("json");
                assert_json_close(&got, want, 2e-4, name);
                assert_eq!(
                    json!(video_quality_passed(Some(&report), &QualityFloors::default())),
                    case["passed"]
                );
            }
            (Err(error), Value::String(want)) => assert_eq!(format!("error: {error}"), *want, "{name}"),
            (got, want) => panic!("{name}: {got:?} vs {want}"),
        }
    }
}

#[test]
fn nut_stream_carries_explicit_timestamps_into_ffmpeg() {
    common::ffmpeg_threads();
    let dir = tempfile::tempdir().expect("dir");
    let nut = dir.path().join("frames.nut");
    let tb = Rational::new(1, 90000).expect("tb");
    let timestamps = [0.0, 0.033, 0.1, 0.1333, 0.4];
    let frame: Vec<u8> = (0..32 * 16 * 3).map(|i| (i % 251) as u8).collect();
    {
        let file = std::fs::File::create(&nut).expect("create");
        let mut writer =
            TimestampedNutWriter::new(file, 32, 16, tb, 900, NutPixelFormat::Bgr24).expect("writer");
        for ts in timestamps {
            writer.write(&frame, ts).expect("write");
        }
        writer.finish().expect("finish");
    }
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "packet=pts:stream=codec_name,pix_fmt,width,height,time_base",
        ])
        .args(["-of", "json"])
        .arg(&nut)
        .output()
        .expect("ffprobe");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let probe: Value = serde_json::from_slice(&out.stdout).expect("json");
    let pts: Vec<i64> = probe["packets"]
        .as_array()
        .expect("packets")
        .iter()
        .map(|p| p["pts"].as_i64().expect("pts"))
        .collect();
    assert_eq!(pts, vec![900, 3870, 9900, 12897, 36900]);
    let stream = &probe["streams"][0];
    assert_eq!(stream["codec_name"], "rawvideo");
    assert_eq!(stream["pix_fmt"], "bgr24");
    assert_eq!(stream["time_base"], "1/90000");
    let decoded = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(&nut)
        .args([
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-fps_mode",
            "passthrough",
            "pipe:1",
        ])
        .output()
        .expect("ffmpeg");
    assert_eq!(decoded.stdout.len(), frame.len() * timestamps.len());
    assert_eq!(&decoded.stdout[..frame.len()], frame.as_slice());
}

#[test]
fn nut_carries_sixteen_bit_frames() {
    let dir = tempfile::tempdir().expect("dir");
    let nut = dir.path().join("frames48.nut");
    let frame: Vec<u8> = (0..8 * 4 * 6).map(|i| (i * 7 % 256) as u8).collect();
    {
        let file = std::fs::File::create(&nut).expect("create");
        let tb = Rational::new(1, 1000).expect("tb");
        let mut writer =
            TimestampedNutWriter::new(file, 8, 4, tb, 0, NutPixelFormat::Bgr48Le).expect("writer");
        writer.write(&frame, 0.0).expect("write");
        writer.write(&frame, 0.05).expect("write");
        writer.finish().expect("finish");
    }
    let decoded = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(&nut)
        .args(["-f", "rawvideo", "-pix_fmt", "bgr48le", "pipe:1"])
        .output()
        .expect("ffmpeg");
    assert!(
        decoded.status.success(),
        "{}",
        String::from_utf8_lossy(&decoded.stderr)
    );
    assert_eq!(&decoded.stdout[..frame.len()], frame.as_slice());
}
