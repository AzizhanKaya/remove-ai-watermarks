//! `video message`: the known-message accumulator against Python, and the identify / detect /
//! remove plumbing over the golden VideoSeal-embedded clip with a fake decoder.

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use base64::Engine;
use raiw_core::testing::golden;
use raiw_video::messages::scores::{MessageAccumulator, reference_bits};
use raiw_video::messages::{MessageScheme, VideoMessageModel, VideoMessageModels};
use raiw_video::{
    MessageRemovalOptions, NeuralBackends, VideoRuntime, cli, detect_video_watermark_presence,
    identify_video_message, remove_video_message,
};
use serde_json::Value;

fn b64(text: &str) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD
        .decode(text)
        .expect("base64")
}

#[test]
fn accumulator_matches_python_over_uneven_batches() {
    let doc = golden("video_messages", "scores");
    for (name, case) in doc["cases"].as_object().expect("cases") {
        let expected: Vec<bool> = case["expected"]
            .as_array()
            .expect("bits")
            .iter()
            .map(|b| b.as_u64() == Some(1))
            .collect();
        let bits = expected.len();
        let raw = b64(case["logits_f32le_b64"].as_str().expect("logits"));
        let logits: Vec<Vec<f32>> = raw
            .chunks_exact(4 * bits)
            .map(|row| {
                row.chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect()
            })
            .collect();
        assert_eq!(logits.len() as u64, case["frames"].as_u64().expect("frames"));
        let mut accumulator = MessageAccumulator::new(expected).expect("accumulator");
        let mut start = 0;
        for size in case["batches"].as_array().expect("batches") {
            let size = size.as_u64().expect("size") as usize;
            accumulator.update(&logits[start..start + size]).expect("update");
            start += size;
        }
        let reading = serde_json::to_value(accumulator.finish().expect("finish")).expect("json");
        assert_eq!(reading, case["reading"], "{name}");
    }
    for (hex, bits) in doc["reference_bits"].as_object().expect("bits") {
        let want: Vec<bool> = bits
            .as_array()
            .expect("bits")
            .iter()
            .map(|b| b.as_u64() == Some(1))
            .collect();
        assert_eq!(reference_bits(hex, 256).expect("parse"), want);
    }
    assert!(reference_bits("abc", 256).is_err());
    assert!(reference_bits(&"g".repeat(64), 256).is_err());
}

#[test]
fn accumulator_poisons_after_invalid_input() {
    let mut accumulator = MessageAccumulator::new(vec![true; 256]).expect("accumulator");
    assert!(accumulator.update(&[vec![f32::NAN; 256]]).is_err());
    assert!(accumulator.update(&[vec![1.0; 256]]).is_err());
    assert!(accumulator.finish().is_err());
}

/// Returns the reference message for every frame (strong presence), or noise.
#[derive(Debug)]
struct FakeDecoder {
    scheme: MessageScheme,
    message: Vec<bool>,
}

impl VideoMessageModel for FakeDecoder {
    fn scheme(&self) -> MessageScheme {
        self.scheme
    }

    fn predict(
        &self,
        frames: &[f32],
        count: usize,
        height: usize,
        width: usize,
    ) -> raiw_video::Result<Vec<Vec<f32>>> {
        assert_eq!(frames.len(), count * height * width * 3);
        Ok((0..count)
            .map(|_| {
                std::iter::once(4.0)
                    .chain(self.message.iter().map(|b| if *b { 2.0 } else { -2.0 }))
                    .collect()
            })
            .collect())
    }
}

#[derive(Debug)]
struct FakeModels(Vec<bool>);

impl VideoMessageModels for FakeModels {
    fn load(&self, scheme: MessageScheme) -> raiw_video::Result<Arc<dyn VideoMessageModel>> {
        let mut message = self.0.clone();
        message.resize(scheme.bits(), false);
        Ok(Arc::new(FakeDecoder { scheme, message }))
    }
}

fn golden_clip(dir: &Path) -> (PathBuf, String) {
    let doc = golden("video_messages", "clip");
    let path = dir.join("marked.mp4");
    std::fs::write(&path, b64(doc["clip_b64"].as_str().expect("clip"))).expect("write clip");
    (path, doc["reference"].as_str().expect("reference").to_owned())
}

#[test]
fn identify_and_detect_follow_the_decoder_and_bind_the_hash() {
    common::ffmpeg_threads();
    let dir = tempfile::tempdir().expect("dir");
    let (clip, reference) = golden_clip(dir.path());
    let doc = golden("video_messages", "clip");
    let bits = reference_bits(&reference, 256).expect("bits");
    let runtime = VideoRuntime {
        neural: NeuralBackends {
            messages: Some(Arc::new(FakeModels(bits))),
            ..NeuralBackends::default()
        },
        ..VideoRuntime::classical()
    };
    let report = identify_video_message(&runtime, &clip, &reference, MessageScheme::VideoSeal, true)
        .expect("identify");
    let want = &doc["cases"]["identify"];
    assert_eq!(report.status, "matched");
    assert_eq!(report.sha256, want["sha256"].as_str().expect("sha"));
    assert_eq!(
        serde_json::to_value(&report.caveats).expect("json"),
        want["caveats"]
    );
    let reading = report.reading.as_ref().expect("reading");
    assert_eq!(reading.frames, 24);
    assert_eq!(reading.windows.len(), 13);
    assert_eq!(reading.max_bit_accuracy, 1.0);
    assert_eq!(
        cli::message_identify_output(&report, false).lines[0],
        "  Public videoseal known message: matched"
    );

    let presence =
        detect_video_watermark_presence(&runtime, &clip, MessageScheme::VideoSeal).expect("detect");
    assert_eq!(presence.status, "detected");
    assert_eq!(presence.frames, 24);
    assert_eq!(presence.decoded_hex.as_deref(), Some(reference.as_str()));
    assert_eq!(presence.bit_sign_consistency, Some(1.0));
    assert_eq!(
        serde_json::to_value(&presence.caveats).expect("json"),
        doc["cases"]["detect"]["caveats"]
    );

    let plain = VideoRuntime::classical();
    let unverified = identify_video_message(&plain, &clip, &reference, MessageScheme::VideoSeal, true)
        .expect("unverified");
    assert_eq!(unverified.status, "unverified");
    assert!(
        unverified
            .caveats
            .last()
            .is_some_and(|c| c.contains("video-detect"))
    );
    let blind = detect_video_watermark_presence(&plain, &clip, MessageScheme::ChunkySeal).expect("blind");
    assert_eq!(blind.status, "unverified");
    assert!(blind.caveats[0].contains("1024-bit"));
    assert!(identify_video_message(&plain, &clip, "00", MessageScheme::VideoSeal, true).is_err());
}

#[test]
fn removal_fails_closed_when_every_candidate_still_matches() {
    common::ffmpeg_threads();
    let dir = tempfile::tempdir().expect("dir");
    let (clip, reference) = golden_clip(dir.path());
    let bits = reference_bits(&reference, 256).expect("bits");
    let runtime = VideoRuntime {
        neural: NeuralBackends {
            messages: Some(Arc::new(FakeModels(bits))),
            ..NeuralBackends::default()
        },
        ..VideoRuntime::classical()
    };
    let output = dir.path().join("out.mp4");
    let options = MessageRemovalOptions {
        reference_hex: reference.clone(),
        method: "codec".into(),
        long_side: 128,
        ..MessageRemovalOptions::default()
    };
    let error = remove_video_message(&runtime, &clip, &output, &options).unwrap_err();
    assert!(
        error
            .to_string()
            .starts_with("Saved video message check is matched"),
        "{error}"
    );
    assert!(!output.exists());
    let json = cli::verification_error_json(&error);
    let parsed: Value = serde_json::from_str(&json.lines[0]).expect("json");
    assert_eq!(parsed["after"]["status"], "matched");

    let forced = remove_video_message(
        &runtime,
        &clip,
        &output,
        &MessageRemovalOptions {
            force: true,
            ..options.clone()
        },
    )
    .expect("forced");
    assert_eq!(forced.verified_removed, Some(false));
    assert_eq!(forced.copied_audio_streams, vec![1]);
    assert_eq!(forced.method.as_deref(), Some("codec"));
    let quality = forced.quality.as_ref().expect("quality");
    assert_eq!(quality["frames"], 12);
    assert_eq!(quality["file"], "out.mp4");
    assert_eq!(cli::message_remove_output(&forced, false).exit_code, 1);

    let skipped = remove_video_message(
        &VideoRuntime::classical(),
        &clip,
        &output,
        &MessageRemovalOptions {
            verify: true,
            ..options.clone()
        },
    )
    .expect("skipped");
    assert!(skipped.skipped);
    assert_eq!(cli::message_remove_output(&skipped, false).exit_code, 2);
    assert!(remove_video_message(&runtime, &clip, &clip, &options).is_err());
    assert!(
        remove_video_message(
            &runtime,
            &clip,
            &output,
            &MessageRemovalOptions {
                verify: false,
                ..options
            }
        )
        .is_err()
    );
}
