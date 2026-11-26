//! Phases 3-5 behavior: visible removal against Python's pre-encode frames, region erase with
//! frame ranges and tracking, VFR timing, clean-input byte copies, batch, audio streams, and the
//! invisible orchestration with a fake VAE.

mod common;

use std::path::Path;
use std::sync::Arc;

use common::{clip, sha256_hex};
use raiw_core::testing::{fixture, golden};
use raiw_cv::Mat;
use raiw_metadata::get_ai_metadata;
use raiw_video::api::{AllVideoOptions, BatchVideoOptions, EraseVideoOptions, VisibleVideoOptions};
use raiw_video::invisible::{
    InvisibleOptions, Latents, NeuralBackends, VideoVae, regenerate_video_candidate,
};
use raiw_video::probe::{count_decoded_video_frames, probe_video_timestamps};
use raiw_video::provenance::{VIDEO_VISIBLE_MARKS, non_contradicted_marks};
use raiw_video::visible::{
    CleanOptions, Region, for_each_clean_frame, merge_selected_visible_marks, scan_video_marks,
    select_stable_visible_marks,
};
use raiw_video::{
    BatchMode, InvisibleMode, VideoAudioOptions, VideoRuntime, cli, remove_video_all, remove_video_audio,
    remove_video_audio_all_streams, remove_video_batch, remove_video_invisible, remove_video_region,
    remove_video_visible,
};
use raiw_visible::eraser::{Backend, FillBackends};
use serde_json::Value;

#[test]
fn cleaned_frames_track_python_before_encoding() {
    common::ffmpeg_threads();
    let golden = golden("video_removal", "removal");
    let fills = FillBackends::classical();
    for (key, want) in golden["visible"].as_object().expect("visible") {
        let path = fixture(format!("visible/{key}/example.mp4"));
        let markers = get_ai_metadata(&path);
        let active = non_contradicted_marks(&VIDEO_VISIBLE_MARKS, &markers);
        let scans = scan_video_marks(&path, &active, true).expect("scan");
        let total = scans[0].1.detections.len();
        let selected = select_stable_visible_marks(&scans, &markers, &VIDEO_VISIBLE_MARKS, false);
        let merged = merge_selected_visible_marks(&selected, total).expect("a stable mark");
        assert_eq!(
            merged.primary_mark,
            want["primary"].as_str().expect("primary"),
            "{key}"
        );
        assert_eq!(
            merged.detected_frames as u64,
            want["detected"].as_u64().expect("detected"),
            "{key}"
        );
        let options = CleanOptions {
            backend: Backend::Cv2,
            padding_fraction: merged.padding_fraction,
            mask_style: merged.mask_style,
            ..CleanOptions::default()
        };
        let frames = want["frames"].as_array().expect("frames");
        let (mut exact, mut index, mut worst) = (0usize, 0usize, 0.0f64);
        for_each_clean_frame::<u8>(
            &path,
            &merged.scan,
            &merged.regions,
            &fills,
            &options,
            |frame, removed| {
                let expected = &frames[index];
                assert_eq!(
                    Some(removed),
                    expected["removed"].as_bool(),
                    "{key} frame {index} removed flag"
                );
                if Some(sha256_hex(frame.as_bytes()).as_str()) == expected["sha256"].as_str() {
                    exact += 1;
                }
                let pixels = (frame.width() * frame.height()) as f64;
                for c in 0..3 {
                    let sum: u64 = frame
                        .data()
                        .iter()
                        .skip(c)
                        .step_by(3)
                        .map(|v| u64::from(*v))
                        .sum();
                    let diff = (sum as f64 - expected["sum"][c].as_f64().expect("sum")).abs() / pixels;
                    worst = worst.max(diff);
                }
                index += 1;
                Ok(())
            },
        )
        .expect("clean");
        assert_eq!(index, frames.len(), "{key} frame count");
        eprintln!(
            "{key}: {exact}/{} frames byte-identical, worst mean channel delta {worst:.5}",
            frames.len()
        );
        assert!(
            worst < 0.05,
            "{key}: cleaned frames drifted from Python by {worst}"
        );
    }
}

#[test]
fn visible_removal_publishes_a_clean_full_clip() {
    common::ffmpeg_threads();
    let runtime = VideoRuntime::discover();
    let dir = tempfile::tempdir().expect("dir");
    let want = &golden("video_removal", "removal")["visible"]["sora"]["result"];
    let source = fixture("visible/sora/example.mp4");
    let output = dir.path().join("sora_clean.mp4");
    let result = remove_video_visible(&runtime, &source, Some(&output), &VisibleVideoOptions::default())
        .expect("remove");
    assert_eq!(result.mark, want["mark"].as_str().expect("mark"));
    assert_eq!(
        result.total_frames as u64,
        want["total_frames"].as_u64().expect("total")
    );
    assert_eq!(
        result.detected_frames as u64,
        want["detected_frames"].as_u64().expect("detected")
    );
    assert_eq!(
        result.removed_frames as u64,
        want["removed_frames"].as_u64().expect("removed")
    );
    assert!(result.remaining_metadata.is_empty());
    assert_eq!(
        count_decoded_video_frames(&output).expect("count"),
        Some(result.total_frames as u64)
    );
    let rescan = scan_video_marks(&output, &["sora"], false).expect("rescan");
    let still = raiw_video::visible::stabilize_localizations("sora", &rescan[0].1.detections, false);
    assert!(
        still.iter().all(Option::is_none),
        "the Sora mark survived removal"
    );
    let text = cli::visible_output(&result, "auto", false);
    assert_eq!(text.exit_code, 0);
}

fn erase_options(regions: &[(i32, i32, i32, i32)]) -> EraseVideoOptions {
    EraseVideoOptions {
        regions: regions.iter().map(|r| Region::new(r.0, r.1, r.2, r.3)).collect(),
        frame_range: None,
        backend: Backend::Cv2,
        dilate: 3,
        strip_metadata: true,
        temporal_consistency: true,
        track: false,
    }
}

#[test]
fn region_erase_matches_python_results() {
    common::ffmpeg_threads();
    let runtime = VideoRuntime::discover();
    let dir = tempfile::tempdir().expect("dir");
    for (n, case) in golden("video_removal", "removal")["erase"]
        .as_array()
        .expect("erase")
        .iter()
        .enumerate()
    {
        let name = case["clip"].as_str().expect("clip");
        let source = clip(name);
        let kwargs = &case["kwargs"];
        let regions: Vec<(i32, i32, i32, i32)> = kwargs["regions"]
            .as_array()
            .expect("regions")
            .iter()
            .map(|r| {
                (
                    r[0].as_i64().expect("x") as i32,
                    r[1].as_i64().expect("y") as i32,
                    r[2].as_i64().expect("w") as i32,
                    r[3].as_i64().expect("h") as i32,
                )
            })
            .collect();
        let mut options = erase_options(&regions);
        options.track = kwargs.get("track").and_then(Value::as_bool).unwrap_or(false);
        options.frame_range = kwargs.get("frame_range").and_then(Value::as_array).map(|r| {
            (
                r[0].as_u64().expect("s") as usize,
                r[1].as_u64().expect("e") as usize,
            )
        });
        let output = dir.path().join(format!(
            "erase{n}{}",
            Path::new(source)
                .extension()
                .map(|e| format!(".{}", e.to_string_lossy()))
                .unwrap_or_default()
        ));
        let result = remove_video_region(&runtime, source, Some(&output), &options).expect("erase");
        let want = &case["result"];
        assert_eq!(
            result.total_frames as u64,
            want["total_frames"].as_u64().expect("total"),
            "case {n}"
        );
        assert_eq!(
            result.erased_frames as u64,
            want["erased_frames"].as_u64().expect("erased"),
            "case {n}"
        );
        assert_eq!(
            serde_json::to_value(result.frame_range).expect("json"),
            want["frame_range"],
            "case {n}"
        );
        if !options.track {
            assert_eq!(
                serde_json::to_value(&result.regions).expect("json"),
                want["regions"],
                "case {n}"
            );
        }
        assert_eq!(
            probe_video_timestamps(&output).expect("ts"),
            probe_video_timestamps(source).expect("ts"),
            "case {n} timing"
        );
    }
}

#[test]
fn clean_inputs_are_byte_copied_by_all() {
    common::ffmpeg_threads();
    let runtime = VideoRuntime::discover();
    let dir = tempfile::tempdir().expect("dir");
    let flat = dir.path().join("flat.mp4");
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-threads",
            "4",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x336699:s=160x96:d=1:r=24",
        ])
        .args([
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=1",
            "-shortest",
        ])
        .args([
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac",
        ])
        .arg(&flat)
        .status()
        .expect("ffmpeg");
    assert!(status.success());
    let source = flat.as_path();
    let output = dir.path().join("all.mp4");
    let result = remove_video_all(&runtime, source, Some(&output), &AllVideoOptions::default()).expect("all");
    assert!(result.visible_mark.is_none());
    assert!(!result.invisible_removed);
    assert_eq!(
        std::fs::read(&output).expect("out"),
        std::fs::read(source).expect("src")
    );
    assert_eq!(
        cli::all_output(&result, "auto", false).lines[0],
        "  Visible mark: none found; pixels preserved"
    );
}

#[test]
fn batch_processes_every_supported_file() {
    common::ffmpeg_threads();
    let runtime = VideoRuntime::discover();
    let dir = tempfile::tempdir().expect("dir");
    let input = dir.path().join("in");
    std::fs::create_dir(&input).expect("mkdir");
    std::fs::copy(clip("cfr_audio"), input.join("b.mp4")).expect("copy");
    std::fs::copy(clip("odd_size"), input.join("A.mp4")).expect("copy");
    std::fs::write(input.join("notes.txt"), b"skip").expect("write");
    std::fs::write(input.join("broken.mp4"), b"not a video").expect("write");
    let options = BatchVideoOptions {
        mode: BatchMode::Metadata,
        all: AllVideoOptions::default(),
    };
    let result = remove_video_batch(&runtime, &input, None, &options).expect("batch");
    let names: Vec<String> = result
        .items
        .iter()
        .map(|i| i.source.file_name().expect("name").to_string_lossy().into_owned())
        .collect();
    assert_eq!(names, ["A.mp4", "b.mp4", "broken.mp4"]);
    assert_eq!(result.failed(), 1);
    assert_eq!(result.processed(), 2);
    assert!(result.output_directory.ends_with("in_clean"));
    assert_eq!(cli::batch_output(&result, &input, false, false).exit_code, 1);
    let empty = dir.path().join("empty");
    std::fs::create_dir(&empty).expect("mkdir");
    let none = remove_video_batch(&runtime, &empty, None, &options).expect("batch");
    assert_eq!(
        cli::batch_output(&none, &empty, false, true).exit_code,
        cli::EXIT_NO_SIGNAL
    );
}

fn spectral_unverified() -> VideoAudioOptions {
    VideoAudioOptions {
        method: raiw_audio::Method::Spectral,
        scheme: raiw_audio::Scheme::Auto,
        allow_unverified: true,
        verify: false,
        ..VideoAudioOptions::default()
    }
}

#[test]
fn audio_track_processing_preserves_video_packets() {
    common::ffmpeg_threads();
    let runtime = VideoRuntime::discover();
    let dir = tempfile::tempdir().expect("dir");
    let output = dir.path().join("audio.mkv");
    let source = dir.path().join("source.mkv");
    let status = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-i"])
        .arg(clip("cfr_audio"))
        .args(["-c", "copy"])
        .arg(&source)
        .status()
        .expect("ffmpeg");
    assert!(status.success());
    let result =
        remove_video_audio(&runtime.audio, &source, &output, 0, &spectral_unverified(), None).expect("audio");
    assert!(!result.skipped);
    assert_eq!(result.unchanged_streams, vec![0]);
    assert_eq!(result.method, "spectral");
    let multi_out = dir.path().join("multi.mp4");
    let multi = remove_video_audio_all_streams(
        &runtime.audio,
        clip("two_audio"),
        &multi_out,
        &spectral_unverified(),
        None,
    )
    .expect("multi");
    assert_eq!(multi.processed_ordinals, vec![0, 1]);
    assert_eq!(multi.unchanged_streams, vec![0]);
    assert_eq!(cli::audio_multi_output(&multi, "auto", false).exit_code, 0);
}

#[test]
fn outro_bumper_cut_matches_python() {
    common::ffmpeg_threads();
    let dir = tempfile::tempdir().expect("dir");
    let clip_path = dir.path().join("bumper.mp4");
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-threads",
            "4",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=duration=3:rate=24:size=160x96",
        ])
        .args([
            "-f",
            "lavfi",
            "-i",
            "color=c=0x202020:duration=1:rate=24:size=160x96",
        ])
        .args([
            "-filter_complex",
            "[0][1]concat=n=2:v=1[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&clip_path)
        .status()
        .expect("ffmpeg");
    assert!(status.success());
    let frames = raiw_video::decode::iter_decoded_bgr_frames(&clip_path, None).expect("decode");
    assert_eq!(
        raiw_video::temporal::detect_ai_bumper_start(frames, 24.0).expect("bumper"),
        Some(72)
    );
    let plain = raiw_video::decode::iter_decoded_bgr_frames(clip("cfr_audio"), None).expect("decode");
    assert_eq!(
        raiw_video::temporal::detect_ai_bumper_start(plain, 24.0).expect("bumper"),
        None
    );
}

#[derive(Debug)]
struct FakeVae;

impl VideoVae for FakeVae {
    fn model(&self) -> &str {
        raiw_video::invisible::DEFAULT_VIDEO_SYNTHID_VAE
    }
    fn requested_device(&self) -> &'static str {
        "auto"
    }
    fn scaling_factor(&self) -> f64 {
        raiw_video::invisible::VIDEO_SYNTHID_VAE_SCALING_FACTOR
    }
    fn encode(&self, rgb: &[Mat<f32>]) -> raiw_video::Result<Latents> {
        let (w, h) = rgb[0].size();
        let mut data = Vec::new();
        for frame in rgb {
            data.extend_from_slice(frame.data());
        }
        Ok(Latents {
            shape: [rgb.len(), 3, h, w],
            data,
        })
    }
    fn decode(&self, latents: &Latents) -> raiw_video::Result<Vec<Mat<f32>>> {
        let [n, c, h, w] = latents.shape;
        Ok((0..n)
            .map(|i| {
                Mat::from_vec(w, h, c, latents.data[i * c * h * w..(i + 1) * c * h * w].to_vec())
                    .expect("plane")
            })
            .collect())
    }
    fn shared_noise(&self, shape: [usize; 3], _seed: u64) -> raiw_video::Result<Vec<f32>> {
        Ok(vec![0.0; shape.iter().product()])
    }
}

#[test]
fn invisible_orchestration_runs_on_a_fake_vae() {
    common::ffmpeg_threads();
    let dir = tempfile::tempdir().expect("dir");
    let output = dir.path().join("regen.mp4");
    let metrics = regenerate_video_candidate(
        &FakeVae,
        clip("cfr_audio"),
        &output,
        &InvisibleOptions::default(),
        None,
        None,
    )
    .expect("regen");
    assert_eq!((metrics.width, metrics.height), (512, 304));
    assert_eq!(metrics.fps, 12.0);
    assert_eq!(metrics.frames, 24);
    assert!(
        metrics.psnr_db > 30.0,
        "identity VAE should keep fidelity: {}",
        metrics.psnr_db
    );
    assert_eq!(count_decoded_video_frames(&output).expect("count"), Some(24));

    let runtime = VideoRuntime::classical();
    let unavailable =
        remove_video_invisible(&runtime, clip("cfr_audio"), None, &InvisibleOptions::default()).unwrap_err();
    assert!(
        matches!(unavailable, raiw_video::VideoError::Unavailable(_)),
        "{unavailable}"
    );
    let with_vae = VideoRuntime {
        neural: NeuralBackends {
            vae: Some(Arc::new(FakeVae)),
            ..NeuralBackends::default()
        },
        ..runtime
    };
    let result = remove_video_invisible(
        &with_vae,
        clip("cfr_audio"),
        Some(&dir.path().join("inv.mp4")),
        &InvisibleOptions::default(),
    )
    .expect("invisible");
    assert_eq!(result.profile, "synthid");
    assert_eq!(cli::invisible_output(&result, false).exit_code, 0);
    let auto = AllVideoOptions {
        include_invisible: InvisibleMode::On,
        ..AllVideoOptions::default()
    };
    // testsrc2 carries a stable jimeng_pill lookalike (Python agrees), so auto routes to Seedance.
    let seedance = remove_video_all(
        &with_vae,
        clip("cfr_audio"),
        Some(&dir.path().join("auto.mp4")),
        &auto,
    )
    .expect("seedance");
    assert!(seedance.invisible_removed);
    assert_eq!(
        count_decoded_video_frames(&dir.path().join("auto.mp4")).expect("count"),
        Some(48)
    );
    let synthid = AllVideoOptions {
        include_invisible: InvisibleMode::On,
        invisible: InvisibleOptions {
            profile: "synthid".into(),
            ..InvisibleOptions::default()
        },
        ..AllVideoOptions::default()
    };
    let combined = remove_video_all(
        &with_vae,
        clip("cfr_audio"),
        Some(&dir.path().join("all.mp4")),
        &synthid,
    )
    .expect("all");
    assert!(combined.invisible_removed);
    assert_eq!(combined.visual_invisible_action(), "regenerated");
    assert_eq!(combined.visible_mark.as_deref(), Some("jimeng_pill"));
}
