//! Wan 2.1 window streaming and eligibility (`video_wan.py`) with a fake pipeline, and the
//! guide/depth paths of the 2D VAE orchestration with a fake VAE.

mod common;

use std::sync::Mutex;

use common::clip;
use raiw_cv::Mat;
use raiw_video::probe::{count_decoded_video_frames, probe_video_timestamps};
use raiw_video::wan::{
    WAN21_MODEL_ID, WAN21_MODEL_REVISION, WanCall, WanPipeline, WanWindowing, WanWindows,
    can_run_wan_on_source, effective_chunk_frames, effective_overlap_frames, wan_guide_size,
};
use raiw_video::{SeedanceQuality, WanOptions, regenerate_wan_experimental};

/// `(frames, height, width, seed, steps)` of one pipeline call.
type Call = (usize, usize, usize, u64, u32);

/// Adds `seed / 100` to every value so windows and their blend stay distinguishable.
#[derive(Debug, Default)]
struct FakeWan {
    model: &'static str,
    small_vram: bool,
    calls: Mutex<Vec<Call>>,
}

impl FakeWan {
    fn pinned() -> Self {
        Self {
            model: WAN21_MODEL_ID,
            ..Self::default()
        }
    }
}

impl WanPipeline for FakeWan {
    fn model(&self) -> &str {
        self.model
    }
    fn revision(&self) -> &str {
        WAN21_MODEL_REVISION
    }
    fn resolve_device(&self, requested: &str) -> raiw_video::Result<String> {
        match requested {
            "auto" | "cuda" => Ok("cuda".into()),
            "mps" => Ok("mps".into()),
            _ => Err(raiw_video::VideoError::value(
                "Wan 2.1 3D video diffusion requires device='auto', 'cuda', or 'mps'",
            )),
        }
    }
    fn vram_below(&self, _limit: u64) -> bool {
        self.small_vram
    }
    fn generate(&self, frames: &[Mat<f32>], call: &WanCall<'_>) -> raiw_video::Result<Vec<Mat<f32>>> {
        assert_eq!((frames.len() - 1) % 4, 0, "windows are padded to 1 + 4k frames");
        assert_eq!(call.height % 16, 0);
        assert_eq!(call.width % 16, 0);
        self.calls.lock().expect("calls").push((
            frames.len(),
            call.height,
            call.width,
            call.seed,
            call.steps,
        ));
        let shift = call.seed as f32 / 100.0;
        Ok(frames.iter().map(|f| f.map(|v| (v + shift).min(1.0))).collect())
    }
}

fn gray_frames(count: usize, width: usize, height: usize) -> Vec<Mat<u8>> {
    (0..count)
        .map(|i| Mat::from_vec(width, height, 3, vec![(40 + i) as u8; width * height * 3]).expect("frame"))
        .collect()
}

#[test]
fn windows_overlap_with_a_raised_cosine_blend() {
    let wan = FakeWan::pinned();
    let input = gray_frames(30, 20, 10);
    let windowing = WanWindowing {
        chunk_frames: 17,
        overlap_frames: 5,
    };
    let stream = WanWindows::new(
        input.clone().into_iter().map(Ok),
        &wan,
        "cuda",
        0.38,
        7,
        3,
        windowing,
    )
    .expect("windows");
    let output: Vec<Mat<u8>> = stream.collect::<raiw_video::Result<_>>().expect("stream");
    assert_eq!(output.len(), input.len());
    let calls = wan.calls.lock().expect("calls").clone();
    // 17 frames, then 12 new + 5 overlap = 17, then the last 1 + 5 overlap padded to 9.
    assert_eq!(
        calls,
        vec![(17, 16, 32, 7, 3), (17, 16, 32, 8, 3), (9, 16, 32, 9, 3)]
    );
    let shifted = |frame: usize, seed: f32| ((40 + frame) as f32 / 255.0 + seed / 100.0) * 255.0;
    for (i, frame) in output.iter().enumerate() {
        let value = f32::from(frame.data()[0]);
        let expected = if i < 12 {
            shifted(i, 7.0).round_ties_even()
        } else if i < 17 {
            let fade = 0.5 - 0.5 * (std::f32::consts::PI * (i - 11) as f32 / 6.0).cos();
            let previous = shifted(i, 7.0).round_ties_even();
            let current = shifted(i, 8.0).round_ties_even();
            (previous * (1.0 - fade) + current * fade).round_ties_even()
        } else if i < 24 {
            shifted(i, 8.0).round_ties_even()
        } else if i < 29 {
            let fade = 0.5 - 0.5 * (std::f32::consts::PI * (i - 23) as f32 / 6.0).cos();
            let previous = shifted(i, 8.0).round_ties_even();
            let current = shifted(i, 9.0).round_ties_even();
            (previous * (1.0 - fade) + current * fade).round_ties_even()
        } else {
            shifted(i, 9.0).round_ties_even()
        };
        assert_eq!(value, expected, "frame {i}");
        assert!(
            frame.data().iter().all(|v| f32::from(*v) == value),
            "frame {i} is uniform"
        );
    }
    for (chunk_frames, overlap_frames) in [(18, 1), (13, 1), (17, 17)] {
        let none = std::iter::empty::<raiw_video::Result<Mat<u8>>>();
        let windowing = WanWindowing {
            chunk_frames,
            overlap_frames,
        };
        assert!(WanWindows::new(none, &wan, "cuda", 0.38, 0, 1, windowing).is_err());
    }
}

#[test]
fn window_sizes_and_eligibility_follow_python() {
    let wan = FakeWan::pinned();
    assert_eq!(wan_guide_size(1920, 1080, 16), None);
    assert_eq!(wan_guide_size(3840, 2160, 16), Some((1920, 1072)));
    assert_eq!(wan_guide_size(3840, 2160, 32), Some((1920, 1056)));
    assert_eq!(effective_chunk_frames(193, 17, (1280, 720), "cuda", &wan), 193);
    assert_eq!(effective_chunk_frames(193, 17, (1920, 1080), "cuda", &wan), 81);
    assert_eq!(effective_chunk_frames(193, 17, (640, 360), "mps", &wan), 33);
    assert_eq!(effective_chunk_frames(97, 17, (1920, 1080), "cuda", &wan), 97);
    assert_eq!(effective_overlap_frames(17, 33, "mps"), 9);
    assert_eq!(effective_overlap_frames(17, 81, "cuda"), 17);
    let small = FakeWan {
        small_vram: true,
        ..FakeWan::pinned()
    };
    assert_eq!(effective_chunk_frames(193, 17, (640, 360), "cuda", &small), 81);

    let source = clip("cfr_audio");
    assert!(can_run_wan_on_source(source, "auto", Some(&wan), true));
    assert!(can_run_wan_on_source(source, "cuda", Some(&wan), false));
    assert!(
        !can_run_wan_on_source(source, "mps", Some(&wan), true),
        "auto never routes to MPS"
    );
    assert!(can_run_wan_on_source(source, "mps", Some(&wan), false));
    assert!(!can_run_wan_on_source(source, "cpu", Some(&wan), false));
    assert!(!can_run_wan_on_source(source, "auto", None, false));
    assert!(
        !can_run_wan_on_source(source, "auto", Some(&small), true),
        "auto needs 8 GiB"
    );
    assert!(can_run_wan_on_source(source, "cuda", Some(&small), false));
    let foreign = FakeWan {
        model: "other/model",
        ..FakeWan::default()
    };
    assert!(!can_run_wan_on_source(source, "auto", Some(&foreign), false));
}

/// Identity VAE: latents are the RGB planes themselves.
#[derive(Debug)]
struct IdentityVae;

impl raiw_video::VideoVae for IdentityVae {
    fn model(&self) -> &str {
        raiw_video::invisible::DEFAULT_VIDEO_SYNTHID_VAE
    }
    fn requested_device(&self) -> &'static str {
        "auto"
    }
    fn scaling_factor(&self) -> f64 {
        raiw_video::invisible::VIDEO_SYNTHID_VAE_SCALING_FACTOR
    }
    fn encode(&self, rgb: &[Mat<f32>]) -> raiw_video::Result<raiw_video::invisible::Latents> {
        let (w, h) = rgb[0].size();
        Ok(raiw_video::invisible::Latents {
            shape: [rgb.len(), 3, h, w],
            data: rgb.iter().flat_map(|f| f.data().iter().copied()).collect(),
        })
    }
    fn decode(&self, latents: &raiw_video::invisible::Latents) -> raiw_video::Result<Vec<Mat<f32>>> {
        let [_, c, h, w] = latents.shape;
        Ok(latents
            .data
            .chunks_exact(c * h * w)
            .map(|plane| Mat::from_vec(w, h, c, plane.to_vec()).expect("plane"))
            .collect())
    }
    fn shared_noise(&self, shape: [usize; 3], _seed: u64) -> raiw_video::Result<Vec<f32>> {
        Ok(vec![0.0; shape.iter().product()])
    }
}

#[test]
fn native_guides_and_high_depth_regenerate_at_source_geometry() {
    common::ffmpeg_threads();
    let dir = tempfile::tempdir().expect("dir");
    let guided = raiw_video::invisible::InvisibleOptions {
        profile: "synthid".into(),
        long_side: 64,
        native_guide: true,
        ..raiw_video::invisible::InvisibleOptions::default()
    };
    let output = dir.path().join("guided.mp4");
    let metrics = raiw_video::invisible::regenerate_video_candidate(
        &IdentityVae,
        clip("cfr_audio"),
        &output,
        &guided,
        None,
        None,
    )
    .expect("guided");
    assert_eq!((metrics.width, metrics.height, metrics.frames), (160, 96, 24));
    assert!(metrics.psnr_db > 18.0, "{}", metrics.psnr_db);

    let deep = raiw_video::invisible::InvisibleOptions {
        profile: "synthid".into(),
        long_side: 64,
        ..raiw_video::invisible::InvisibleOptions::default()
    };
    let output = dir.path().join("deep.mp4");
    let metrics = raiw_video::invisible::regenerate_video_candidate(
        &IdentityVae,
        clip("hevc_10bit"),
        &output,
        &deep,
        None,
        None,
    )
    .expect("10-bit");
    assert_eq!((metrics.width, metrics.height, metrics.frames), (64, 32, 12));
    assert!(
        metrics.psnr_db > 30.0,
        "16-bit PSNR uses the 65535 peak: {}",
        metrics.psnr_db
    );
    let profile = raiw_video::probe::probe_video_encode_profile(&output).expect("profile");
    assert_eq!(profile.pixel_format, "yuv420p10le");
}

#[test]
fn wan_regeneration_keeps_every_timestamp() {
    common::ffmpeg_threads();
    let wan = FakeWan::pinned();
    let dir = tempfile::tempdir().expect("dir");
    let output = dir.path().join("wan.mp4");
    let source = clip("cfr_audio");
    let options = WanOptions {
        quality: SeedanceQuality::Delivery,
        windowing: WanWindowing {
            chunk_frames: 21,
            overlap_frames: 5,
        },
        inference_steps: Some(2),
        ..WanOptions::default()
    };
    let metrics = regenerate_wan_experimental::<u8>(&wan, source, &output, &options, None).expect("wan");
    assert_eq!(metrics.frames, 48);
    assert_eq!((metrics.width, metrics.height), (160, 96));
    assert!(metrics.psnr_db > 25.0, "{}", metrics.psnr_db);
    assert_eq!(count_decoded_video_frames(&output).expect("count"), Some(48));
    assert_eq!(
        probe_video_timestamps(&output).expect("ts"),
        probe_video_timestamps(source).expect("ts")
    );
    let seeds: Vec<u64> = wan.calls.lock().expect("calls").iter().map(|c| c.3).collect();
    assert_eq!(seeds, vec![0, 1, 2]);
    let cpu = WanOptions {
        device: "cpu".into(),
        ..options
    };
    assert!(regenerate_wan_experimental::<u8>(&wan, source, &output, &cpu, None).is_err());
}
