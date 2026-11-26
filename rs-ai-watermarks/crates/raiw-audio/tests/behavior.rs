//! Behavioral assertions ported from `tests/test_audio_resampling.py`, `test_audio_spectral.py`,
//! `test_audio_codec.py`, `test_audio_dac.py` and `test_audio.py`.

mod common;
mod fakes;

use std::f64::consts::PI;
use std::path::Path;
use std::sync::{Arc, Mutex};

use common::{encode_raw, splitmix_uniform};
use raiw_audio::codecs::{
    CodecError, CodecKind, CodecRequest, CodecSet, NeuralCodec, ResolvedDevice, StereoMode,
};
use raiw_audio::codecs::{encodec, raw};
use raiw_audio::dsp::window::firwin_kaiser;
use raiw_audio::metadata::MetadataSeam;
use raiw_audio::options::{Method, Scheme};
use raiw_audio::report::Markers;
use raiw_audio::{
    AllOptions, AudioError, AudioRuntime, BatchMode, Frames, IdentifyOptions, RationalClock, RemovalRequest,
    spectral,
};
use raiw_audio::{
    Permissions, ProcessingArgs, identify_audio, remove_audio_all, remove_audio_batch, remove_audio_invisible,
};

fn noise(seed: u64, frames: usize, channels: usize) -> Frames {
    let values: Vec<f32> = splitmix_uniform(seed, frames * channels)
        .into_iter()
        .map(|u| (2.0 * u - 1.0) as f32)
        .collect();
    Frames::new(values, channels).unwrap()
}

fn roundtrip(value: &Frames, rate: u32) -> Frames {
    let clock = RationalClock::standard(value.frames(), rate).unwrap();
    let model = clock.forward(value).unwrap();
    clock
        .inverse(&clock.unpad_model(&clock.pad_model(&model).unwrap()).unwrap())
        .unwrap()
}

#[test]
fn clock_geometry_holds_for_every_residue() {
    for frames in (1..321).chain(480_000..480_160) {
        let clock = RationalClock::standard(frames, 48_000).unwrap();
        assert_eq!(clock.model_frames(), (frames * 147).div_ceil(160));
        assert_eq!(clock.inverse_frames(), (clock.model_frames() * 160).div_ceil(147));
        // (model_frames - 1) / 44100 < frames / 48000 <= model_frames / 44100
        assert!((clock.model_frames() - 1) * 48_000 < frames * 44_100);
        assert!(frames * 44_100 <= clock.model_frames() * 48_000);
        // Excluded inverse instants k / 48000 lie at or after the duration T.
        assert!((frames..clock.inverse_frames()).all(|k| k * 44_100 >= frames * 44_100));
        assert!(clock.inverse_frames() >= frames);
        if frames <= 320 {
            let value = Frames::mono((0..frames).map(|i| i as f32 / frames as f32).collect());
            assert_eq!(clock.forward(&value).unwrap().frames(), clock.model_frames());
            assert_eq!(roundtrip(&value, 48_000).frames(), frames);
        }
    }
}

fn direct_fir(value: &[f32], up: usize, down: usize) -> Vec<f64> {
    let maximum = up.max(down);
    let half = 10 * maximum;
    let kernel: Vec<f64> = firwin_kaiser(2 * half + 1, 1.0 / maximum as f64, 5.0)
        .into_iter()
        .map(|v| f64::from(v as f32 * up as f32))
        .collect();
    let len = (value.len() * up).div_ceil(down);
    (0..len)
        .map(|index| {
            (0..value.len())
                .filter_map(|j| {
                    let tap = (half + index * down) as isize - (j * up) as isize;
                    (tap >= 0 && (tap as usize) < kernel.len())
                        .then(|| f64::from(value[j]) * kernel[tap as usize])
                })
                .sum()
        })
        .collect()
}

#[test]
fn forward_and_inverse_match_a_direct_fir_reference() {
    for frames in [1usize, 2, 13, 159, 160, 161, 511, 557, 558, 559, 1024, 1200] {
        let mut positions = vec![0, frames - 1, (frames - 1).min(557)];
        positions.sort_unstable();
        positions.dedup();
        for position in positions {
            let mut value = vec![0.0f32; frames];
            value[position] = 1.0;
            let clock = RationalClock::standard(frames, 48_000).unwrap();
            let model = clock.forward(&Frames::mono(value.clone())).unwrap().into_vec();
            for (a, e) in model.iter().zip(direct_fir(&value, 147, 160)) {
                assert!((f64::from(*a) - e).abs() <= 2e-6 + 2e-6 * e.abs());
            }
            let result = clock.inverse(&Frames::mono(model.clone())).unwrap().into_vec();
            for (a, e) in result.iter().zip(direct_fir(&model, 160, 147)) {
                assert!((f64::from(*a) - e).abs() <= 2e-6 + 2e-6 * e.abs());
            }
            let peak = result
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                .unwrap()
                .0;
            assert!(peak.abs_diff(position) <= 1);
        }
    }
}

#[test]
fn last_source_sample_reaches_the_output_tail() {
    let mut value = vec![0.0f32; 160];
    value[159] = 1.0;
    let clock = RationalClock::standard(160, 48_000).unwrap();
    let model = clock.forward(&Frames::mono(value.clone())).unwrap().into_vec();
    assert!(
        model[model.len() - 16..]
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()))
            > 0.05
    );
    assert!(roundtrip(&Frames::mono(value), 48_000).as_slice()[159].abs() > 0.1);
}

#[test]
fn reflection_context_and_identity_unpad() {
    for frames in [1usize, 2, 557, 558, 559, 1114, 1115, 1116] {
        let clock = RationalClock::standard(frames, 48_000).unwrap();
        let value = noise(12, frames, 2);
        let model = clock.forward(&value).unwrap();
        let padded = clock.pad_model(&model).unwrap();
        assert!(padded.frames() >= 1024 && padded.frames().is_multiple_of(512));
        assert_eq!(clock.unpad_model(&padded).unwrap(), model);
        if clock.model_frames() == 1 {
            assert!(padded.rows().all(|row| row == model.row(0)));
        }
        let short = padded.slice(0..padded.frames() - 1);
        assert!(
            clock
                .unpad_model(&short)
                .unwrap_err()
                .to_string()
                .contains("frames")
        );
    }
}

#[test]
fn stereo_channels_stay_separate_and_ordered() {
    for (channel, position) in [(0usize, 100usize), (1, 1500)] {
        let mut value = Frames::zeros(1600, 2);
        value.as_mut_slice()[position * 2 + channel] = 1.0;
        let result = roundtrip(&value, 48_000);
        assert!(result.channel(1 - channel).iter().all(|&v| v == 0.0));
        let ch = result.channel(channel);
        let peak = ch.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        assert_eq!(peak, position);
    }
}

#[test]
fn inband_tones_survive_and_near_nyquist_is_filtered() {
    let tone = |parts: &[(f64, f64)]| -> Frames {
        Frames::mono(
            (0..48_000)
                .map(|n| {
                    let t = n as f64 / 48_000.0;
                    parts
                        .iter()
                        .map(|(f, a)| a * (2.0 * PI * f * t).sin())
                        .sum::<f64>() as f32
                })
                .collect(),
        )
    };
    let value = tone(&[(997.0, 0.3), (4003.0, 0.2), (11987.0, 0.1)]);
    let result = roundtrip(&value, 48_000);
    let interior = 256..48_000 - 256;
    let rms = |a: &[f32], b: &[f32]| {
        (a.iter()
            .zip(b)
            .map(|(x, y)| f64::from(x - y).powi(2))
            .sum::<f64>()
            / a.len() as f64)
            .sqrt()
    };
    let zeros = vec![0.0f32; interior.len()];
    assert!(
        rms(
            &result.as_slice()[interior.clone()],
            &value.as_slice()[interior.clone()]
        ) / 0.3
            < 0.02
    );
    let high = tone(&[(23_500.0, 1.0)]);
    let filtered = roundtrip(&high, 48_000);
    let ratio =
        rms(&filtered.as_slice()[interior.clone()], &zeros) / rms(&high.as_slice()[interior.clone()], &zeros);
    assert!(ratio < 0.25);
}

#[test]
fn other_native_rates_keep_exact_counts() {
    for rate in [8000u32, 16_000, 22_050, 44_100, 48_000, 96_000] {
        let value = noise(6, rate as usize / 20 + 1, 2);
        let result = roundtrip(&value, rate);
        assert_eq!((result.frames(), result.channels()), (value.frames(), 2));
        if rate == 44_100 {
            assert_eq!(result, value);
        }
    }
    assert!(
        RationalClock::standard(4, 0)
            .unwrap_err()
            .to_string()
            .contains("positive integer")
    );
    assert!(RationalClock::new(4, 48_000, 44_100, 0, 1024).is_err());
}

fn stereo_tones(rate: u32, seconds: f64) -> Frames {
    let frames = (f64::from(rate) * seconds) as usize;
    let t = |n: usize| n as f64 / f64::from(rate);
    let left = (0..frames)
        .map(|n| (0.2 * (2.0 * PI * 500.0 * t(n)).sin() + 0.1 * (2.0 * PI * 1250.0 * t(n)).sin()) as f32);
    let right = (0..frames)
        .map(|n| (0.2 * (2.0 * PI * 3000.0 * t(n)).sin() + 0.1 * (2.0 * PI * 1250.0 * t(n)).sin()) as f32);
    Frames::from_channels(&[left.collect(), right.collect()])
}

fn amplitude(x: &[f32], rate: f64, freq: f64) -> f64 {
    fakes::band_amplitude(x, rate, freq)
}

#[test]
fn research_band_is_suppressed_without_moving_other_tones() {
    let source = stereo_tones(48_000, 3.1);
    let output: Vec<Frames> = spectral::filter_chunks(std::iter::once(Ok(source.clone())), 48_000)
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    let output = Frames::concat(&output, 2);
    assert_eq!(output.frames(), source.frames());
    let section = output.slice(48_000..output.frames() - 48_000);
    for (channel, kept) in [(0, 500.0), (1, 3000.0)] {
        let ch = section.channel(channel);
        assert!(amplitude(&ch, 48_000.0, 1250.0) < 0.0001);
        assert!((amplitude(&ch, 48_000.0, kept) - 0.2).abs() < 0.002);
    }
}

#[test]
fn chunked_filter_matches_whole_clip_and_keeps_short_tail() {
    let source = stereo_tones(48_000, 3.0003);
    let run = |chunks: Vec<Frames>| -> Vec<Frames> {
        spectral::filter_chunks(chunks.into_iter().map(Ok), 48_000)
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap()
    };
    let whole = Frames::concat(&run(vec![source.clone()]), 2);
    let chunks: Vec<Frames> = (0..source.frames())
        .step_by(24_000)
        .map(|s| source.slice(s..s + 24_000))
        .collect();
    let delivered = run(chunks.clone());
    assert_eq!(
        delivered.iter().map(Frames::frames).collect::<Vec<_>>(),
        chunks.iter().map(Frames::frames).collect::<Vec<_>>()
    );
    let joined = Frames::concat(&delivered, 2);
    assert!(
        joined
            .as_slice()
            .iter()
            .zip(whole.as_slice())
            .all(|(a, b)| (a - b).abs() <= 1e-6)
    );
    let single = run(vec![Frames::new(vec![0.1, -0.1], 2).unwrap()]);
    assert_eq!(single[0].frames(), 1);
    assert!(single[0].is_finite());
}

#[test]
fn upstream_failure_propagates_through_the_filter() {
    let chunks = vec![
        Ok(Frames::zeros(10_000, 2)),
        Err(AudioError::failed("upstream read failed")),
    ];
    let result: Result<Vec<Frames>, _> = spectral::filter_chunks(chunks.into_iter(), 48_000)
        .unwrap()
        .collect();
    assert_eq!(result.unwrap_err().to_string(), "upstream read failed");
}

/// A codec that records stem lengths and applies `stem * gain`, padded with zeros.
struct Recording {
    kind: CodecKind,
    gain: f32,
    extra: usize,
    fault: Option<&'static str>,
    calls: Mutex<Vec<usize>>,
}

impl Recording {
    fn new(kind: CodecKind, gain: f32, extra: usize) -> Self {
        Self {
            kind,
            gain,
            extra,
            fault: None,
            calls: Mutex::new(Vec::new()),
        }
    }
}

impl NeuralCodec for Recording {
    fn kind(&self) -> CodecKind {
        self.kind
    }

    fn reconstruct(&self, stem: &[f32], _request: &CodecRequest) -> Result<Vec<f32>, CodecError> {
        self.calls.lock().unwrap().push(stem.len());
        let mut out: Vec<f32> = stem.iter().map(|v| v * self.gain).collect();
        match self.fault {
            Some("short") => {
                out.truncate(out.len().saturating_sub(8));
            }
            Some("nan") => out.fill(f32::NAN),
            _ => out.extend(std::iter::repeat_n(0.0, self.extra)),
        }
        Ok(out)
    }
}

const CPU: CodecRequest = CodecRequest {
    device: ResolvedDevice::Cpu,
    bandwidth_kbps: 24.0,
};

#[test]
fn encodec_keeps_channels_and_trims_only_padding() {
    let codec = Recording::new(CodecKind::Encodec, 0.5, 5);
    let samples = Frames::new((0..34).map(|v| v as f32 / 100.0).collect(), 2).unwrap();
    let delivered = encodec::resynthesize(&codec, &samples, 24_000, StereoMode::Independent, &CPU).unwrap();
    assert_eq!(delivered, samples.scale(0.5));
    for (fault, message) in [
        ("short", "EnCodec returned insufficient samples"),
        ("nan", "EnCodec returned nonfinite samples"),
    ] {
        let codec = Recording {
            fault: Some(fault),
            ..Recording::new(CodecKind::Encodec, 0.5, 0)
        };
        let err = encodec::resynthesize(
            &codec,
            &Frames::zeros(17, 2).map(|_| 1.0),
            24_000,
            StereoMode::Independent,
            &CPU,
        );
        assert!(err.unwrap_err().to_string().starts_with(message));
    }
    let bad = CodecRequest {
        bandwidth_kbps: 7.0,
        ..CPU
    };
    assert!(encodec::resynthesize(&codec, &samples, 24_000, StereoMode::Independent, &bad).is_err());
}

#[test]
fn encodec_streaming_uses_one_second_of_context() {
    let codec = Recording::new(CodecKind::Encodec, 0.5, 0);
    let samples = Frames::new((0..20 * 16_000 * 2 + 2).map(|v| v as f32).collect(), 2).unwrap();
    let chunks: Vec<_> = (0..samples.frames())
        .step_by(128_000)
        .map(|s| Ok(samples.slice(s..s + 128_000)))
        .collect();
    let out: Vec<Frames> =
        encodec::resynthesize_chunks(&codec, chunks.into_iter(), 16_000, StereoMode::Independent, CPU)
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
    let joined = Frames::concat(&out, 2);
    assert_eq!(joined.frames(), samples.frames());
    let calls = codec.calls.lock().unwrap().clone();
    let frames_per_call: Vec<usize> = calls.chunks(2).map(|pair| pair[0] * 2 / 3).collect();
    assert_eq!(frames_per_call, vec![144_000, 160_000, 80_001]);
}

#[test]
fn mid_side_feeds_exact_mid_and_side_stems() {
    struct Seen(Mutex<Vec<Vec<f32>>>);
    impl NeuralCodec for Seen {
        fn kind(&self) -> CodecKind {
            CodecKind::Encodec
        }
        fn reconstruct(&self, stem: &[f32], _request: &CodecRequest) -> Result<Vec<f32>, CodecError> {
            self.0.lock().unwrap().push(stem.to_vec());
            let is_side = stem.iter().all(|v| v.abs() < 1e-6);
            Ok(stem
                .iter()
                .map(|v| {
                    if is_side {
                        v * 0.88 - 0.015
                    } else {
                        v * 0.96 + 0.015
                    }
                })
                .collect())
        }
    }
    let center: Vec<f32> = (0..2400)
        .map(|n| (0.35 * (2.0 * PI * 220.0 * n as f64 / 2400.0).sin()) as f32)
        .collect();
    let stereo = Frames::from_channels(&[center.clone(), center.clone()]);
    let codec = Seen(Mutex::new(Vec::new()));
    let ms = encodec::resynthesize(&codec, &stereo, 24_000, StereoMode::MidSide, &CPU).unwrap();
    let seen = codec.0.lock().unwrap();
    assert!(
        seen.iter()
            .any(|s| s.iter().zip(&center).all(|(a, b)| (a - b).abs() <= 1e-6))
    );
    assert!(seen.iter().any(|s| s.iter().all(|&v| v == 0.0)));
    assert!(
        ms.channel(0)
            .iter()
            .zip(ms.channel(1))
            .all(|(l, r)| ((l - r) + 0.03).abs() < 1e-6)
    );
}

#[test]
fn dac_identity_restores_rms_on_the_native_grid() {
    for (mode, channels) in [
        (StereoMode::MidSide, 2usize),
        (StereoMode::Independent, 2),
        (StereoMode::MidSide, 1),
    ] {
        let codec = Recording::new(CodecKind::Dac, 0.6, 17);
        let x = noise(7, 1024, 2).scale(0.3).select_channels(0..channels);
        let y = raw::resynthesize(&codec, &x, 44_100, mode, None, &CPU).unwrap();
        assert_eq!((y.frames(), y.channels()), (x.frames(), x.channels()));
        assert!(
            y.as_slice()
                .iter()
                .zip(x.as_slice())
                .all(|(a, b)| (a - b).abs() <= 1e-5 + 1e-5 * b.abs())
        );
        assert_eq!(codec.calls.lock().unwrap().len(), channels);
    }
}

#[test]
fn dac_skips_a_silent_side_stem_and_keeps_exact_counts() {
    let codec = Recording::new(CodecKind::Dac, 1.0, 0);
    let mono: Vec<f32> = (0..2048)
        .map(|n| (0.25 * (20.0 * n as f64 / 2047.0).sin()) as f32)
        .collect();
    let dual = Frames::from_channels(&[mono.clone(), mono]);
    let y = raw::resynthesize(&codec, &dual, 44_100, StereoMode::MidSide, None, &CPU).unwrap();
    assert_eq!(codec.calls.lock().unwrap().len(), 1);
    assert!(
        y.channel(0)
            .iter()
            .zip(y.channel(1))
            .all(|(a, b)| (a - b).abs() <= 1e-7)
    );
    for (n, rate) in [(48_001usize, 48_000u32), (16_003, 16_000), (44_101, 44_100)] {
        let codec = Recording::new(CodecKind::Dac, 1.0, 0);
        let x = Frames::mono(
            (0..n)
                .map(|i| (0.2 * (2.0 * PI * 440.0 * i as f64 / f64::from(rate)).sin()) as f32)
                .collect(),
        );
        let clock = RationalClock::new(n, rate, 44_100, 512, 512).unwrap();
        let y = raw::resynthesize(&codec, &x, rate, StereoMode::MidSide, None, &CPU).unwrap();
        assert_eq!(codec.calls.lock().unwrap()[0], clock.padded_frames());
        assert_eq!(y.frames(), n);
        assert!(y.is_finite());
    }
    for fault in ["short", "nan"] {
        let codec = Recording {
            fault: Some(fault),
            ..Recording::new(CodecKind::Dac, 1.0, 0)
        };
        let err = raw::resynthesize(
            &codec,
            &Frames::zeros(1024, 1).map(|_| 0.2),
            44_100,
            StereoMode::MidSide,
            None,
            &CPU,
        );
        assert!(err.unwrap_err().to_string().starts_with("DAC returned"));
    }
    for bad in [
        Frames::zeros(0, 1),
        Frames::zeros(8, 3),
        Frames::mono(vec![f32::NAN]),
        Frames::mono(vec![f32::INFINITY]),
    ] {
        let err = raw::resynthesize(&codec, &bad, 44_100, StereoMode::MidSide, None, &CPU).unwrap_err();
        assert!(err.to_string().contains("DAC"), "{err}");
    }
    assert!(raw::validate_preserve_hf_hz(7999.0, 44_100).is_err());
    assert_eq!(
        raw::validate_preserve_hf_hz(9000.0, 16_000)
            .unwrap_err()
            .to_string(),
        "preserve_hf_hz must be between 8000 and 7200 Hz for 16000 Hz"
    );
}

/// A metadata seam that declares every structured source unreadable.
struct Truncated;

impl MetadataSeam for Truncated {
    fn ai_metadata(&self, _path: &Path) -> raiw_audio::Result<Markers> {
        Ok(Markers::new())
    }
    fn unreadable_reason(&self, _path: &Path) -> Option<String> {
        Some("truncated RIFF chunk".into())
    }
    fn markers_claim_ai(&self, _markers: &Markers) -> bool {
        false
    }
    fn strip_and_verify(
        &self,
        _s: &Path,
        _o: &Path,
        _k: bool,
    ) -> raiw_audio::Result<(std::path::PathBuf, Markers)> {
        Err(AudioError::unavailable("no metadata stage"))
    }
}

fn wav(dir: &Path, name: &str, frames: &Frames, rate: u32) -> std::path::PathBuf {
    let path = dir.join(name);
    encode_raw(frames, rate, &path, &["-c:a", "pcm_s16le"], None);
    path
}

fn forced() -> RemovalRequest {
    RemovalRequest {
        args: ProcessingArgs {
            verify: false,
            ..ProcessingArgs::default()
        },
        permissions: Permissions {
            force: true,
            ..Permissions::default()
        },
        ..RemovalRequest::default()
    }
}

#[test]
fn unreadable_containers_are_input_errors_not_skips() {
    let dir = tempfile::tempdir().unwrap();
    let source = wav(dir.path(), "source.wav", &stereo_tones(16_000, 1.0), 16_000);
    let runtime = AudioRuntime::bare().with_metadata(Arc::new(Truncated));
    let output = dir.path().join("output.wav");
    let err = remove_audio_invisible(&runtime, &source, &output, &forced()).unwrap_err();
    assert_eq!(err.to_string(), "cannot read source.wav: truncated RIFF chunk");
    assert!(!output.exists());
}

#[test]
fn source_change_during_resynthesis_blocks_publication() {
    struct Mutating(std::path::PathBuf);
    impl NeuralCodec for Mutating {
        fn kind(&self) -> CodecKind {
            CodecKind::Dac
        }
        fn reconstruct(&self, stem: &[f32], _request: &CodecRequest) -> Result<Vec<f32>, CodecError> {
            let mut data = std::fs::read(&self.0).map_err(|e| CodecError(e.to_string()))?;
            if let Some(last) = data.last_mut() {
                *last = last.wrapping_add(1);
            }
            std::fs::write(&self.0, data).map_err(|e| CodecError(e.to_string()))?;
            Ok(stem.to_vec())
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let source = wav(dir.path(), "source.wav", &stereo_tones(16_000, 1.0), 16_000);
    let output = dir.path().join("protected.wav");
    std::fs::write(&output, b"existing output").unwrap();
    let runtime = AudioRuntime::bare().with_codecs(CodecSet {
        dac: Some(Arc::new(Mutating(source.clone()))),
        ..CodecSet::default()
    });
    let mut request = forced();
    request.args.method = Method::Dac;
    let err = remove_audio_invisible(&runtime, &source, &output, &request).unwrap_err();
    assert_eq!(err.to_string(), "Audio source changed during processing");
    assert_eq!(std::fs::read(&output).unwrap(), b"existing output");
}

#[test]
fn missing_backends_name_their_extras() {
    let dir = tempfile::tempdir().unwrap();
    let source = wav(dir.path(), "source.wav", &stereo_tones(16_000, 1.0), 16_000);
    let mut request = forced();
    request.args.method = Method::Dac;
    let no_ffmpeg = AudioRuntime {
        tools: None,
        ..AudioRuntime::bare()
    };
    let err = remove_audio_invisible(&no_ffmpeg, &source, &dir.path().join("out.wav"), &request).unwrap_err();
    assert!(
        matches!(err, AudioError::Unavailable(_)) && err.to_string().contains("audio-codec"),
        "{err}"
    );
    let err = remove_audio_invisible(
        &AudioRuntime::bare(),
        &source,
        &dir.path().join("out.wav"),
        &request,
    )
    .unwrap_err();
    assert_eq!(
        err.to_string(),
        "DAC processing requires remove-ai-watermarks[audio-codec]"
    );
    let report = identify_audio(&AudioRuntime::bare(), &source, &IdentifyOptions::default()).unwrap();
    assert_eq!(report.watermark_status.as_str(), "unverified");
    assert!(report.detections.is_empty());
    assert_eq!(
        report.caveats,
        vec!["audioseal decoding is unavailable; install remove-ai-watermarks[audio-detect]."]
    );
}

#[test]
fn all_and_batch_route_metadata_and_invisible_stages() {
    let dir = tempfile::tempdir().unwrap();
    let batch_in = dir.path().join("batch_in");
    std::fs::create_dir(&batch_in).unwrap();
    let marked = wav(&batch_in, "b_marked.wav", &stereo_tones(16_000, 1.0), 16_000);
    let clean = wav(
        &batch_in,
        "a_clean.wav",
        &stereo_tones(16_000, 1.0).map(|v| v * 0.0 + 0.01),
        16_000,
    );
    let markers: Markers = [("claim_generator".to_owned(), "Suno".to_owned())]
        .into_iter()
        .collect();
    let metadata = fakes::FakeMetadata {
        source: clean.canonicalize().unwrap(),
        markers: markers.clone(),
        claims_ai: true,
    };
    let runtime = AudioRuntime::bare()
        .with_decoders(fakes::decoders(&serde_json::json!({ "audioseal": "band" })))
        .with_metadata(Arc::new(metadata));
    let options = AllOptions {
        include_invisible: true,
        keep_standard: true,
        removal: RemovalRequest {
            args: ProcessingArgs {
                scheme: Scheme::Auto,
                ..ProcessingArgs::default()
            },
            ..RemovalRequest::default()
        },
    };
    let all = remove_audio_all(
        &runtime,
        &marked,
        Some(&dir.path().join("marked_all.wav")),
        &options,
    )
    .unwrap();
    assert!(all.invisible_applied);
    assert_eq!(
        all.invisible_result.as_ref().unwrap().verified_removed,
        Some(true)
    );
    let meta_only = remove_audio_all(
        &runtime,
        &clean,
        Some(&dir.path().join("clean_all.wav")),
        &options,
    )
    .unwrap();
    assert!(!meta_only.invisible_applied);
    assert_eq!(meta_only.detected_metadata, markers);
    assert!(meta_only.remaining_metadata.is_empty());
    let summary = remove_audio_batch(&runtime, &batch_in, None, BatchMode::All, &options).unwrap();
    assert_eq!(
        (summary.processed(), summary.failed(), summary.invisible_applied()),
        (2, 0, 1)
    );
    assert_eq!(summary.output_directory, dir.path().join("batch_in_clean"));
    assert!(summary.output_directory.join("a_clean.wav").is_file());
    assert!(summary.output_directory.join("b_marked.wav").is_file());
    let err = remove_audio_batch(&runtime, &batch_in, None, BatchMode::Metadata, &options).unwrap_err();
    assert_eq!(
        err.to_string(),
        "The invisible audio stage is available only in all mode"
    );
}
