//! Differential tests of the SciPy-equivalent kernels against `tools/goldens/audio.py`.

mod common;

use common::{check_f32, check_f64, golden, sha_f32, splitmix_uniform, synth};
use raiw_audio::codecs::stream::fade_in;
use raiw_audio::dsp::numpy::cos_f32;
use raiw_audio::dsp::{Band, Section, butter_sos, resample_poly, sosfilt_zi, sosfiltfilt};
use raiw_audio::{Frames, RationalClock, spectral};
use serde_json::Value;

#[test]
fn float32_cosine_and_crossfade_match_numpy_bit_for_bit() {
    let golden = golden("fade");
    for case in golden["fades"].as_array().unwrap() {
        let overlap = case["overlap"].as_u64().unwrap() as usize;
        assert!(
            check_f32(&fade_in(overlap), &case["fade"], 1e-6, &format!("fade {overlap}")),
            "fade {overlap}"
        );
    }
    let x: Vec<f32> = splitmix_uniform(77, 100_000)
        .into_iter()
        .map(|u| (40.0 * u - 20.0) as f32)
        .collect();
    let cos: Vec<f32> = x.iter().map(|&v| cos_f32(v)).collect();
    assert!(
        check_f32(&cos, &golden["cos"], 1e-6, "np.cos float32"),
        "np.cos float32 bit-identical"
    );
}

fn design(value: &Value) -> (Band, usize, f64) {
    let design = value.as_array().unwrap();
    let btype = design[0].as_str().unwrap();
    let order = design[1].as_u64().unwrap() as usize;
    let rate = design[3].as_f64().unwrap();
    let band = match btype {
        "bandstop" => {
            let edges = design[2].as_array().unwrap();
            Band::Bandstop(edges[0].as_f64().unwrap(), edges[1].as_f64().unwrap())
        }
        "lowpass" | "low" => Band::Lowpass(design[2].as_f64().unwrap()),
        "high" => Band::Highpass(design[2].as_f64().unwrap()),
        other => panic!("{other}"),
    };
    (band, order, rate)
}

#[test]
fn resample_poly_matches_scipy_bit_for_bit() {
    let golden = golden("resample");
    let mut exact = 0;
    let cases = golden["cases"].as_array().unwrap();
    for case in cases {
        let name = case["signal"].as_str().unwrap();
        let x = synth(&golden["signals"][name]).channel(0);
        assert_eq!(
            sha_f32(&x),
            case["input_sha256"].as_str().unwrap(),
            "{name}: synthesized input"
        );
        let (up, down) = (
            case["up"].as_u64().unwrap() as usize,
            case["down"].as_u64().unwrap() as usize,
        );
        let y = resample_poly(&x, up, down);
        if check_f32(&y, &case["output"], 1e-6, &format!("{name} {up}/{down}")) {
            exact += 1;
        }
    }
    assert_eq!(
        exact,
        cases.len(),
        "every resample_poly case should be bit-identical"
    );
}

#[test]
fn butterworth_designs_match_scipy() {
    let golden = golden("butter");
    let mut identical = 0;
    let cases = golden["cases"].as_array().unwrap();
    for case in cases {
        let design_value = serde_json::json!([case["btype"], case["order"], case["cutoff"], case["rate"]]);
        let (band, order, rate) = design(&design_value);
        let sos = butter_sos(order, band, rate).unwrap();
        let expected: Vec<Section> = case["sos"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                let row: Vec<f64> = row
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect();
                row.try_into().unwrap()
            })
            .collect();
        assert_eq!(sos.len(), expected.len(), "{design_value}");
        let mut same = true;
        for (a, e) in sos.iter().zip(&expected) {
            for (x, y) in a.iter().zip(e) {
                assert!(
                    (x - y).abs() <= 1e-12 * y.abs().max(1e-3),
                    "{design_value}: {a:?} != {e:?}"
                );
                same &= x == y;
            }
        }
        identical += usize::from(same);
        let zi = sosfilt_zi(&sos);
        for (a, e) in zi.iter().zip(case["zi"].as_array().unwrap()) {
            for (x, y) in a.iter().zip(e.as_array().unwrap()) {
                let y = y.as_f64().unwrap();
                assert_eq!(*x, y, "{design_value}: zi {a:?} != {e}");
            }
        }
    }
    assert_eq!(
        identical,
        cases.len(),
        "every Butterworth design should be bit-identical"
    );
}

#[test]
fn sosfiltfilt_matches_scipy() {
    let golden = golden("sosfiltfilt");
    for case in golden["cases"].as_array().unwrap() {
        let name = case["signal"].as_str().unwrap();
        let x = synth(&golden["signals"][name]).channel(0);
        let (band, order, rate) = design(&case["design"]);
        let sos = butter_sos(order, band, rate).unwrap();
        let padlen = case["padlen"].as_u64().map(|v| v as usize);
        let y = if case["float64_input"].as_bool().unwrap() {
            let wide: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
            sosfiltfilt(&sos, &wide, padlen).unwrap()
        } else {
            sosfiltfilt(&sos, &x, padlen).unwrap()
        };
        let context = format!("{name} {}", case["design"]);
        assert!(
            check_f64(&y, &case["output"], 1e-9, &context),
            "{context}: float64 bit-identical"
        );
        let narrow: Vec<f32> = y.iter().map(|&v| v as f32).collect();
        assert!(
            check_f32(&narrow, &case["output_f32"], 1e-6, &context),
            "{context}: float32 bit-identical"
        );
    }
}

#[test]
fn spectral_profile_matches_python_streaming() {
    let golden = golden("spectral");
    for case in golden["cases"].as_array().unwrap() {
        let name = case["signal"].as_str().unwrap();
        let spec = &golden["signals"][name];
        let x = synth(spec);
        assert_eq!(
            sha_f32(x.as_slice()),
            case["input_sha256"].as_str().unwrap(),
            "{name}"
        );
        let rate = spec["rate"].as_u64().unwrap() as u32;
        let chunks: Vec<Frames> = match case["chunk"].as_u64() {
            None => vec![x.clone()],
            Some(size) => {
                let size = size as usize;
                (0..x.frames())
                    .step_by(size)
                    .map(|s| x.slice(s..s + size))
                    .collect()
            }
        };
        let out: Vec<Frames> = spectral::filter_chunks(chunks.into_iter().map(Ok), rate)
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        let lengths: Vec<u64> = out.iter().map(|c| c.frames() as u64).collect();
        let expected: Vec<u64> = case["lengths"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        assert_eq!(lengths, expected, "{name}");
        let joined = Frames::concat(&out, x.channels());
        let context = format!("spectral {name} {:?}", case["chunk"]);
        assert!(
            check_f32(joined.as_slice(), &case["output"], 1e-6, &context),
            "{context}: bit-identical"
        );
    }
    let errors = &golden["errors"];
    assert_eq!(
        spectral::filter_chunks(std::iter::empty(), 3000)
            .err()
            .map(|e| e.to_string())
            .as_deref(),
        errors["3000"].as_str()
    );
    assert!(errors["2999"].is_string());
}

#[test]
fn rational_clock_records_and_roundtrips_match() {
    let golden = golden("clock");
    for case in golden["records"].as_array().unwrap() {
        let args: Vec<u64> = case["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        let clock = RationalClock::new(
            args[0] as usize,
            args[1] as u32,
            args[2] as u32,
            args[3] as usize,
            args[4] as usize,
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(clock.record()).unwrap(),
            case["record"],
            "{args:?}"
        );
        assert_eq!(
            clock.padded_frames() as u64,
            case["padded_frames"].as_u64().unwrap()
        );
    }
    for case in golden["roundtrips"].as_array().unwrap() {
        let name = case["signal"].as_str().unwrap();
        let x = synth(&case["spec"]);
        assert_eq!(
            sha_f32(x.as_slice()),
            case["input_sha256"].as_str().unwrap(),
            "{name}"
        );
        let rate = case["spec"]["rate"].as_u64().unwrap() as u32;
        let clock = RationalClock::new(
            x.frames(),
            rate,
            44_100,
            case["hop"].as_u64().unwrap() as usize,
            1024,
        )
        .unwrap();
        let model = clock.forward(&x).unwrap();
        assert!(
            check_f32(model.as_slice(), &case["model"], 1e-6, name),
            "{name}: forward bit-identical"
        );
        let padded = clock.pad_model(&model).unwrap();
        assert!(
            check_f32(padded.as_slice(), &case["padded"], 1e-6, name),
            "{name}: reflect pad"
        );
        let back = clock.inverse(&clock.unpad_model(&padded).unwrap()).unwrap();
        assert!(
            check_f32(back.as_slice(), &case["inverse"], 1e-6, name),
            "{name}: inverse bit-identical"
        );
    }
}
