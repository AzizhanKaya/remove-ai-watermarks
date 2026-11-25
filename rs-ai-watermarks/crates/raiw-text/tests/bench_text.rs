//! Release-mode timings mirroring `tools/bench/text.py` (run with `--ignored --nocapture`).

use std::time::Instant;

use raiw_core::testing::repo_root;
use raiw_text::{
    ThreefryOptions, encode_unicode_tags, encode_vsrmark, identify_text, mask_local_rewrite_facts, remove_text_metadata,
    score_synthid_threefry_text, score_textseal2_text, split_protected_text_spans, verify_text_rewrite_facts,
};
use serde_json::Value;

fn inputs() -> (String, Vec<(String, String)>) {
    let path = repo_root().join("data/evaluations/text-watermark/fidelity-cases.json");
    let cases: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let pairs: Vec<(String, String)> = cases["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| (case["source"].as_str().unwrap().to_owned(), case["candidate"].as_str().unwrap().to_owned()))
        .collect();
    let prose: Vec<&str> = pairs.iter().map(|(source, _)| source.as_str()).collect();
    let block = format!("{}\n\n```python\nvalue = 1\n```\n\nSee https://raiw.cc/docs and `code` here.\n\n", prose.join("\n\n"));
    let document = block.repeat(4);
    let marked = encode_unicode_tags(&encode_vsrmark(&document, b"bench-payload").unwrap(), "hidden").unwrap() + "\u{200b}\u{200c}";
    (marked, pairs)
}

fn bench(label: &str, repeat: u32, mut function: impl FnMut()) {
    function();
    let start = Instant::now();
    for _ in 0..repeat {
        function();
    }
    let elapsed = start.elapsed() / repeat;
    eprintln!("{label:<34} {:10.3} ms", elapsed.as_secs_f64() * 1000.0);
}

#[test]
#[ignore = "timing benchmark; run in release with --ignored --nocapture"]
fn text_hot_paths() {
    let (document, pairs) = inputs();
    eprintln!("document: {} chars, {} fact pairs", document.chars().count(), pairs.len());
    bench("identify_text", 5, || {
        identify_text(document.as_str()).unwrap();
    });
    bench("remove_text_metadata", 5, || {
        remove_text_metadata(document.as_str(), None).unwrap();
    });
    bench("split_protected_text_spans", 5, || {
        split_protected_text_spans(&document);
    });
    bench("verify_text_rewrite_facts (all)", 3, || {
        for (source, candidate) in &pairs {
            verify_text_rewrite_facts(source, candidate);
        }
    });
    bench("_mask_local_rewrite_facts (all)", 3, || {
        for (source, _) in &pairs {
            mask_local_rewrite_facts(source);
        }
    });
    #[cfg(feature = "langid")]
    {
        let model = repo_root().join(".venv/lib/python3.12/site-packages/py3langid/data/model.npz.xz");
        if model.exists() {
            let start = Instant::now();
            let identifier = raiw_text::langid::LanguageIdentifier::from_path(&model).unwrap();
            eprintln!("{:<34} {:10.3} ms", "langid model load", start.elapsed().as_secs_f64() * 1000.0);
            bench("_classify_text_language (all)", 3, || {
                for (source, _) in &pairs {
                    identifier.classify_text(source);
                }
            });
        }
    }
    let tokens: Vec<i64> = (0..4000i64).map(|i| ((i * 1_103_515_245 + 12345) >> 16) % 4990 + 10).collect();
    bench("score_synthid_threefry_text", 3, || {
        score_synthid_threefry_text(&tokens, 123_456_789, &ThreefryOptions::default()).unwrap();
    });
    bench("score_textseal2_text", 3, || {
        score_textseal2_text(&tokens, 424_242, None, 3, raiw_text::DEFAULT_ALPHA_001_Z).unwrap();
    });
}
