//! `metadata --check` / `--remove` text output, pinned to the Python CLI's stdout.

use raiw_core::testing::fixture;
use raiw_metadata::report::metadata_command;

#[test]
fn check_output_matches_the_python_cli() {
    let marked = metadata_command(
        &fixture("provenance/grok-1.jpg"),
        true,
        false,
        None,
        true,
        "0.75.0",
    );
    assert_eq!(
        marked.stdout,
        "Remove-AI-Watermarks v0.75.0\nVisible & invisible watermark removal\n  \
         Warning: AI metadata detected in grok-1.jpg:\n  Key  Value\n  \
         xai_signature  xAI/Grok signature (UUID + Signature blob)\n"
    );
    assert_eq!(marked.exit_code, 0);
    let clean = metadata_command(
        &fixture("visible/baidu/example.mp4"),
        true,
        false,
        None,
        true,
        "0.75.0",
    );
    assert_eq!(
        clean.stdout,
        "Remove-AI-Watermarks v0.75.0\nVisible & invisible watermark removal\n  \
         No AI metadata found in example.mp4\n  This is not the same as 'clean': a pixel watermark such as \
         SynthID cannot be\n  detected here once its metadata proxy is absent.\n"
    );
}

#[test]
fn remove_reports_the_stripped_output_and_unreadable_input() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("clean.jpg");
    let removed = metadata_command(
        &fixture("provenance/grok-1.jpg"),
        false,
        true,
        Some(&out),
        true,
        "0.75.0",
    );
    assert_eq!(removed.exit_code, 0, "{}", removed.stdout);
    assert!(
        removed
            .stdout
            .contains(&format!("  AI metadata stripped -> {}\n", out.display()))
    );
    let junk = dir.path().join("junk.png");
    std::fs::write(&junk, b"not an image").unwrap();
    let refused = metadata_command(&junk, true, false, None, true, "0.75.0");
    assert_eq!(refused.exit_code, 1);
    assert!(refused.stdout.ends_with(
        "  Error: cannot read junk.png: the content is not a recognized image, video or audio container. \
         Nothing was scanned, not a no-signal result.\n"
    ));
}
