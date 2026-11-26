//! The plain-text output of the `metadata` command (`cli.py`), so the CLI only prints it.

use std::fmt::Write as _;
use std::path::Path;

use crate::detect::{AiMetadata, get_ai_metadata, has_ai_metadata};
use crate::readability::unreadable_reason;
use crate::strip::file_name;
use crate::verify::strip_and_verify;

/// `_print_metadata_not_a_clean_verdict`.
pub const NOT_A_CLEAN_VERDICT: &str = "  This is not the same as 'clean': a pixel watermark such as SynthID cannot be\n  detected here once its metadata proxy is absent.";

/// `_banner()`.
pub fn banner(version: &str) -> String {
    format!("Remove-AI-Watermarks v{version}\nVisible & invisible watermark removal")
}

/// `_print_metadata_report(source, has_ai, metadata)`, one line per `console.print`.
pub fn metadata_report(source: &Path, has_ai: bool, metadata: &AiMetadata) -> String {
    let name = file_name(source);
    if !has_ai {
        return format!("  No AI metadata found in {name}\n{NOT_A_CLEAN_VERDICT}\n");
    }
    let mut out = format!("  Warning: AI metadata detected in {name}:\n");
    if let Some(synthid) = metadata.get("synthid_watermark").filter(|s| !s.is_empty()) {
        let _ = writeln!(out, "  Warning: SynthID watermark {synthid}");
    }
    out.push_str("  Key  Value");
    for (key, value) in metadata {
        let shown: String = value.chars().take(80).collect();
        let _ = write!(out, "\n  {key}  {shown}");
    }
    out.push('\n');
    out
}

/// `_unreadable_input_exit`'s message.
pub fn unreadable_message(source: &Path, reason: &str, after_report: bool) -> String {
    let scanned = if after_report {
        "The scan above is incomplete"
    } else {
        "Nothing was scanned"
    };
    format!(
        "Error: cannot read {}: {reason}. {scanned}, not a no-signal result.",
        file_name(source)
    )
}

/// What one `metadata` invocation prints and its exit code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandOutput {
    pub stdout: String,
    pub exit_code: i32,
}

/// The whole `metadata SOURCE [--check] [--remove] [-o OUT] [--keep-standard/--remove-all]` run.
pub fn metadata_command(
    source: &Path,
    check: bool,
    remove: bool,
    output: Option<&Path>,
    keep_standard: bool,
    version: &str,
) -> CommandOutput {
    let mut stdout = format!("{}\n", banner(version));
    let done = |stdout: String, exit_code| CommandOutput { stdout, exit_code };
    if let Some(reason) = unreadable_reason(source) {
        let _ = writeln!(stdout, "  {}", unreadable_message(source, &reason, false));
        return done(stdout, 1);
    }
    if check || !remove {
        let has_ai = match has_ai_metadata(source) {
            Ok(flag) => flag,
            Err(err) => {
                let _ = writeln!(stdout, "  Error: cannot process {}: {err}", file_name(source));
                return done(stdout, 1);
            }
        };
        let metadata = if has_ai {
            get_ai_metadata(source)
        } else {
            AiMetadata::new()
        };
        stdout.push_str(&metadata_report(source, has_ai, &metadata));
        if !remove {
            return done(stdout, 0);
        }
    }
    match strip_and_verify(source, output, keep_standard) {
        Err(err) => {
            let _ = writeln!(stdout, "  Error: cannot process {}: {err}", file_name(source));
            done(stdout, 1)
        }
        Ok(outcome) if !outcome.remaining.is_empty() => {
            let mut keys: Vec<&String> = outcome.remaining.keys().collect();
            keys.sort();
            let joined: Vec<&str> = keys.iter().map(|k| k.as_str()).collect();
            let _ = writeln!(
                stdout,
                "  FAILED: {} AI metadata marker(s) survived in {}",
                outcome.remaining.len(),
                outcome.output.display()
            );
            let _ = writeln!(stdout, "    still present: {}", joined.join(", "));
            done(stdout, 1)
        }
        Ok(outcome) => {
            let _ = writeln!(stdout, "  AI metadata stripped -> {}", outcome.output.display());
            let _ = writeln!(stdout, "{NOT_A_CLEAN_VERDICT}");
            done(stdout, 0)
        }
    }
}
