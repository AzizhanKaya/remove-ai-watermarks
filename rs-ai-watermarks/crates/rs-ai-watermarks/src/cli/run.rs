//! Command handlers: each maps parsed arguments onto one library call and prints its output.

#[cfg(feature = "metadata")]
use super::image::MetadataCmd;

/// `metadata SOURCE [--check|--remove] [-o OUT] [--keep-standard/--remove-all]`.
#[cfg(feature = "metadata")]
pub fn metadata(cmd: &MetadataCmd) -> anyhow::Result<i32> {
    let output = raiw_metadata::report::metadata_command(
        &cmd.source,
        cmd.mode.check,
        cmd.mode.remove,
        cmd.output.as_deref(),
        cmd.keep_standard(),
        crate::VERSION,
    );
    print!("{}", output.stdout);
    Ok(output.exit_code)
}
