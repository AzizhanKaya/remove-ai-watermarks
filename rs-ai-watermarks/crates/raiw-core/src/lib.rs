//! Shared foundations for the `rs-ai-watermarks` workspace.

pub mod constants;
pub mod fs;
pub mod hash;
pub mod pyjson;
pub mod schema;
pub mod tc260;
#[cfg(any(test, feature = "testing"))]
pub mod testing;

pub use fs::{StagedSibling, model_cache_dir, resolve_lenient};
pub use hash::{sha256_bytes, sha256_file};
pub use pyjson::{JsonStyle, to_py_json};
