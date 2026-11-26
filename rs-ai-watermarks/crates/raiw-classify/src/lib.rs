//! Source-pipeline classification (never part of `identify`).
//!
//! Port of `source_classify.py` in progress: constants, the result type, and the
//! pickle-free `.npz` reader are in place; model loading, features, and the ViT are not.

pub mod consts;
pub mod error;
pub mod npz;
pub mod result;

pub use error::{ClassifyError, Result};
pub use npz::{NpyArray, NpyData, NpyHeader, Npz};
pub use result::{Scores, SourceClassification, SourceReason};
