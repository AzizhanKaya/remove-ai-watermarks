//! Byte-level container walkers used by metadata scanning, stripping and readability probes.
//!
//! Every walker is bounded and fail-closed: a malformed structure stops the walk, and a
//! rewrite either keeps every byte offset or returns the input untouched. Metadata-level
//! AI-marker rules are reached through [`MetadataHooks`].

pub mod av1;
pub mod ebml;
pub mod flv;
pub mod gif;
pub mod hooks;
pub mod isobmff;
pub mod jpeg;
pub mod markers;
pub mod png;
pub mod readability;
pub mod riff;
pub mod source;
pub mod tiff;

pub use hooks::MetadataHooks;
pub use source::{FileSource, ReadAt};
