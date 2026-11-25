//! AI-generation metadata: detection, lossless removal and portable records.

pub mod tc260_signature;

pub use tc260_signature::{SignatureStatus, Tc260Signature, check_tc260_signature};
