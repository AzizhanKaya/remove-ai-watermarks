//! Error type shared by the runtime, hub, weight loaders and models.

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("unknown device {0:?}; expected auto, cpu, cuda, mps, vulkan, metal or rocm")]
    UnknownDevice(String),
    #[error("requested {requested} device is unavailable: {reason}")]
    DeviceUnavailable { requested: String, reason: String },
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("hub file {repo}/{filename}@{revision} is not cached and downloads are disabled")]
    NotCached {
        repo: String,
        revision: String,
        filename: String,
    },
    #[error("hub request for {url} failed: {reason}")]
    Http { url: String, reason: String },
    #[error("pinned digest mismatch for {filename}: {actual} != {expected}")]
    DigestMismatch {
        filename: String,
        actual: String,
        expected: String,
    },
    #[error("cannot read weights {path}: {reason}")]
    Weights { path: PathBuf, reason: String },
    #[error("missing weight tensor {0}")]
    MissingTensor(String),
    #[error("weight tensor {name} has shape {actual:?}, expected {expected:?}")]
    ShapeMismatch {
        name: String,
        actual: Vec<usize>,
        expected: Vec<usize>,
    },
    #[error("unused weight tensors: {0:?}")]
    UnusedTensors(Vec<String>),
    #[error("invalid model configuration: {0}")]
    Config(String),
    #[error("invalid input: {0}")]
    Input(String),
    #[error("model produced non-finite values")]
    NonFinite,
    #[error("tensor readback failed: {0}")]
    Readback(String),
}

pub type Result<T, E = GpuError> = std::result::Result<T, E>;

impl GpuError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}
