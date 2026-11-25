//! Building blocks shared by the ported architectures.

pub mod conv;
pub mod lstm;
pub mod residual;
pub mod seanet;
pub mod snake;
pub mod vq;

pub use conv::{Conv1d, Conv1dSpec, ConvTranspose1d, ConvTranspose1dSpec};
pub use lstm::Lstm;
pub use residual::{ResidualUnit, UnitNaming};
pub use seanet::{
    PadKind, SeanetConfig, SeanetDecoder, SeanetEncoder, SeanetLayout, SeanetResnetBlock, StreamingConv1d,
    StreamingConvTranspose1d,
};
pub use snake::Snake1d;
pub use vq::{CodeScore, VectorQuantizer, VqSpec};
