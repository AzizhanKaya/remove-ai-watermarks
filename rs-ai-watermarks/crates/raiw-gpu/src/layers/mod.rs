//! Building blocks shared by the ported architectures.

pub mod conv;
pub mod conv2d;
pub mod linear;
pub mod lstm;
pub mod norm;
pub mod residual;
pub mod seanet;
pub mod snake;
pub mod vq;

pub use conv::{Conv1d, Conv1dSpec, ConvTranspose1d, ConvTranspose1dSpec};
pub use conv2d::{Conv2d, Conv2dSpec};
pub use linear::Linear;
pub use lstm::Lstm;
pub use residual::{ResidualUnit, UnitNaming};
pub use seanet::{
    PadKind, SeanetConfig, SeanetDecoder, SeanetEncoder, SeanetLayout, SeanetResnetBlock, StreamingConv1d,
    StreamingConvTranspose1d,
};
pub use snake::Snake1d;
pub use vq::{CodeScore, VectorQuantizer, VqSpec};
