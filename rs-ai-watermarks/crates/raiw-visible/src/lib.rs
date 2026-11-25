//! Visible AI mark detection and removal: the mark registry, provider engines, the shared region
//! eraser, and the image codec boundary.
//!
//! Removal is localize -> fill -> validate: an engine localizes its mark to a mask, one shared
//! fill backend erases it, and the same engine checks the result read-only.

mod assets;
mod cvx;
pub mod engines;
pub mod eraser;
pub mod error;
pub mod geometry;
pub mod image_io;
mod np;
pub mod raster;
pub mod registry;
pub mod remove;
pub mod text_mark;

pub use engines::{EngineDetection, MarkEngine, VisibleEngine};
pub use eraser::{Backend, EraseOptions, FillBackend, FillBackends, FillKind, Selection, boxes_to_mask, erase};
pub use error::{Result, VisibleError};
pub use geometry::{MatchBox, Region};
pub use image_io::{DecodeState, LoadedImage, load_image, save_image};
pub use raster::{Image, Mask, Pixels, to_bgr};
pub use registry::{
    Candidate, Context, Decision, KnownMark, Localization, MarkDetection, MarkRemovalResult, MarkValidationStatus,
    Registry, Sensitivity, Trust, VisibleRemovalResult, VisibleRemovalStatus, validate_sensitivity,
};
pub use remove::{MetadataStripper, VisibleOptions, remove_visible};
