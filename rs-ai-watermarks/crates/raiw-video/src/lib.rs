//! Video watermark inspection and removal (port of the Python `video*.py` modules).
//!
//! Every stage shells out to the resolved system ffmpeg/ffprobe: frames are decoded to BGR the
//! way OpenCV's FFmpeg backend does, processed with `raiw-cv`, and re-encoded through a frame
//! pipe; audio and metadata travel by stream copy.

pub mod api;
pub mod audio;
pub mod cli;
pub mod decode;
pub mod encode;
pub mod error;
pub mod invisible;
pub mod messages;
#[cfg(feature = "gpu")]
pub mod neural;
pub mod nut;
pub mod probe;
mod process;
pub mod provenance;
pub mod py;
pub mod quality;
pub mod report;
pub mod sampling;
pub mod seedance;
pub mod temporal;
pub mod tools;
pub mod visible;
pub mod wan;

pub use api::{
    AllVideoOptions, BatchVideoOptions, EraseVideoOptions, IdentifyVideoOptions, TrimBumper, VideoRuntime,
    VisibleVideoOptions, identify_video, inspect_video_metadata, parse_mark_selector, remove_video_all,
    remove_video_batch, remove_video_invisible, remove_video_metadata, remove_video_region,
    remove_video_visible, video_output, video_source,
};
pub use audio::{
    VideoAudioMultiStreamResult, VideoAudioOptions, VideoAudioRemovalResult, remove_video_audio,
    remove_video_audio_all_streams,
};
pub use decode::{VideoReader, open_video_capture, open_video_capture_16bit, set_ffmpeg_threads};
pub use error::{Result, VideoError};
pub use invisible::{InvisibleMode, InvisibleOptions, NeuralBackends, VideoVae};
pub use messages::{
    MessageRemovalOptions, MessageScheme, VideoMessageModel, VideoMessageModels, VideoMessageRemovalResult,
    VideoMessageReport, VideoWatermarkPresenceReport, detect_video_watermark_presence,
    identify_video_message, remove_video_message,
};
pub use process::RawVideoEncoder;
pub use report::*;
pub use seedance::{SeedanceQuality, regenerate_seedance_experimental};
pub use wan::{
    WanCall, WanOptions, WanPipeline, WanWindowing, can_run_wan_on_source, regenerate_wan_experimental,
};
