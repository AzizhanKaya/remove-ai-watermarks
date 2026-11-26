//! Visible AI-mark localization and removal for video (port of `video_visible.py`).

pub mod clean;
pub mod detect;
pub mod encode;
pub mod frame;
pub mod mask;
pub mod policy;
pub mod scan;
pub mod select;
pub mod templates;

pub use clean::{CleanOptions, for_each_clean_frame};
pub use encode::{EncodeCleanOptions, encode_clean_video};
pub use frame::{FrameLocalization, PreparedFrame, Region, region_iou};
pub use policy::{MaskStyle, VisibleMarkPolicy, policy, stabilize_localizations};
pub use scan::{Scans, VideoScan, probe_empty_video_scan, scan_video_marks};
pub use select::{
    FrameRegions, MergedMarks, SelectedMark, merge_selected_visible_marks, select_stable_visible_marks,
};
