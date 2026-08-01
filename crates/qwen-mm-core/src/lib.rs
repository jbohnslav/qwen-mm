//! Dependency-light contracts for Qwen multimodal preprocessing.
//!
//! The crate owns immutable offline profiles, Rust-native request/media types,
//! resource-safe preflight, stable errors, and exact parity output layouts.
//! It deliberately contains no Python, Torch, `OpenCV`, networking, or model
//! execution.
//!
//! ```
//! assert_eq!(qwen_mm_core::version(), "0.1.0");
//! ```

#![forbid(unsafe_code)]

pub mod error;
pub mod geometry;
pub mod limits;
pub mod media;
pub mod output;
pub mod patchify;
pub mod profile;
pub mod request;
pub mod resize;
pub mod text;

#[cfg(test)]
mod media_conformance_tests;
#[cfg(test)]
mod patchify_conformance_tests;

pub use error::{DiagnosticValue, ErrorCategory, QwenError, Result, ValidationStage};
pub use geometry::{
    ImageGeometryPlan, explicit_dimensions, plan_image_geometry, round_by_factor, smart_resize,
};
pub use limits::{
    LimitOverrides, PreflightSummary, ProfiledRequest, ResourceLimits, checked_add,
    checked_capacity_bytes, checked_mul, preflight_batch,
};
pub use media::{PreparedRgbImage, prepare_image_rgb8};
pub use output::{
    CoordinateRange, ImageSidecar, IntegrationSidecar, Matrix, PreparedArrays, PreparedBatch,
    ReplacementRange, VideoSidecar,
};
pub use patchify::{ImagePatchifyPlan, PreparedImage, patchify_image_rgb8, plan_image_patchify};
pub use profile::{
    ProcessorClasses, Profile, ProfileAlias, ProfileRegistry, TokenizerProfile, VisualProfile,
};
pub use request::{
    ContentItem, ExcludedOptions, FunctionCall, ImageFormat, ImageInput, ImageOptions, ImageRef,
    Message, MessageContent, OccurrenceLocation, Request, RequestOptions, Rgb8, Role, ToolCall,
    ToolDefinition, VideoInput, VideoOptions, VideoRef,
};
pub use resize::{resize_image_rgb8, resize_video_rgb8_to_f32};
pub use text::{
    PlannedTextRequest, PreparedTextBatch, PreparedTextRequest, TextProcessor, TextReplacement,
    VisualExpansion, VisualModality,
};

/// Returns the core crate version embedded at compile time.
#[must_use]
pub const fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    #[test]
    fn version_matches_workspace_package_version() {
        assert_eq!(super::version(), env!("CARGO_PKG_VERSION"));
    }
}
