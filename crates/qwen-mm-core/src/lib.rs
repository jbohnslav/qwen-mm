//! Core primitives for Qwen multimodal preprocessing.
//!
//! This bootstrap intentionally contains no processor behavior. Semantic
//! implementations land behind conformance tickets after the compatibility
//! contract and golden reference are frozen.
//!
//! ```
//! assert_eq!(qwen_mm_core::version(), "0.1.0");
//! ```

#![forbid(unsafe_code)]

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
