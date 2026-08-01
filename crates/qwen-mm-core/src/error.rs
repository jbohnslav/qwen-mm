//! Stable public failures and machine-readable diagnostic context.

use std::{collections::BTreeMap, fmt};

/// Stable error categories defined by compatibility contract v1.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum ErrorCategory {
    /// The request structure or a value in it is invalid.
    InvalidRequest,
    /// A deliberately excluded option was requested.
    UnsupportedOption,
    /// A media source or format is outside the supported envelope.
    UnsupportedMedia,
    /// A profile or one of its immutable inputs does not match the contract.
    ProfileMismatch,
    /// Recognized encoded media could not be decoded.
    MediaDecode,
    /// Media geometry or a geometry-related option is invalid.
    MediaGeometry,
    /// A configured resource limit was exceeded.
    ResourceLimit,
    /// Checked size, offset, stride, or capacity arithmetic overflowed.
    ArithmeticOverflow,
    /// A caller-owned destination is missing, overlapping, or too small.
    DestinationTooSmall,
    /// Validated state violated an internal processor invariant.
    InternalInvariant,
}

impl ErrorCategory {
    /// Returns the compatibility-stable snake-case category name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidRequest => "invalid_request",
            Self::UnsupportedOption => "unsupported_option",
            Self::UnsupportedMedia => "unsupported_media",
            Self::ProfileMismatch => "profile_mismatch",
            Self::MediaDecode => "media_decode",
            Self::MediaGeometry => "media_geometry",
            Self::ResourceLimit => "resource_limit",
            Self::ArithmeticOverflow => "arithmetic_overflow",
            Self::DestinationTooSmall => "destination_too_small",
            Self::InternalInvariant => "internal_invariant",
        }
    }
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Compatibility-stable validation stages in their deterministic order.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ValidationStage {
    /// Request/message/content structure.
    RequestStructure,
    /// Alias and immutable profile verification.
    Profile,
    /// Supported and explicitly excluded options.
    Options,
    /// Checked resource and capacity preflight.
    ResourcePreflight,
    /// Media recognition and decoding.
    Decode,
    /// Media geometry and processing-stage validation.
    Geometry,
    /// Caller-owned destination validation.
    Destination,
    /// Execution-time invariant checks.
    ExecutionInvariant,
}

impl ValidationStage {
    /// Every stage in frozen compatibility precedence.
    pub const ORDER: [Self; 8] = [
        Self::RequestStructure,
        Self::Profile,
        Self::Options,
        Self::ResourcePreflight,
        Self::Decode,
        Self::Geometry,
        Self::Destination,
        Self::ExecutionInvariant,
    ];

    /// Returns the stable snake-case stage name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RequestStructure => "request_structure",
            Self::Profile => "profile",
            Self::Options => "options",
            Self::ResourcePreflight => "resource_preflight",
            Self::Decode => "decode",
            Self::Geometry => "geometry",
            Self::Destination => "destination",
            Self::ExecutionInvariant => "execution_invariant",
        }
    }
}

/// A typed value in an error's diagnostic context.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum DiagnosticValue {
    /// UTF-8 text.
    Text(String),
    /// A signed integer.
    Integer(i128),
    /// An unsigned integer.
    Unsigned(u128),
    /// A floating-point number.
    Float(f64),
    /// A Boolean value.
    Boolean(bool),
}

impl fmt::Display for DiagnosticValue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text(value) => formatter.write_str(value),
            Self::Integer(value) => value.fmt(formatter),
            Self::Unsigned(value) => value.fmt(formatter),
            Self::Float(value) => value.fmt(formatter),
            Self::Boolean(value) => value.fmt(formatter),
        }
    }
}

impl From<&str> for DiagnosticValue {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

impl From<String> for DiagnosticValue {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<usize> for DiagnosticValue {
    fn from(value: usize) -> Self {
        Self::Unsigned(value as u128)
    }
}

impl From<u64> for DiagnosticValue {
    fn from(value: u64) -> Self {
        Self::Unsigned(u128::from(value))
    }
}

impl From<i64> for DiagnosticValue {
    fn from(value: i64) -> Self {
        Self::Integer(i128::from(value))
    }
}

impl From<f64> for DiagnosticValue {
    fn from(value: f64) -> Self {
        Self::Float(value)
    }
}

impl From<bool> for DiagnosticValue {
    fn from(value: bool) -> Self {
        Self::Boolean(value)
    }
}

/// A categorized public error with stable, structured context.
#[derive(Clone, Debug, PartialEq)]
pub struct QwenError {
    category: ErrorCategory,
    message: String,
    context: BTreeMap<String, DiagnosticValue>,
}

impl QwenError {
    /// Creates a categorized error.
    pub fn new(category: ErrorCategory, message: impl Into<String>) -> Self {
        Self {
            category,
            message: message.into(),
            context: BTreeMap::new(),
        }
    }

    /// Adds or replaces one context field.
    #[must_use]
    pub fn with_context(
        mut self,
        key: impl Into<String>,
        value: impl Into<DiagnosticValue>,
    ) -> Self {
        self.context.insert(key.into(), value.into());
        self
    }

    /// Returns the stable category.
    #[must_use]
    pub const fn category(&self) -> ErrorCategory {
        self.category
    }

    /// Returns the human-readable diagnostic message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Returns structured context in deterministic key order.
    #[must_use]
    pub const fn context(&self) -> &BTreeMap<String, DiagnosticValue> {
        &self.context
    }
}

impl fmt::Display for QwenError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.category, self.message)?;
        if !self.context.is_empty() {
            formatter.write_str(" (")?;
            for (index, (key, value)) in self.context.iter().enumerate() {
                if index != 0 {
                    formatter.write_str(", ")?;
                }
                write!(formatter, "{key}={value}")?;
            }
            formatter.write_str(")")?;
        }
        Ok(())
    }
}

impl std::error::Error for QwenError {}

/// The core result type.
pub type Result<T> = std::result::Result<T, QwenError>;

#[cfg(test)]
mod tests {
    use super::{DiagnosticValue, ErrorCategory, QwenError, ValidationStage};

    #[test]
    fn every_category_has_the_frozen_name() {
        let cases = [
            (ErrorCategory::InvalidRequest, "invalid_request"),
            (ErrorCategory::UnsupportedOption, "unsupported_option"),
            (ErrorCategory::UnsupportedMedia, "unsupported_media"),
            (ErrorCategory::ProfileMismatch, "profile_mismatch"),
            (ErrorCategory::MediaDecode, "media_decode"),
            (ErrorCategory::MediaGeometry, "media_geometry"),
            (ErrorCategory::ResourceLimit, "resource_limit"),
            (ErrorCategory::ArithmeticOverflow, "arithmetic_overflow"),
            (ErrorCategory::DestinationTooSmall, "destination_too_small"),
            (ErrorCategory::InternalInvariant, "internal_invariant"),
        ];

        for (category, expected) in cases {
            assert_eq!(category.as_str(), expected);
            assert_eq!(category.to_string(), expected);
        }
    }

    #[test]
    fn context_is_typed_and_deterministically_ordered() {
        let error = QwenError::new(ErrorCategory::ResourceLimit, "too many messages")
            .with_context("limit", 256_usize)
            .with_context("actual", 257_usize);

        assert_eq!(
            error.context().get("actual"),
            Some(&DiagnosticValue::Unsigned(257))
        );
        assert_eq!(
            error.to_string(),
            "resource_limit: too many messages (actual=257, limit=256)"
        );
    }

    #[test]
    fn validation_precedence_contains_every_frozen_stage() {
        assert_eq!(
            ValidationStage::ORDER.map(ValidationStage::as_str),
            [
                "request_structure",
                "profile",
                "options",
                "resource_preflight",
                "decode",
                "geometry",
                "destination",
                "execution_invariant",
            ]
        );
        assert!(ValidationStage::Profile < ValidationStage::Decode);
        assert!(ValidationStage::ResourcePreflight < ValidationStage::Geometry);
    }
}
