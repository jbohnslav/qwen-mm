//! Stable core-error to Python-exception mapping.

use pyo3::{
    PyTypeInfo, create_exception,
    exceptions::{PyException, PyMemoryError},
    prelude::*,
    types::{PyAny, PyDict},
};
use qwen_mm_core::{DiagnosticValue, ErrorCategory, QwenError};

create_exception!(qwen_mm._native, QwenMMError, PyException);
create_exception!(qwen_mm._native, InvalidRequestError, QwenMMError);
create_exception!(qwen_mm._native, UnsupportedOptionError, QwenMMError);
create_exception!(qwen_mm._native, UnsupportedMediaError, QwenMMError);
create_exception!(qwen_mm._native, ProfileMismatchError, QwenMMError);
create_exception!(qwen_mm._native, MediaDecodeError, QwenMMError);
create_exception!(qwen_mm._native, MediaGeometryError, QwenMMError);
create_exception!(qwen_mm._native, ResourceLimitError, QwenMMError);
create_exception!(qwen_mm._native, ArithmeticOverflowError, QwenMMError);
create_exception!(qwen_mm._native, DestinationTooSmallError, QwenMMError);
create_exception!(qwen_mm._native, InternalInvariantError, QwenMMError);

pub(crate) fn invalid_request(message: impl Into<String>) -> QwenError {
    QwenError::new(ErrorCategory::InvalidRequest, message)
}

pub(crate) fn python_conversion_error(detail: impl Into<String>) -> QwenError {
    QwenError::new(
        ErrorCategory::InternalInvariant,
        "native output could not be transferred to NumPy",
    )
    .with_context("detail", detail.into())
}

pub(crate) fn input_allocation_error(name: &'static str, elements: usize) -> QwenError {
    QwenError::new(
        ErrorCategory::InternalInvariant,
        "native input ownership allocation failed",
    )
    .with_context("python_memory_error", true)
    .with_context("name", name)
    .with_context("elements", elements)
}

pub(crate) fn to_python_error(py: Python<'_>, error: &QwenError) -> PyErr {
    if error.context().contains_key("python_memory_error") {
        return PyMemoryError::new_err(error.to_string());
    }
    let category = error.category();
    let message = error.to_string();
    let python_error = match category {
        ErrorCategory::InvalidRequest => PyErr::new::<InvalidRequestError, _>(message),
        ErrorCategory::UnsupportedOption => PyErr::new::<UnsupportedOptionError, _>(message),
        ErrorCategory::UnsupportedMedia => PyErr::new::<UnsupportedMediaError, _>(message),
        ErrorCategory::ProfileMismatch => PyErr::new::<ProfileMismatchError, _>(message),
        ErrorCategory::MediaDecode => PyErr::new::<MediaDecodeError, _>(message),
        ErrorCategory::MediaGeometry => PyErr::new::<MediaGeometryError, _>(message),
        ErrorCategory::ResourceLimit => PyErr::new::<ResourceLimitError, _>(message),
        ErrorCategory::ArithmeticOverflow => PyErr::new::<ArithmeticOverflowError, _>(message),
        ErrorCategory::DestinationTooSmall => PyErr::new::<DestinationTooSmallError, _>(message),
        ErrorCategory::InternalInvariant => PyErr::new::<InternalInvariantError, _>(message),
        _ => PyErr::new::<QwenMMError, _>(message),
    };
    let context = PyDict::new(py);
    for (key, value) in error.context() {
        if let Err(error) = set_diagnostic(&context, key, value) {
            return error;
        }
    }
    let value = python_error.value(py);
    if let Err(error) = value.setattr("category", category.as_str()) {
        return error;
    }
    if let Err(error) = value.setattr("context", context) {
        return error;
    }
    python_error
}

pub(crate) fn memory_error(name: &'static str, elements: usize) -> PyErr {
    PyMemoryError::new_err(format!(
        "could not allocate native {name} destination with {elements} elements"
    ))
}

fn set_diagnostic(context: &Bound<'_, PyDict>, key: &str, value: &DiagnosticValue) -> PyResult<()> {
    match value {
        DiagnosticValue::Text(value) => context.set_item(key, value),
        DiagnosticValue::Integer(value) => context.set_item(key, value),
        DiagnosticValue::Unsigned(value) => context.set_item(key, value),
        DiagnosticValue::Float(value) => context.set_item(key, value),
        DiagnosticValue::Boolean(value) => context.set_item(key, value),
        _ => context.set_item(key, value.to_string()),
    }
}

fn add_exception<T>(
    module: &Bound<'_, PyModule>,
    name: &'static str,
    category: Option<&'static str>,
) -> PyResult<()>
where
    T: PyTypeInfo,
{
    let exception = module.py().get_type::<T>();
    if let Some(category) = category {
        exception.setattr("category", category)?;
    }
    module.add(name, exception)
}

pub(crate) fn add_exceptions(module: &Bound<'_, PyModule>) -> PyResult<()> {
    add_exception::<QwenMMError>(module, "QwenMMError", None)?;
    add_exception::<InvalidRequestError>(module, "InvalidRequestError", Some("invalid_request"))?;
    add_exception::<UnsupportedOptionError>(
        module,
        "UnsupportedOptionError",
        Some("unsupported_option"),
    )?;
    add_exception::<UnsupportedMediaError>(
        module,
        "UnsupportedMediaError",
        Some("unsupported_media"),
    )?;
    add_exception::<ProfileMismatchError>(
        module,
        "ProfileMismatchError",
        Some("profile_mismatch"),
    )?;
    add_exception::<MediaDecodeError>(module, "MediaDecodeError", Some("media_decode"))?;
    add_exception::<MediaGeometryError>(module, "MediaGeometryError", Some("media_geometry"))?;
    add_exception::<ResourceLimitError>(module, "ResourceLimitError", Some("resource_limit"))?;
    add_exception::<ArithmeticOverflowError>(
        module,
        "ArithmeticOverflowError",
        Some("arithmetic_overflow"),
    )?;
    add_exception::<DestinationTooSmallError>(
        module,
        "DestinationTooSmallError",
        Some("destination_too_small"),
    )?;
    add_exception::<InternalInvariantError>(
        module,
        "InternalInvariantError",
        Some("internal_invariant"),
    )?;
    Ok(())
}

pub(crate) fn py_detail(error: &PyErr) -> String {
    error.to_string()
}

pub(crate) fn type_name(value: &Bound<'_, PyAny>) -> String {
    value.get_type().name().map_or_else(
        |_| "<unknown>".to_owned(),
        |name| name.to_string_lossy().into_owned(),
    )
}

#[cfg(test)]
mod tests {
    use pyo3::Python;
    use pyo3::types::{PyAnyMethods, PyTypeMethods};
    use qwen_mm_core::{ErrorCategory, QwenError};

    use super::to_python_error;

    #[test]
    fn every_stable_category_maps_to_its_specific_exception() {
        Python::initialize();
        Python::attach(|py| {
            for (category, expected) in [
                (ErrorCategory::InvalidRequest, "InvalidRequestError"),
                (ErrorCategory::UnsupportedOption, "UnsupportedOptionError"),
                (ErrorCategory::UnsupportedMedia, "UnsupportedMediaError"),
                (ErrorCategory::ProfileMismatch, "ProfileMismatchError"),
                (ErrorCategory::MediaDecode, "MediaDecodeError"),
                (ErrorCategory::MediaGeometry, "MediaGeometryError"),
                (ErrorCategory::ResourceLimit, "ResourceLimitError"),
                (ErrorCategory::ArithmeticOverflow, "ArithmeticOverflowError"),
                (
                    ErrorCategory::DestinationTooSmall,
                    "DestinationTooSmallError",
                ),
                (ErrorCategory::InternalInvariant, "InternalInvariantError"),
            ] {
                let error = to_python_error(py, &QwenError::new(category, "test"));
                assert_eq!(
                    error
                        .value(py)
                        .get_type()
                        .name()
                        .expect("exception type name"),
                    expected
                );
            }
        });
    }
}
