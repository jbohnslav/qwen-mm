//! Thin Python binding boundary for `qwen-mm-core`.

use pyo3::prelude::*;

/// Returns the native package version.
#[pyfunction]
fn native_version() -> &'static str {
    qwen_mm_core::version()
}

/// Defines the private native module imported by the Python package.
#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", qwen_mm_core::version())?;
    module.add_function(wrap_pyfunction!(native_version, module)?)?;
    Ok(())
}
