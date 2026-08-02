//! Thin Python binding boundary for `qwen-mm-core`.

mod errors;
mod input;
mod output;

use std::{path::PathBuf, sync::Arc};

#[cfg(feature = "test-hooks")]
use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::prelude::*;
use qwen_mm_core::{ProfileRegistry, QwenImageProcessor, ResourceLimits};

use crate::{
    errors::{add_exceptions, memory_error, to_python_error},
    input::{OwnedRequest, parse_limits, parse_requests},
    output::{NativeBatch, PyPreparedBatch, RunError, run_batch},
};

/// Returns the native package version.
#[pyfunction]
fn native_version() -> &'static str {
    qwen_mm_core::version()
}

#[cfg(feature = "test-hooks")]
static TEST_NATIVE_BATCH_ACTIVE: AtomicBool = AtomicBool::new(false);

#[cfg(feature = "test-hooks")]
struct TestNativeBatchActiveGuard;

#[cfg(feature = "test-hooks")]
impl TestNativeBatchActiveGuard {
    fn enter() -> Self {
        TEST_NATIVE_BATCH_ACTIVE.store(true, Ordering::SeqCst);
        Self
    }
}

#[cfg(feature = "test-hooks")]
impl Drop for TestNativeBatchActiveGuard {
    fn drop(&mut self) {
        TEST_NATIVE_BATCH_ACTIVE.store(false, Ordering::SeqCst);
    }
}

/// Test-only signal that native batch execution is inside the detached window.
#[cfg(feature = "test-hooks")]
#[pyfunction]
fn _test_native_batch_active() -> bool {
    TEST_NATIVE_BATCH_ACTIVE.load(Ordering::SeqCst)
}

/// Profile-bound native processor exposed to Python.
#[pyclass(name = "Processor", module = "qwen_mm._native", frozen)]
struct PyProcessor {
    inner: Arc<QwenImageProcessor>,
    limits: ResourceLimits,
    supports_thinking: bool,
}

#[pymethods]
impl PyProcessor {
    /// Loads one hash-pinned profile from a local snapshot directory.
    #[new]
    #[pyo3(signature = (profile, assets_directory, *, limits=None))]
    fn new(
        py: Python<'_>,
        profile: &str,
        assets_directory: PathBuf,
        limits: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let limits = parse_limits(limits).map_err(|error| to_python_error(py, &error))?;
        let registry = ProfileRegistry::bundled().map_err(|error| to_python_error(py, &error))?;
        let profile = registry
            .resolve(profile)
            .map_err(|error| to_python_error(py, &error))?;
        let supports_thinking = profile.supports_thinking();
        let processor = QwenImageProcessor::from_local_assets(profile, assets_directory, limits)
            .map_err(|error| to_python_error(py, &error))?;
        Ok(Self {
            inner: Arc::new(processor),
            limits,
            supports_thinking,
        })
    }

    /// Prepares one heterogeneous batch in a single GIL-free native call.
    fn prepare_batch(
        &self,
        py: Python<'_>,
        requests: &Bound<'_, PyAny>,
    ) -> PyResult<PyPreparedBatch> {
        let requests = parse_requests(
            py,
            requests,
            self.limits,
            self.supports_thinking,
            self.inner.profile().alias.as_str(),
        )
        .map_err(|error| to_python_error(py, &error))?;
        let processor = Arc::clone(&self.inner);
        let native = py
            .detach(move || {
                #[cfg(feature = "test-hooks")]
                let _active = TestNativeBatchActiveGuard::enter();
                run_batch(&processor, &requests)
            })
            .map_err(|error| match error {
                RunError::Core(error) => to_python_error(py, &error),
                RunError::Allocation { name, elements } => memory_error(name, elements),
            })?;
        PyPreparedBatch::from_native(py, native)
    }

    /// Exact public profile alias bound to this processor.
    #[getter]
    fn profile(&self) -> &str {
        self.inner.profile().alias.as_str()
    }

    /// Immutable compatibility profile fingerprint.
    #[getter]
    fn profile_fingerprint(&self) -> &str {
        &self.inner.profile().fingerprint
    }
}

fn assert_detached_types_are_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    fn assert_send<T: Send>() {}
    assert_send_sync::<QwenImageProcessor>();
    assert_send_sync::<Arc<QwenImageProcessor>>();
    assert_send::<Vec<OwnedRequest>>();
    assert_send::<NativeBatch>();
}

/// Defines the private native module imported by the Python package.
#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    assert_detached_types_are_send_sync();
    module.add("__version__", qwen_mm_core::version())?;
    module.add_function(wrap_pyfunction!(native_version, module)?)?;
    #[cfg(feature = "test-hooks")]
    module.add_function(wrap_pyfunction!(_test_native_batch_active, module)?)?;
    module.add_class::<PyProcessor>()?;
    module.add_class::<PyPreparedBatch>()?;
    add_exceptions(module)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use pyo3::Python;
    use pyo3::types::PyAnyMethods;

    use super::assert_detached_types_are_send_sync;
    use crate::errors::{invalid_request, to_python_error};

    #[test]
    fn detached_boundary_types_remain_send_and_sync() {
        assert_detached_types_are_send_sync();
    }

    #[test]
    fn typed_errors_keep_category_and_context() {
        Python::initialize();
        Python::attach(|py| {
            let error = invalid_request("bad schema").with_context("request_index", 3_usize);
            let python = to_python_error(py, &error);
            let value = python.value(py);
            assert_eq!(
                value
                    .getattr("category")
                    .expect("category")
                    .extract::<&str>()
                    .expect("string"),
                "invalid_request"
            );
            let context = value.getattr("context").expect("context");
            assert_eq!(
                context
                    .get_item("request_index")
                    .expect("context item")
                    .extract::<u64>()
                    .expect("integer"),
                3
            );
        });
    }
}
