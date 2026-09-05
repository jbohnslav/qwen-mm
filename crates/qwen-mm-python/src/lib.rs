//! Thin Python binding boundary for `qwen-mm-core`.

mod errors;
mod input;
mod output;

use std::{
    env,
    path::{Path, PathBuf},
    sync::Arc,
};

#[cfg(feature = "test-hooks")]
use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::{
    exceptions::{PyFileNotFoundError, PyValueError},
    prelude::*,
    types::{PyDict, PyList, PyType},
};
use qwen_mm_core::{
    ObservationRecorder, ObservationScope, ProcessorConfig, ProfileAlias, ProfileRegistry,
    QwenImageProcessor, ResourceLimits,
};

use crate::{
    errors::{add_exceptions, memory_error, to_python_error},
    input::{
        OwnedRequest, drop_owned_media, parse_limits, parse_requests, parse_requests_observed,
        parse_thread_budget,
    },
    output::{
        NativeBatch, PaddingSide, PyPreparedBatch, RunError, native_batch_bytes,
        observation_report_dict, run_batch, run_batch_observed,
    },
};

fn parse_padding_side(value: &str) -> PyResult<PaddingSide> {
    match value {
        "left" => Ok(PaddingSide::Left),
        "right" => Ok(PaddingSide::Right),
        _ => Err(PyValueError::new_err(
            "padding_side must be either 'left' or 'right'",
        )),
    }
}

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

fn non_empty_environment_path(name: &str) -> Option<PathBuf> {
    env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn default_huggingface_hub_cache() -> Option<PathBuf> {
    non_empty_environment_path("HF_HUB_CACHE")
        .or_else(|| non_empty_environment_path("HF_HOME").map(|path| path.join("hub")))
        .or_else(|| {
            non_empty_environment_path("XDG_CACHE_HOME")
                .map(|path| path.join("huggingface").join("hub"))
        })
        .or_else(|| {
            non_empty_environment_path("HOME")
                .map(|path| path.join(".cache").join("huggingface").join("hub"))
        })
}

fn huggingface_snapshot_directory(cache: &Path, model_id: &str, revision: &str) -> PathBuf {
    cache
        .join(format!("models--{}", model_id.replace('/', "--")))
        .join("snapshots")
        .join(revision)
}

#[pymethods]
impl PyProcessor {
    /// Decodes generated IDs without loading another tokenizer or processor.
    #[pyo3(signature = (token_ids, *, skip_special_tokens=false))]
    fn decode(
        &self,
        py: Python<'_>,
        token_ids: Vec<u32>,
        skip_special_tokens: bool,
    ) -> PyResult<String> {
        let processor = Arc::clone(&self.inner);
        py.detach(move || processor.decode(&token_ids, skip_special_tokens))
            .map_err(|error| to_python_error(py, &error))
    }

    /// Loads one hash-pinned profile from a local snapshot directory.
    #[new]
    #[pyo3(signature = (profile, assets_directory, *, limits=None, thread_budget=None))]
    fn new(
        py: Python<'_>,
        profile: &str,
        assets_directory: PathBuf,
        limits: Option<&Bound<'_, PyAny>>,
        thread_budget: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let limits = parse_limits(limits).map_err(|error| to_python_error(py, &error))?;
        let thread_budget =
            parse_thread_budget(thread_budget).map_err(|error| to_python_error(py, &error))?;
        let registry = ProfileRegistry::bundled().map_err(|error| to_python_error(py, &error))?;
        let profile = registry
            .resolve(profile)
            .map_err(|error| to_python_error(py, &error))?;
        let supports_thinking = profile.supports_thinking();
        let processor = QwenImageProcessor::from_local_assets_with_config(
            profile,
            assets_directory,
            limits,
            ProcessorConfig::with_thread_budget(thread_budget),
        )
        .map_err(|error| to_python_error(py, &error))?;
        Ok(Self {
            inner: Arc::new(processor),
            limits,
            supports_thinking,
        })
    }

    /// Loads this profile's exact pinned snapshot from a local Hugging Face
    /// hub cache. This performs no network access or implicit download.
    #[classmethod]
    #[pyo3(signature = (profile, *, cache_directory=None, limits=None, thread_budget=None))]
    fn from_huggingface_cache(
        _class: &Bound<'_, PyType>,
        py: Python<'_>,
        profile: &str,
        cache_directory: Option<PathBuf>,
        limits: Option<&Bound<'_, PyAny>>,
        thread_budget: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let registry = ProfileRegistry::bundled().map_err(|error| to_python_error(py, &error))?;
        let resolved = registry
            .resolve(profile)
            .map_err(|error| to_python_error(py, &error))?;
        let cache = cache_directory
            .or_else(default_huggingface_hub_cache)
            .ok_or_else(|| {
                PyFileNotFoundError::new_err(concat!(
                    "cannot resolve the Hugging Face hub cache; set HF_HUB_CACHE or pass ",
                    "cache_directory=",
                ))
            })?;
        let snapshot =
            huggingface_snapshot_directory(&cache, &resolved.model_id, &resolved.revision);
        if !snapshot.is_dir() {
            return Err(PyFileNotFoundError::new_err(format!(
                concat!(
                    "pinned snapshot not found at {}; download it with ",
                    "`hf download {} --revision {}` or pass the Hugging Face hub cache root ",
                    "as cache_directory=",
                ),
                snapshot.display(),
                resolved.model_id,
                resolved.revision,
            )));
        }
        Self::new(py, profile, snapshot, limits, thread_budget)
    }

    /// Frozen profiles accepted by this wheel.
    #[staticmethod]
    fn supported_profiles(py: Python<'_>) -> PyResult<Py<PyAny>> {
        let registry = ProfileRegistry::bundled().map_err(|error| to_python_error(py, &error))?;
        let profiles = PyList::empty(py);
        for alias in [ProfileAlias::Qwen3Vl8b, ProfileAlias::Qwen35_9b] {
            let profile = registry.get(alias);
            let value = PyDict::new(py);
            value.set_item("profile", profile.alias.as_str())?;
            value.set_item("model_id", &profile.model_id)?;
            value.set_item("revision", &profile.revision)?;
            value.set_item("fingerprint", &profile.fingerprint)?;
            profiles.append(value)?;
        }
        Ok(profiles.into_any().unbind())
    }

    /// Prepares one heterogeneous batch in a single GIL-free native call.
    /// Right padding is the default; pass `padding_side="left"` for batched
    /// generation, matching the official tokenizer examples.
    #[pyo3(signature = (requests, *, padding_side="right"))]
    fn prepare_batch(
        &self,
        py: Python<'_>,
        requests: &Bound<'_, PyAny>,
        padding_side: &str,
    ) -> PyResult<PyPreparedBatch> {
        let padding_side = parse_padding_side(padding_side)?;
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
                run_batch(&processor, &requests, padding_side)
            })
            .map_err(|error| match error {
                RunError::Core(error) => to_python_error(py, &error),
                RunError::Allocation { name, elements } => memory_error(name, elements),
            })?;
        PyPreparedBatch::from_native(py, native)
    }

    /// Prepares one batch and returns the same arrays plus a bounded JSON-ready
    /// whole-operation observation report. The normal `prepare_batch` path
    /// remains completely uninstrumented.
    #[pyo3(signature = (requests, *, event_capacity=4096, padding_side="right"))]
    fn prepare_batch_observed(
        &self,
        py: Python<'_>,
        requests: &Bound<'_, PyAny>,
        event_capacity: usize,
        padding_side: &str,
    ) -> PyResult<(PyPreparedBatch, Py<PyAny>)> {
        let padding_side = parse_padding_side(padding_side)?;
        let mut recorder = ObservationRecorder::new(event_capacity);
        recorder.calls_mut().public_python_calls = 1;
        let operation_span =
            recorder.begin("binding.prepare_batch", ObservationScope::default(), 0);
        let parse_span = recorder.begin("binding.parse_requests", ObservationScope::default(), 0);
        let parsed = parse_requests_observed(
            py,
            requests,
            self.limits,
            self.supports_thinking,
            self.inner.profile().alias.as_str(),
            &mut recorder,
        );
        let requests = match parsed {
            Ok(requests) => {
                recorder.finish_success(parse_span, 0, &[requests.len() as u64]);
                requests
            }
            Err(error) => {
                recorder.finish_error(parse_span, &error);
                recorder.discard_retained_outputs();
                recorder.release_all_transients();
                recorder.finish_error(operation_span, &error);
                let python = to_python_error(py, &error);
                return Err(attach_observation_report(py, python, &recorder.report()));
            }
        };
        recorder.calls_mut().native_batch_calls = 1;
        let processor = Arc::clone(&self.inner);
        let (native, mut recorder) = py.detach(move || {
            #[cfg(feature = "test-hooks")]
            let _active = TestNativeBatchActiveGuard::enter();
            let result = run_batch_observed(&processor, &requests, padding_side, &mut recorder);
            drop_owned_media(requests, &mut recorder);
            recorder.release_all_transients();
            (result, recorder)
        });
        let native = match native {
            Ok(native) => native,
            Err(RunError::Core(error)) => {
                recorder.discard_retained_outputs();
                recorder.release_all_transients();
                recorder.finish_error(operation_span, &error);
                let python = to_python_error(py, &error);
                return Err(attach_observation_report(py, python, &recorder.report()));
            }
            Err(RunError::Allocation { name, elements }) => {
                recorder.discard_retained_outputs();
                recorder.release_all_transients();
                recorder.finish_error_category(operation_span, "memory_error");
                let python = memory_error(name, elements);
                return Err(attach_observation_report(py, python, &recorder.report()));
            }
        };
        let output_bytes = native_batch_bytes(&native);
        let materialize_span = recorder.begin(
            "binding.numpy.materialize",
            ObservationScope::default(),
            output_bytes,
        );
        let prepared = match PyPreparedBatch::from_native(py, native) {
            Ok(prepared) => prepared,
            Err(error) => {
                recorder.finish_error_category(materialize_span, "python_conversion");
                recorder.discard_retained_outputs();
                recorder.release_all_transients();
                recorder.finish_error_category(operation_span, "python_conversion");
                return Err(attach_observation_report(py, error, &recorder.report()));
            }
        };
        recorder.finish_success(materialize_span, output_bytes, &[output_bytes]);
        recorder.finish_success(operation_span, output_bytes, &[output_bytes]);
        let report = recorder.report();
        let report = observation_report_dict(py, &report)?.into_any().unbind();
        Ok((prepared, report))
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

    /// Exact upstream Hugging Face model identifier.
    #[getter]
    fn model_id(&self) -> &str {
        &self.inner.profile().model_id
    }

    /// Immutable upstream repository revision required by this profile.
    #[getter]
    fn revision(&self) -> &str {
        &self.inner.profile().revision
    }

    /// Exact total native-worker budget owned by this processor.
    #[getter]
    fn thread_budget(&self) -> usize {
        self.inner.thread_budget()
    }
}

fn attach_observation_report(
    py: Python<'_>,
    error: PyErr,
    report: &qwen_mm_core::ObservationReport,
) -> PyErr {
    if let Ok(value) = observation_report_dict(py, report) {
        let _ = error.value(py).setattr("observation_report", value);
    }
    error
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
