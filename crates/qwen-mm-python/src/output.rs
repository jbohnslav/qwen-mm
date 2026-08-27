//! Native execution and zero-copy `NumPy` result construction.

use std::sync::Arc;

use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyUntypedArrayMethods, ndarray::Array2};
use pyo3::{
    exceptions::PyKeyError,
    prelude::*,
    types::{PyDict, PyList},
};
use qwen_mm_core::{
    ArrayCapacity, BatchCapacities, BatchDestinations, BatchImageLayout, BatchPlan,
    BatchRequestLayout, BufferClass, ContentItem, CoordinateRange, FunctionCall, ImageInput,
    ImageSidecar, IntegrationSidecar, Message, MessageContent, ObservationRecorder,
    ObservationReport, ObservationScope, PreparedTextRequest, ProcessedBatchImageOccurrence,
    QwenError, QwenImageProcessor, Request, Rgb8, StageOutcome, TextReplacement, ToolCall,
    ToolDefinition, VisualModality,
};

use crate::{
    errors::python_conversion_error,
    input::{OwnedContentItem, OwnedImage, OwnedMessageContent, OwnedRequest},
};

#[derive(Debug)]
pub(crate) enum RunError {
    Core(QwenError),
    Allocation { name: &'static str, elements: usize },
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum PaddingSide {
    Left,
    #[default]
    Right,
}

impl PaddingSide {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Left => "left",
            Self::Right => "right",
        }
    }
}

impl From<QwenError> for RunError {
    fn from(error: QwenError) -> Self {
        Self::Core(error)
    }
}

#[derive(Debug)]
pub(crate) struct NativeMatrix<T> {
    pub(crate) shape: [usize; 2],
    pub(crate) data: Vec<T>,
    pub(crate) allocation_ptr: usize,
}

#[derive(Debug)]
pub(crate) struct NativeBatch {
    pub(crate) contract_id: String,
    pub(crate) profile_fingerprint: String,
    pub(crate) text: Vec<PreparedTextRequest>,
    pub(crate) sidecar: IntegrationSidecar,
    pub(crate) images: Vec<ProcessedBatchImageOccurrence>,
    pub(crate) request_layouts: Vec<BatchRequestLayout>,
    pub(crate) image_layouts: Vec<BatchImageLayout>,
    pub(crate) padding_side: PaddingSide,
    pub(crate) input_ids: NativeMatrix<i64>,
    pub(crate) attention_mask: NativeMatrix<i64>,
    pub(crate) mm_token_type_ids: NativeMatrix<i64>,
    pub(crate) pixel_values: Option<NativeMatrix<f32>>,
    pub(crate) image_grid_thw: Option<NativeMatrix<i64>>,
}

struct AllocatedBatchDestinations {
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
    mm_token_type_ids: Vec<i64>,
    pixel_values: Option<Vec<f32>>,
    image_grid_thw: Option<Vec<i64>>,
}

struct ExecutedBatchMetadata {
    contract_id: String,
    profile_fingerprint: String,
    text: Vec<PreparedTextRequest>,
    sidecar: IntegrationSidecar,
    images: Vec<ProcessedBatchImageOccurrence>,
}

impl AllocatedBatchDestinations {
    fn into_native_batch(
        self,
        capacities: BatchCapacities,
        metadata: ExecutedBatchMetadata,
        request_layouts: Vec<BatchRequestLayout>,
        image_layouts: Vec<BatchImageLayout>,
        padding_side: PaddingSide,
    ) -> NativeBatch {
        let Self {
            input_ids,
            attention_mask,
            mm_token_type_ids,
            pixel_values,
            image_grid_thw,
        } = self;
        let mut batch = NativeBatch {
            contract_id: metadata.contract_id,
            profile_fingerprint: metadata.profile_fingerprint,
            text: metadata.text,
            sidecar: metadata.sidecar,
            images: metadata.images,
            request_layouts,
            image_layouts,
            padding_side,
            input_ids: native_matrix(input_ids, capacities.input_ids),
            attention_mask: native_matrix(attention_mask, capacities.attention_mask),
            mm_token_type_ids: native_matrix(mm_token_type_ids, capacities.mm_token_type_ids),
            pixel_values: pixel_values
                .zip(capacities.pixel_values)
                .map(|(values, capacity)| native_matrix(values, capacity)),
            image_grid_thw: image_grid_thw
                .zip(capacities.image_grid_thw)
                .map(|(values, capacity)| native_matrix(values, capacity)),
        };
        batch.apply_padding_side();
        batch
    }
}

impl NativeBatch {
    fn apply_padding_side(&mut self) {
        if self.padding_side != PaddingSide::Left {
            return;
        }
        for layout in &self.request_layouts {
            let padding = layout.right_padding;
            if padding == 0 {
                continue;
            }
            for row in [
                &mut self.input_ids.data,
                &mut self.attention_mask.data,
                &mut self.mm_token_type_ids.data,
            ] {
                row[layout.text_elements.start..layout.text_elements.end].rotate_right(padding);
            }
        }
    }
}

fn native_matrix<T>(data: Vec<T>, capacity: ArrayCapacity) -> NativeMatrix<T> {
    let allocation_ptr = data.as_ptr() as usize;
    NativeMatrix {
        shape: capacity.shape,
        data,
        allocation_ptr,
    }
}

pub(crate) fn native_batch_bytes(batch: &NativeBatch) -> u64 {
    let mut total = matrix_bytes(&batch.input_ids)
        .saturating_add(matrix_bytes(&batch.attention_mask))
        .saturating_add(matrix_bytes(&batch.mm_token_type_ids));
    if let Some(values) = &batch.pixel_values {
        total = total.saturating_add(matrix_bytes(values));
    }
    if let Some(values) = &batch.image_grid_thw {
        total = total.saturating_add(matrix_bytes(values));
    }
    total
}

fn matrix_bytes<T>(matrix: &NativeMatrix<T>) -> u64 {
    let elements = matrix.shape[0].saturating_mul(matrix.shape[1]);
    u64::try_from(elements.saturating_mul(std::mem::size_of::<T>())).unwrap_or(u64::MAX)
}

/// One completed batch with official arrays separated from adapter metadata.
#[pyclass(name = "PreparedBatch", module = "qwen_mm._native", frozen)]
pub(crate) struct PyPreparedBatch {
    arrays: Py<PyDict>,
    metadata: Py<PyDict>,
    official_keys: Vec<String>,
}

#[pymethods]
impl PyPreparedBatch {
    /// Official conditional processor outputs in frozen key order.
    #[getter]
    fn arrays(&self, py: Python<'_>) -> Py<PyDict> {
        self.arrays.clone_ref(py)
    }

    /// Adapter-only contract, prompt, occurrence, range, and cache metadata.
    #[getter]
    fn metadata(&self, py: Python<'_>) -> Py<PyDict> {
        self.metadata.clone_ref(py)
    }

    /// Official keys in the same order as `.arrays` iteration.
    fn official_keys(&self) -> Vec<String> {
        self.official_keys.clone()
    }

    /// Official model array for `key`, matching `.arrays[key]`.
    fn __getitem__(&self, py: Python<'_>, key: Py<PyAny>) -> PyResult<Py<PyAny>> {
        self.arrays
            .bind(py)
            .get_item(key.bind(py))?
            .map(Bound::unbind)
            .ok_or_else(|| PyKeyError::new_err(key))
    }

    /// Iterates over official model-input keys without exposing metadata.
    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.arrays.bind(py).call_method0("__iter__")?.unbind())
    }

    fn __len__(&self) -> usize {
        self.official_keys.len()
    }

    fn __contains__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.arrays.bind(py).contains(key)
    }

    /// Official model-input keys in frozen processor order.
    fn keys(&self) -> Vec<String> {
        self.official_keys.clone()
    }

    /// Official model-input values in frozen processor order.
    fn values(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.arrays.bind(py).call_method0("values")?.unbind())
    }

    /// Official model-input `(key, value)` pairs in frozen processor order.
    fn items(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.arrays.bind(py).call_method0("items")?.unbind())
    }

    /// Returns an official model array, or `default` when the key is absent.
    #[pyo3(signature = (key, default=None))]
    fn get(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        default: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        Ok(self
            .arrays
            .bind(py)
            .get_item(key)?
            .map_or_else(|| default.unwrap_or_else(|| py.None()), Bound::unbind))
    }

    fn __repr__(&self) -> String {
        format!("PreparedBatch(keys={:?})", self.official_keys)
    }
}

pub(crate) fn observation_report_dict<'py>(
    py: Python<'py>,
    report: &ObservationReport,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    value.set_item("schema_version", &report.schema_version)?;
    value.set_item("event_capacity", report.event_capacity)?;
    value.set_item("dropped_events", report.dropped_events)?;
    value.set_item("outcome", outcome_name(report.outcome))?;
    value.set_item("error_category", &report.error_category)?;
    value.set_item("duration_ns", report.duration_ns)?;

    let spans = PyList::empty(py);
    for span in &report.spans {
        let item = PyDict::new(py);
        item.set_item("sequence", span.sequence)?;
        item.set_item("parent_sequence", span.parent_sequence)?;
        item.set_item("name", &span.name)?;
        item.set_item("scope", observation_scope_dict(py, span.scope)?)?;
        item.set_item("started_ns", span.started_ns)?;
        item.set_item("duration_ns", span.duration_ns)?;
        item.set_item("exclusive_duration_ns", span.exclusive_duration_ns)?;
        item.set_item("outcome", outcome_name(span.outcome))?;
        item.set_item("error_category", &span.error_category)?;
        item.set_item("input_bytes", span.input_bytes)?;
        item.set_item("output_bytes", span.output_bytes)?;
        item.set_item("shape", &span.shape)?;
        spans.append(item)?;
    }
    value.set_item("spans", spans)?;

    let buffers = PyList::empty(py);
    for buffer in &report.buffers {
        let item = PyDict::new(py);
        item.set_item("sequence", buffer.sequence)?;
        item.set_item("name", &buffer.name)?;
        item.set_item(
            "class",
            match buffer.class {
                BufferClass::RetainedOutput => "retained_output",
                BufferClass::DiscardedOutput => "discarded_output",
                BufferClass::Transient => "transient",
            },
        )?;
        item.set_item("scope", observation_scope_dict(py, buffer.scope)?)?;
        item.set_item("bytes", buffer.bytes)?;
        item.set_item("allocated_at_ns", buffer.allocated_at_ns)?;
        item.set_item("released_at_ns", buffer.released_at_ns)?;
        buffers.append(item)?;
    }
    value.set_item("buffers", buffers)?;

    let copies = PyList::empty(py);
    for copy in &report.copies {
        let item = PyDict::new(py);
        item.set_item("sequence", copy.sequence)?;
        item.set_item("name", &copy.name)?;
        item.set_item("scope", observation_scope_dict(py, copy.scope)?)?;
        item.set_item("bytes", copy.bytes)?;
        copies.append(item)?;
    }
    value.set_item("copies", copies)?;

    let allocations = PyDict::new(py);
    allocations.set_item("allocation_count", report.allocations.allocation_count)?;
    allocations.set_item("allocated_bytes", report.allocations.allocated_bytes)?;
    allocations.set_item("copy_count", report.allocations.copy_count)?;
    allocations.set_item("copied_bytes", report.allocations.copied_bytes)?;
    allocations.set_item(
        "transient_live_bytes",
        report.allocations.transient_live_bytes,
    )?;
    allocations.set_item(
        "peak_transient_live_bytes",
        report.allocations.peak_transient_live_bytes,
    )?;
    allocations.set_item(
        "retained_final_output_bytes",
        report.allocations.retained_final_output_bytes,
    )?;
    value.set_item("allocations", allocations)?;

    let calls = PyDict::new(py);
    calls.set_item("public_python_calls", report.calls.public_python_calls)?;
    calls.set_item("native_batch_calls", report.calls.native_batch_calls)?;
    calls.set_item("native_visual_calls", report.calls.native_visual_calls)?;
    calls.set_item("python_callbacks", report.calls.python_callbacks)?;
    calls.set_item("hugging_face_calls", report.calls.hugging_face_calls)?;
    calls.set_item("qwen_vl_utils_calls", report.calls.qwen_vl_utils_calls)?;
    calls.set_item("pillow_calls", report.calls.pillow_calls)?;
    calls.set_item("torchvision_calls", report.calls.torchvision_calls)?;
    value.set_item("calls", calls)?;
    value.set_item("counter_scope", &report.counter_scope)?;
    Ok(value)
}

fn observation_scope_dict(py: Python<'_>, scope: ObservationScope) -> PyResult<Bound<'_, PyDict>> {
    let value = PyDict::new(py);
    value.set_item("request_index", scope.request_index)?;
    value.set_item("message_index", scope.message_index)?;
    value.set_item("content_item_index", scope.content_item_index)?;
    value.set_item("media_index", scope.media_index)?;
    value.set_item("input_index", scope.input_index)?;
    Ok(value)
}

const fn outcome_name(outcome: StageOutcome) -> &'static str {
    match outcome {
        StageOutcome::Success => "success",
        StageOutcome::Error => "error",
    }
}

impl PyPreparedBatch {
    pub(crate) fn from_native(py: Python<'_>, native: NativeBatch) -> PyResult<Self> {
        let padding_side = native.padding_side;
        let arrays = PyDict::new(py);
        let input_ids = matrix_i64(py, native.input_ids)
            .map_err(|error| crate::errors::to_python_error(py, &error))?;
        let attention_mask = matrix_i64(py, native.attention_mask)
            .map_err(|error| crate::errors::to_python_error(py, &error))?;
        let mm_token_type_ids = matrix_i64(py, native.mm_token_type_ids)
            .map_err(|error| crate::errors::to_python_error(py, &error))?;
        arrays.set_item("input_ids", input_ids)?;
        arrays.set_item("attention_mask", attention_mask)?;
        arrays.set_item("mm_token_type_ids", mm_token_type_ids)?;
        let mut official_keys = vec![
            "input_ids".to_owned(),
            "attention_mask".to_owned(),
            "mm_token_type_ids".to_owned(),
        ];
        if let Some(pixel_values) = native.pixel_values {
            arrays.set_item(
                "pixel_values",
                matrix_f32(py, pixel_values)
                    .map_err(|error| crate::errors::to_python_error(py, &error))?,
            )?;
            official_keys.push("pixel_values".to_owned());
        }
        if let Some(image_grid_thw) = native.image_grid_thw {
            arrays.set_item(
                "image_grid_thw",
                matrix_i64(py, image_grid_thw)
                    .map_err(|error| crate::errors::to_python_error(py, &error))?,
            )?;
            official_keys.push("image_grid_thw".to_owned());
        }
        let metadata = build_metadata(
            py,
            &native.contract_id,
            &native.profile_fingerprint,
            &native.text,
            &native.sidecar,
            &native.images,
            &native.request_layouts,
            &native.image_layouts,
            padding_side,
        )?;
        Ok(Self {
            arrays: arrays.unbind(),
            metadata: metadata.unbind(),
            official_keys,
        })
    }
}

pub(crate) fn run_batch(
    processor: &Arc<QwenImageProcessor>,
    owned: &[OwnedRequest],
    padding_side: PaddingSide,
) -> Result<NativeBatch, RunError> {
    run_batch_internal(processor, owned, padding_side, None)
}

pub(crate) fn run_batch_observed(
    processor: &Arc<QwenImageProcessor>,
    owned: &[OwnedRequest],
    padding_side: PaddingSide,
    recorder: &mut ObservationRecorder,
) -> Result<NativeBatch, RunError> {
    run_batch_internal(processor, owned, padding_side, Some(recorder))
}

fn run_batch_internal(
    processor: &Arc<QwenImageProcessor>,
    owned: &[OwnedRequest],
    padding_side: PaddingSide,
    mut recorder: Option<&mut ObservationRecorder>,
) -> Result<NativeBatch, RunError> {
    let plan = plan_owned_requests(processor, owned, recorder.as_deref_mut())?;
    let capacities = *plan.capacities();
    let allocation_span = recorder.as_deref_mut().map(|recorder| {
        recorder.begin(
            "binding.destination.allocate",
            ObservationScope::default(),
            0,
        )
    });
    let (mut destinations, allocated_bytes) =
        match allocate_batch_destinations(&capacities, recorder.as_deref_mut()) {
            Ok(values) => values,
            Err(error) => {
                match recorder {
                    Some(recorder) => {
                        if let Some(span) = allocation_span {
                            recorder.finish_error_category(span, "memory_error");
                        }
                        recorder.discard_retained_outputs();
                        plan.drop_observed(recorder);
                    }
                    None => drop(plan),
                }
                return Err(error);
            }
        };
    if let (Some(recorder), Some(span)) = (recorder.as_deref_mut(), allocation_span) {
        recorder.finish_success(span, allocated_bytes, &[allocated_bytes]);
    }
    let metadata =
        match execute_batch_plan(processor, &plan, &mut destinations, recorder.as_deref_mut()) {
            Ok(values) => values,
            Err(error) => {
                match recorder {
                    Some(recorder) => {
                        recorder.discard_retained_outputs();
                        plan.drop_observed(recorder);
                    }
                    None => drop(plan),
                }
                return Err(error);
            }
        };
    let request_layouts = plan.request_layouts().to_vec();
    let image_layouts = plan.image_layouts().to_vec();
    match recorder {
        Some(recorder) => plan.drop_observed(recorder),
        None => drop(plan),
    }
    Ok(destinations.into_native_batch(
        capacities,
        metadata,
        request_layouts,
        image_layouts,
        padding_side,
    ))
}

fn plan_owned_requests(
    processor: &QwenImageProcessor,
    owned: &[OwnedRequest],
    recorder: Option<&mut ObservationRecorder>,
) -> Result<BatchPlan, RunError> {
    let image_inputs = owned
        .iter()
        .map(|request| request.images.iter().map(borrow_image).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let content_items = owned
        .iter()
        .map(|request| {
            request
                .messages
                .iter()
                .map(|message| match &message.content {
                    OwnedMessageContent::Text(_) => Vec::new(),
                    OwnedMessageContent::Items(items) => {
                        items.iter().map(borrow_content_item).collect()
                    }
                })
                .collect::<Vec<Vec<_>>>()
        })
        .collect::<Vec<_>>();
    let tool_calls = owned
        .iter()
        .map(|request| {
            request
                .messages
                .iter()
                .map(|message| message.tool_calls.iter().map(borrow_tool_call).collect())
                .collect::<Vec<Vec<_>>>()
        })
        .collect::<Vec<_>>();
    let tool_definitions = owned
        .iter()
        .map(|request| {
            request
                .options
                .tools_json
                .iter()
                .map(|json| ToolDefinition { json })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let messages = owned
        .iter()
        .enumerate()
        .map(|(request_index, request)| {
            request
                .messages
                .iter()
                .enumerate()
                .map(|(message_index, message)| Message {
                    role: message.role,
                    content: match &message.content {
                        OwnedMessageContent::Text(text) => MessageContent::Text(text),
                        OwnedMessageContent::Items(_) => {
                            MessageContent::Items(&content_items[request_index][message_index])
                        }
                    },
                    tool_calls: &tool_calls[request_index][message_index],
                    reasoning_content: message.reasoning_content.as_deref(),
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let requests = owned
        .iter()
        .enumerate()
        .map(|(request_index, request)| Request {
            messages: &messages[request_index],
            images: &image_inputs[request_index],
            videos: &[],
            options: request.options.borrowed(&tool_definitions[request_index]),
        })
        .collect::<Vec<_>>();

    let plan = if let Some(recorder) = recorder {
        processor.plan_batch_observed(&requests, recorder)?
    } else {
        processor.plan_batch(&requests)?
    };
    Ok(plan)
}

fn allocate_batch_destinations(
    capacities: &BatchCapacities,
    mut recorder: Option<&mut ObservationRecorder>,
) -> Result<(AllocatedBatchDestinations, u64), RunError> {
    let mut allocated_bytes = 0_u64;
    let input_ids = allocate_destination(
        "input_ids",
        capacities.input_ids,
        recorder.as_deref_mut(),
        &mut allocated_bytes,
    )?;
    let attention_mask = allocate_destination(
        "attention_mask",
        capacities.attention_mask,
        recorder.as_deref_mut(),
        &mut allocated_bytes,
    )?;
    let mm_token_type_ids = allocate_destination(
        "mm_token_type_ids",
        capacities.mm_token_type_ids,
        recorder.as_deref_mut(),
        &mut allocated_bytes,
    )?;
    let pixel_values = capacities
        .pixel_values
        .map(|capacity| {
            allocate_destination(
                "pixel_values",
                capacity,
                recorder.as_deref_mut(),
                &mut allocated_bytes,
            )
        })
        .transpose()?;
    let image_grid_thw = capacities
        .image_grid_thw
        .map(|capacity| {
            allocate_destination("image_grid_thw", capacity, recorder, &mut allocated_bytes)
        })
        .transpose()?;
    Ok((
        AllocatedBatchDestinations {
            input_ids,
            attention_mask,
            mm_token_type_ids,
            pixel_values,
            image_grid_thw,
        },
        allocated_bytes,
    ))
}

fn allocate_destination<T: Default + Clone>(
    name: &'static str,
    capacity: ArrayCapacity,
    recorder: Option<&mut ObservationRecorder>,
    allocated_bytes: &mut u64,
) -> Result<Vec<T>, RunError> {
    let values = allocate::<T>(name, capacity.elements)?;
    record_destination_allocation(recorder, name, capacity.bytes);
    *allocated_bytes = allocated_bytes.saturating_add(capacity.bytes);
    Ok(values)
}

fn execute_batch_plan(
    processor: &QwenImageProcessor,
    plan: &BatchPlan,
    destinations: &mut AllocatedBatchDestinations,
    recorder: Option<&mut ObservationRecorder>,
) -> Result<ExecutedBatchMetadata, RunError> {
    let native_destinations = BatchDestinations {
        input_ids: &mut destinations.input_ids,
        attention_mask: &mut destinations.attention_mask,
        mm_token_type_ids: &mut destinations.mm_token_type_ids,
        pixel_values: destinations.pixel_values.as_deref_mut(),
        image_grid_thw: destinations.image_grid_thw.as_deref_mut(),
        pixel_values_videos: None,
        video_grid_thw: None,
    };
    let view = if let Some(recorder) = recorder {
        processor.execute_plan_into_observed(plan, native_destinations, recorder)?
    } else {
        processor.execute_plan_into(plan, native_destinations)?
    };
    Ok(ExecutedBatchMetadata {
        contract_id: view.contract_id.to_owned(),
        profile_fingerprint: view.profile_fingerprint.to_owned(),
        text: view.text.to_vec(),
        sidecar: view.sidecar().clone(),
        images: view.images.to_vec(),
    })
}

fn record_destination_allocation(
    recorder: Option<&mut ObservationRecorder>,
    name: &'static str,
    bytes: u64,
) {
    if let Some(recorder) = recorder {
        recorder.record_allocation(
            name,
            BufferClass::RetainedOutput,
            ObservationScope::default(),
            bytes,
        );
    }
}

fn borrow_image(image: &OwnedImage) -> ImageInput<'_> {
    match image {
        OwnedImage::Encoded { data, format } => ImageInput::Encoded {
            data,
            format: *format,
        },
        OwnedImage::Rgb8 {
            data,
            height,
            width,
            row_stride,
        } => ImageInput::Rgb8(Rgb8 {
            data,
            height: *height,
            width: *width,
            row_stride: *row_stride,
        }),
    }
}

fn borrow_content_item(item: &OwnedContentItem) -> ContentItem<'_> {
    match item {
        OwnedContentItem::Text(text) => ContentItem::Text(text),
        OwnedContentItem::Image {
            input_index,
            options,
        } => ContentItem::Image(qwen_mm_core::ImageRef {
            input_index: *input_index,
            options: *options,
        }),
        OwnedContentItem::Video { input_index } => ContentItem::Video(qwen_mm_core::VideoRef {
            input_index: *input_index,
            options: qwen_mm_core::VideoOptions::default(),
        }),
    }
}

fn borrow_tool_call(call: &crate::input::OwnedToolCall) -> ToolCall<'_> {
    let function_call = FunctionCall {
        name: &call.name,
        arguments_json: &call.arguments_json,
    };
    if call.wrapped {
        ToolCall::Function {
            id: call.id.as_deref(),
            function: function_call,
        }
    } else {
        ToolCall::Direct {
            id: call.id.as_deref(),
            call: function_call,
        }
    }
}

fn allocate<T: Default + Clone>(name: &'static str, elements: usize) -> Result<Vec<T>, RunError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(elements)
        .map_err(|_| RunError::Allocation { name, elements })?;
    output.resize(elements, T::default());
    Ok(output)
}

fn matrix_i64(
    py: Python<'_>,
    matrix: NativeMatrix<i64>,
) -> Result<Bound<'_, PyArray2<i64>>, QwenError> {
    validate_numpy_shape(matrix.shape)?;
    let expected_ptr = matrix.allocation_ptr;
    let array = Array2::from_shape_vec(matrix.shape, matrix.data)
        .map_err(|error| python_conversion_error(error.to_string()))?
        .into_pyarray(py);
    debug_assert_eq!(array.data() as usize, expected_ptr);
    debug_assert!(array.is_c_contiguous());
    Ok(array)
}

fn matrix_f32(
    py: Python<'_>,
    matrix: NativeMatrix<f32>,
) -> Result<Bound<'_, PyArray2<f32>>, QwenError> {
    validate_numpy_shape(matrix.shape)?;
    let expected_ptr = matrix.allocation_ptr;
    let array = Array2::from_shape_vec(matrix.shape, matrix.data)
        .map_err(|error| python_conversion_error(error.to_string()))?
        .into_pyarray(py);
    debug_assert_eq!(array.data() as usize, expected_ptr);
    debug_assert!(array.is_c_contiguous());
    Ok(array)
}

fn validate_numpy_shape(shape: [usize; 2]) -> Result<(), QwenError> {
    for (axis, dimension) in shape.into_iter().enumerate() {
        isize::try_from(dimension).map_err(|_| {
            QwenError::new(
                qwen_mm_core::ErrorCategory::ArithmeticOverflow,
                "array dimension does not fit NumPy npy_intp",
            )
            .with_context("axis", axis)
            .with_context("dimension", dimension)
        })?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn build_metadata<'py>(
    py: Python<'py>,
    contract_id: &str,
    profile_fingerprint: &str,
    text: &[PreparedTextRequest],
    sidecar: &IntegrationSidecar,
    images: &[ProcessedBatchImageOccurrence],
    request_layouts: &[BatchRequestLayout],
    image_layouts: &[BatchImageLayout],
    padding_side: PaddingSide,
) -> PyResult<Bound<'py, PyDict>> {
    let metadata = PyDict::new(py);
    metadata.set_item("contract_id", contract_id)?;
    metadata.set_item("profile_fingerprint", profile_fingerprint)?;
    metadata.set_item("padding_side", padding_side.as_str())?;
    let text_items = PyList::empty(py);
    for request in text {
        let item = PyDict::new(py);
        item.set_item("rendered_prompt", &request.rendered_prompt)?;
        item.set_item("expanded_prompt", &request.expanded_prompt)?;
        let replacements = PyList::empty(py);
        for replacement in &request.replacements {
            replacements.append(replacement_dict(py, *replacement)?)?;
        }
        item.set_item("replacements", replacements)?;
        text_items.append(item)?;
    }
    metadata.set_item("text", text_items)?;

    let sidecar_dict = PyDict::new(py);
    let sidecar_images = PyList::empty(py);
    for image in &sidecar.images {
        sidecar_images.append(sidecar_image_dict(py, image)?)?;
    }
    sidecar_dict.set_item("images", sidecar_images)?;
    sidecar_dict.set_item("videos", PyList::empty(py))?;
    metadata.set_item("sidecar", sidecar_dict)?;

    let processed_images = PyList::empty(py);
    for image in images {
        let item = PyDict::new(py);
        let occurrence = image.occurrence;
        item.set_item("request_index", occurrence.location.request_index)?;
        item.set_item("message_index", occurrence.location.message_index)?;
        item.set_item("content_item_index", occurrence.location.content_item_index)?;
        item.set_item("input_index", occurrence.location.input_index)?;
        item.set_item("grid_row", occurrence.grid_row)?;
        item.set_item(
            "pixel_rows",
            [occurrence.pixel_rows.start, occurrence.pixel_rows.end],
        )?;
        item.set_item("source_height", occurrence.source_height)?;
        item.set_item("source_width", occurrence.source_width)?;
        item.set_item("geometry", geometry_dict(py, occurrence.geometry)?)?;
        item.set_item("cache_key", image.cache_key_hex())?;
        processed_images.append(item)?;
    }
    metadata.set_item("images", processed_images)?;

    let requests = PyList::empty(py);
    for layout in request_layouts {
        let item = PyDict::new(py);
        item.set_item("request_index", layout.request_index)?;
        item.set_item(
            "text_elements",
            [layout.text_elements.start, layout.text_elements.end],
        )?;
        item.set_item("token_count", layout.token_count)?;
        let (left_padding, right_padding) = match padding_side {
            PaddingSide::Left => (layout.right_padding, 0),
            PaddingSide::Right => (0, layout.right_padding),
        };
        item.set_item("left_padding", left_padding)?;
        item.set_item("right_padding", right_padding)?;
        item.set_item(
            "image_grid_rows",
            [layout.image_grid_rows.start, layout.image_grid_rows.end],
        )?;
        item.set_item(
            "pixel_rows",
            [layout.pixel_rows.start, layout.pixel_rows.end],
        )?;
        requests.append(item)?;
    }
    metadata.set_item("request_layouts", requests)?;

    let layouts = PyList::empty(py);
    for layout in image_layouts {
        let item = PyDict::new(py);
        item.set_item("request_index", layout.location.request_index)?;
        item.set_item("message_index", layout.location.message_index)?;
        item.set_item("content_item_index", layout.location.content_item_index)?;
        item.set_item("input_index", layout.location.input_index)?;
        item.set_item("grid_row", layout.grid_row)?;
        item.set_item(
            "pixel_rows",
            [layout.pixel_rows.start, layout.pixel_rows.end],
        )?;
        item.set_item("geometry", geometry_dict(py, layout.geometry)?)?;
        item.set_item("cache_key", layout.cache_key_hex())?;
        layouts.append(item)?;
    }
    metadata.set_item("image_layouts", layouts)?;
    Ok(metadata)
}

fn replacement_dict(py: Python<'_>, replacement: TextReplacement) -> PyResult<Bound<'_, PyDict>> {
    let item = PyDict::new(py);
    item.set_item(
        "modality",
        match replacement.modality {
            VisualModality::Image => "image",
            VisualModality::Video => "video",
        },
    )?;
    item.set_item(
        "rendered_code_points",
        coordinate(replacement.rendered_code_points),
    )?;
    item.set_item(
        "expanded_code_points",
        coordinate(replacement.expanded_code_points),
    )?;
    item.set_item("expanded_tokens", coordinate(replacement.expanded_tokens))?;
    Ok(item)
}

fn sidecar_image_dict<'py>(py: Python<'py>, image: &ImageSidecar) -> PyResult<Bound<'py, PyDict>> {
    let item = PyDict::new(py);
    item.set_item("request_index", image.request_index)?;
    item.set_item("grid_row", image.grid_row)?;
    let replacement = PyDict::new(py);
    replacement.set_item("code_points", coordinate(image.replacement.code_points))?;
    replacement.set_item("tokens", coordinate(image.replacement.tokens))?;
    item.set_item("replacement", replacement)?;
    Ok(item)
}

fn geometry_dict(
    py: Python<'_>,
    geometry: qwen_mm_core::ImageGeometryPlan,
) -> PyResult<Bound<'_, PyDict>> {
    let item = PyDict::new(py);
    item.set_item("height", geometry.height)?;
    item.set_item("width", geometry.width)?;
    item.set_item("image_grid_thw", geometry.image_grid_thw)?;
    item.set_item("patch_rows", geometry.patch_rows)?;
    item.set_item("placeholder_count", geometry.placeholder_count)?;
    item.set_item("rgb_row_stride_bytes", geometry.rgb_row_stride_bytes)?;
    item.set_item("rgb_capacity_bytes", geometry.rgb_capacity_bytes)?;
    item.set_item(
        "pixel_values_row_stride_bytes",
        geometry.pixel_values_row_stride_bytes,
    )?;
    item.set_item(
        "pixel_values_capacity_bytes",
        geometry.pixel_values_capacity_bytes,
    )?;
    item.set_item(
        "image_grid_row_stride_bytes",
        geometry.image_grid_row_stride_bytes,
    )?;
    item.set_item(
        "image_grid_capacity_bytes",
        geometry.image_grid_capacity_bytes,
    )?;
    Ok(item)
}

const fn coordinate(range: CoordinateRange) -> [i64; 2] {
    [range.start, range.end]
}

#[cfg(test)]
mod tests {
    use numpy::{PyArrayMethods, PyUntypedArrayMethods};
    use pyo3::{Python, types::PyAnyMethods};

    use super::{NativeMatrix, RunError, allocate, matrix_f32, validate_numpy_shape};

    #[test]
    fn impossible_output_allocation_fails_without_panicking() {
        let error = allocate::<u8>("test_output", usize::MAX).expect_err("allocation must fail");
        assert!(matches!(
            error,
            RunError::Allocation {
                name: "test_output",
                elements: usize::MAX
            }
        ));
    }

    #[test]
    #[ignore = "requires NumPy importable by embedded CPython"]
    fn vec_allocation_moves_into_two_dimensional_numpy_without_copy() {
        Python::initialize();
        Python::attach(|py| {
            let data = vec![0.25_f32; 24];
            let pointer = data.as_ptr() as usize;
            let array = matrix_f32(
                py,
                NativeMatrix {
                    shape: [4, 6],
                    data,
                    allocation_ptr: pointer,
                },
            )
            .expect("NumPy transfer");
            assert_eq!(array.data() as usize, pointer);
            assert_eq!(array.shape(), [4, 6]);
            assert!(array.is_c_contiguous());
            assert!(!array.getattr("base").expect("base").is_none());
            assert_eq!(
                array
                    .readonly()
                    .as_slice()
                    .expect("contiguous NumPy storage"),
                &[0.25_f32; 24]
            );
        });
    }

    #[test]
    fn numpy_dimensions_are_checked_before_transfer() {
        assert!(validate_numpy_shape([2, 3]).is_ok());
    }
}
