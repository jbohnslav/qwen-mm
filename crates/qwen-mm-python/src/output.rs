//! Native execution and zero-copy `NumPy` result construction.

use std::sync::Arc;

use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyUntypedArrayMethods, ndarray::Array2};
use pyo3::{
    prelude::*,
    types::{PyDict, PyList},
};
use qwen_mm_core::{
    BatchDestinations, BatchImageLayout, BatchRequestLayout, ContentItem, CoordinateRange,
    FunctionCall, ImageInput, ImageSidecar, IntegrationSidecar, Message, MessageContent,
    PreparedTextRequest, ProcessedBatchImageOccurrence, QwenError, QwenImageProcessor, Request,
    Rgb8, TextReplacement, ToolCall, ToolDefinition, VisualModality,
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
    pub(crate) input_ids: NativeMatrix<i64>,
    pub(crate) attention_mask: NativeMatrix<i64>,
    pub(crate) mm_token_type_ids: NativeMatrix<i64>,
    pub(crate) pixel_values: Option<NativeMatrix<f32>>,
    pub(crate) image_grid_thw: Option<NativeMatrix<i64>>,
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

    fn __repr__(&self) -> String {
        format!("PreparedBatch(keys={:?})", self.official_keys)
    }
}

impl PyPreparedBatch {
    pub(crate) fn from_native(py: Python<'_>, native: NativeBatch) -> PyResult<Self> {
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
        )?;
        Ok(Self {
            arrays: arrays.unbind(),
            metadata: metadata.unbind(),
            official_keys,
        })
    }
}

#[allow(clippy::too_many_lines)]
pub(crate) fn run_batch(
    processor: &Arc<QwenImageProcessor>,
    owned: &[OwnedRequest],
) -> Result<NativeBatch, RunError> {
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

    let plan = processor.plan_batch(&requests)?;
    let capacities = *plan.capacities();
    let mut input_ids = allocate::<i64>("input_ids", capacities.input_ids.elements)?;
    let mut attention_mask = allocate::<i64>("attention_mask", capacities.attention_mask.elements)?;
    let mut mm_token_type_ids =
        allocate::<i64>("mm_token_type_ids", capacities.mm_token_type_ids.elements)?;
    let mut pixel_values = capacities
        .pixel_values
        .map(|capacity| allocate::<f32>("pixel_values", capacity.elements))
        .transpose()?;
    let mut image_grid_thw = capacities
        .image_grid_thw
        .map(|capacity| allocate::<i64>("image_grid_thw", capacity.elements))
        .transpose()?;
    let input_ids_ptr = input_ids.as_ptr() as usize;
    let attention_mask_ptr = attention_mask.as_ptr() as usize;
    let mm_token_type_ids_ptr = mm_token_type_ids.as_ptr() as usize;
    let pixel_values_ptr = pixel_values
        .as_ref()
        .map_or(0, |values| values.as_ptr() as usize);
    let image_grid_thw_ptr = image_grid_thw
        .as_ref()
        .map_or(0, |values| values.as_ptr() as usize);
    let (contract_id, profile_fingerprint, text, sidecar, images) = {
        let view = processor.execute_plan_into(
            &plan,
            BatchDestinations {
                input_ids: &mut input_ids,
                attention_mask: &mut attention_mask,
                mm_token_type_ids: &mut mm_token_type_ids,
                pixel_values: pixel_values.as_deref_mut(),
                image_grid_thw: image_grid_thw.as_deref_mut(),
                pixel_values_videos: None,
                video_grid_thw: None,
            },
        )?;
        (
            view.contract_id.to_owned(),
            view.profile_fingerprint.to_owned(),
            view.text.to_vec(),
            view.sidecar().clone(),
            view.images.to_vec(),
        )
    };
    Ok(NativeBatch {
        contract_id,
        profile_fingerprint,
        text,
        sidecar,
        images,
        request_layouts: plan.request_layouts().to_vec(),
        image_layouts: plan.image_layouts().to_vec(),
        input_ids: NativeMatrix {
            shape: capacities.input_ids.shape,
            data: input_ids,
            allocation_ptr: input_ids_ptr,
        },
        attention_mask: NativeMatrix {
            shape: capacities.attention_mask.shape,
            data: attention_mask,
            allocation_ptr: attention_mask_ptr,
        },
        mm_token_type_ids: NativeMatrix {
            shape: capacities.mm_token_type_ids.shape,
            data: mm_token_type_ids,
            allocation_ptr: mm_token_type_ids_ptr,
        },
        pixel_values: pixel_values
            .zip(capacities.pixel_values)
            .map(|(data, capacity)| NativeMatrix {
                shape: capacity.shape,
                data,
                allocation_ptr: pixel_values_ptr,
            }),
        image_grid_thw: image_grid_thw
            .zip(capacities.image_grid_thw)
            .map(|(data, capacity)| NativeMatrix {
                shape: capacity.shape,
                data,
                allocation_ptr: image_grid_thw_ptr,
            }),
    })
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
) -> PyResult<Bound<'py, PyDict>> {
    let metadata = PyDict::new(py);
    metadata.set_item("contract_id", contract_id)?;
    metadata.set_item("profile_fingerprint", profile_fingerprint)?;
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
        item.set_item("right_padding", layout.right_padding)?;
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
