//! Bounded, opt-in observability for complete preprocessing operations.
//!
//! The normal processor path does not construct a recorder. Observed calls use
//! stable stage names and explicitly scoped owned-buffer accounting so retained
//! model outputs are never reported as transient scratch memory.

use std::time::Instant;

use serde::Serialize;

use crate::QwenError;

/// Version of the machine-readable observation report.
pub const OBSERVATION_SCHEMA_VERSION: &str = "qwen-mm-observation-v1";

/// Default upper bound for each span, buffer, and copy event list.
pub const DEFAULT_OBSERVATION_EVENT_CAPACITY: usize = 4_096;

/// Correlation coordinates attached to a stage or owned buffer.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct ObservationScope {
    /// Batch request row, when the event belongs to one request.
    pub request_index: Option<usize>,
    /// Message row inside the request, for media occurrences.
    pub message_index: Option<usize>,
    /// Content-item row inside the message, for media occurrences.
    pub content_item_index: Option<usize>,
    /// Global media-occurrence index in deterministic traversal order.
    pub media_index: Option<usize>,
    /// Input index referenced by the occurrence.
    pub input_index: Option<usize>,
}

impl ObservationScope {
    /// Constructs request-only correlation.
    #[must_use]
    pub const fn request(request_index: usize) -> Self {
        Self {
            request_index: Some(request_index),
            message_index: None,
            content_item_index: None,
            media_index: None,
            input_index: None,
        }
    }

    /// Constructs request/media correlation.
    #[must_use]
    pub const fn media(request_index: usize, media_index: usize, input_index: usize) -> Self {
        Self {
            request_index: Some(request_index),
            message_index: None,
            content_item_index: None,
            media_index: Some(media_index),
            input_index: Some(input_index),
        }
    }

    /// Constructs full request/message/content/media correlation.
    #[must_use]
    pub const fn media_at(
        request_index: usize,
        message_index: usize,
        content_item_index: usize,
        media_index: usize,
        input_index: usize,
    ) -> Self {
        Self {
            request_index: Some(request_index),
            message_index: Some(message_index),
            content_item_index: Some(content_item_index),
            media_index: Some(media_index),
            input_index: Some(input_index),
        }
    }
}

/// Stable completion state for one stage span.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StageOutcome {
    /// The stage completed successfully.
    Success,
    /// The stage returned a categorized error.
    Error,
}

/// One bounded stage observation in begin order.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct StageSpan {
    /// Zero-based begin sequence within the operation.
    pub sequence: u64,
    /// Enclosing span sequence, or `None` for a top-level span.
    pub parent_sequence: Option<u64>,
    /// Stable stage name.
    pub name: String,
    /// Optional request/media coordinates.
    pub scope: ObservationScope,
    /// Nanoseconds from operation start to span begin.
    pub started_ns: u64,
    /// Monotonic wall duration in nanoseconds.
    pub duration_ns: u64,
    /// Duration less the inclusive durations of directly nested spans.
    pub exclusive_duration_ns: u64,
    /// Stage completion state.
    pub outcome: StageOutcome,
    /// Categorized error string when `outcome` is `error`.
    pub error_category: Option<String>,
    /// Bytes consumed directly by this stage, when known.
    pub input_bytes: u64,
    /// Bytes produced directly by this stage, when known.
    pub output_bytes: u64,
    /// Stable logical output shape, when applicable.
    pub shape: Vec<u64>,
}

/// Whether an owned buffer is final output or transient working storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BufferClass {
    /// Storage retained by the returned result/NumPy arrays.
    RetainedOutput,
    /// Destination storage allocated but dropped because the call failed.
    DiscardedOutput,
    /// Storage released before the complete operation returns.
    Transient,
}

/// One tracked qwen-mm-owned material buffer.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BufferEvent {
    /// Zero-based allocation sequence.
    pub sequence: u64,
    /// Stable buffer name.
    pub name: String,
    /// Retained versus transient classification.
    pub class: BufferClass,
    /// Optional request/media correlation.
    pub scope: ObservationScope,
    /// Exact logical allocation capacity in bytes.
    pub bytes: u64,
    /// Nanoseconds from operation start to allocation observation.
    pub allocated_at_ns: u64,
    /// Nanoseconds from operation start to release, for tracked transients.
    pub released_at_ns: Option<u64>,
}

/// One controlled byte-for-byte copy in occurrence order.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CopyEvent {
    /// Zero-based copy sequence.
    pub sequence: u64,
    /// Stable copy name identifying its source and purpose.
    pub name: String,
    /// Optional request/media correlation.
    pub scope: ObservationScope,
    /// Exact number of bytes copied.
    pub bytes: u64,
}

/// Aggregate owned-buffer and controlled-copy counters.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct AllocationCounters {
    /// Count of tracked qwen-mm-owned material allocations.
    pub allocation_count: u64,
    /// Sum of their logical byte capacities.
    pub allocated_bytes: u64,
    /// Count of controlled byte-for-byte copies.
    pub copy_count: u64,
    /// Bytes copied by those controlled copies.
    pub copied_bytes: u64,
    /// Transient bytes live when the report snapshot was taken.
    pub transient_live_bytes: u64,
    /// Peak live transient bytes, excluding final outputs.
    pub peak_transient_live_bytes: u64,
    /// Bytes retained by final official arrays.
    pub retained_final_output_bytes: u64,
}

/// Counts proving which language/runtime boundaries executed.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct CallCounters {
    /// Public `Processor.prepare_batch_observed` calls.
    pub public_python_calls: u64,
    /// Native batch executions entered from the binding.
    pub native_batch_calls: u64,
    /// Media occurrences processed by the native candidate.
    pub native_visual_calls: u64,
    /// Python callbacks made after entering the native batch path.
    pub python_callbacks: u64,
    /// Hugging Face processor calls made by the candidate path.
    pub hugging_face_calls: u64,
    /// Qwen VL Utils calls made by the candidate path.
    pub qwen_vl_utils_calls: u64,
    /// Pillow calls made by the candidate path.
    pub pillow_calls: u64,
    /// `TorchVision` calls made by the candidate path.
    pub torchvision_calls: u64,
}

/// Complete bounded report for one observed operation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ObservationReport {
    /// Machine-readable schema identifier.
    pub schema_version: String,
    /// Maximum retained events in each span, buffer, and copy list.
    pub event_capacity: usize,
    /// Events omitted after reaching the configured bound.
    pub dropped_events: u64,
    /// Stable operation outcome.
    pub outcome: StageOutcome,
    /// Top-level error category, if the operation failed.
    pub error_category: Option<String>,
    /// Monotonic complete-operation duration.
    pub duration_ns: u64,
    /// Spans in stable begin order.
    pub spans: Vec<StageSpan>,
    /// Tracked owned buffers in allocation order.
    pub buffers: Vec<BufferEvent>,
    /// Controlled copies in occurrence order.
    pub copies: Vec<CopyEvent>,
    /// Allocation/copy aggregates.
    pub allocations: AllocationCounters,
    /// Python/native/fallback call counts.
    pub calls: CallCounters,
    /// Honest scope statement for the counters.
    pub counter_scope: String,
}

/// One value-or-error paired with the observation report from the same call.
///
/// Keeping the error inside the envelope lets bindings attach the report to a
/// raised exception without relying on mutable "last report" processor state.
#[derive(Debug)]
pub struct Observed<T> {
    /// Complete operation result.
    pub result: crate::Result<T>,
    /// Report finalized after all success/error buffer lifetimes were closed.
    pub report: ObservationReport,
}

impl<T> Observed<T> {
    /// Splits the envelope into its result and report.
    pub fn into_parts(self) -> (crate::Result<T>, ObservationReport) {
        (self.result, self.report)
    }
}

/// Opaque handle returned when a span begins.
#[derive(Clone, Copy, Debug)]
pub struct SpanToken {
    sequence: u64,
    started: Instant,
    index: Option<usize>,
}

#[derive(Debug)]
struct ActiveSpan {
    sequence: u64,
}

/// Mutable recorder used only by explicitly observed calls.
#[derive(Debug)]
pub struct ObservationRecorder {
    started: Instant,
    capacity: usize,
    next_span_sequence: u64,
    next_buffer_sequence: u64,
    next_copy_sequence: u64,
    dropped_events: u64,
    spans: Vec<StageSpan>,
    active_spans: Vec<ActiveSpan>,
    buffers: Vec<BufferEvent>,
    copies: Vec<CopyEvent>,
    allocations: AllocationCounters,
    calls: CallCounters,
    outcome: StageOutcome,
    error_category: Option<String>,
}

impl Default for ObservationRecorder {
    fn default() -> Self {
        Self::new(DEFAULT_OBSERVATION_EVENT_CAPACITY)
    }
}

impl ObservationRecorder {
    /// Starts one recorder. A zero capacity is accepted and records only
    /// aggregate counters plus the number of dropped events.
    #[must_use]
    pub fn new(event_capacity: usize) -> Self {
        Self {
            started: Instant::now(),
            capacity: event_capacity,
            next_span_sequence: 0,
            next_buffer_sequence: 0,
            next_copy_sequence: 0,
            dropped_events: 0,
            spans: Vec::with_capacity(event_capacity.min(256)),
            active_spans: Vec::with_capacity(8),
            buffers: Vec::with_capacity(event_capacity.min(256)),
            copies: Vec::with_capacity(event_capacity.min(256)),
            allocations: AllocationCounters::default(),
            calls: CallCounters::default(),
            outcome: StageOutcome::Success,
            error_category: None,
        }
    }

    /// Begins a stable stage span.
    pub fn begin(
        &mut self,
        name: &'static str,
        scope: ObservationScope,
        input_bytes: u64,
    ) -> SpanToken {
        let sequence = self.next_span_sequence;
        self.next_span_sequence = self.next_span_sequence.saturating_add(1);
        let started = Instant::now();
        let started_ns = nanos(started.duration_since(self.started));
        let parent_sequence = self.active_spans.last().map(|active| active.sequence);
        let index = if self.spans.len() < self.capacity {
            let index = self.spans.len();
            self.spans.push(StageSpan {
                sequence,
                parent_sequence,
                name: name.to_owned(),
                scope,
                started_ns,
                duration_ns: 0,
                exclusive_duration_ns: 0,
                outcome: StageOutcome::Success,
                error_category: None,
                input_bytes,
                output_bytes: 0,
                shape: Vec::new(),
            });
            Some(index)
        } else {
            self.dropped_events = self.dropped_events.saturating_add(1);
            None
        };
        self.active_spans.push(ActiveSpan { sequence });
        SpanToken {
            sequence,
            started,
            index,
        }
    }

    /// Completes a successful stage span.
    pub fn finish_success(&mut self, token: SpanToken, output_bytes: u64, shape: &[u64]) {
        self.finish(token, StageOutcome::Success, None, output_bytes, shape);
    }

    /// Completes a failed span with the stable core category.
    pub fn finish_error(&mut self, token: SpanToken, error: &QwenError) {
        self.finish(
            token,
            StageOutcome::Error,
            Some(error.category().as_str()),
            0,
            &[],
        );
        self.mark_error_category(error.category().as_str());
    }

    /// Completes a failed binding span whose category is already stable.
    pub fn finish_error_category(&mut self, token: SpanToken, category: &str) {
        self.finish(token, StageOutcome::Error, Some(category), 0, &[]);
        self.mark_error_category(category);
    }

    fn finish(
        &mut self,
        token: SpanToken,
        outcome: StageOutcome,
        error_category: Option<&str>,
        output_bytes: u64,
        shape: &[u64],
    ) {
        let duration_ns = nanos(token.started.elapsed());
        let active_index = self
            .active_spans
            .iter()
            .rposition(|active| active.sequence == token.sequence);
        if let Some(index) = active_index {
            self.active_spans.remove(index);
        }
        if let Some(index) = token.index {
            let span = &mut self.spans[index];
            debug_assert_eq!(span.sequence, token.sequence);
            span.duration_ns = duration_ns;
            // Recomputed from parent links in `report`, making attribution
            // independent of span completion order.
            span.exclusive_duration_ns = duration_ns;
            span.outcome = outcome;
            span.error_category = error_category.map(str::to_owned);
            span.output_bytes = output_bytes;
            span.shape = shape.to_vec();
        }
    }

    /// Records one qwen-mm-owned material allocation.
    pub fn record_allocation(
        &mut self,
        name: &'static str,
        class: BufferClass,
        scope: ObservationScope,
        bytes: u64,
    ) {
        self.allocations.allocation_count = self.allocations.allocation_count.saturating_add(1);
        self.allocations.allocated_bytes = self.allocations.allocated_bytes.saturating_add(bytes);
        match class {
            BufferClass::RetainedOutput => {
                self.allocations.retained_final_output_bytes = self
                    .allocations
                    .retained_final_output_bytes
                    .saturating_add(bytes);
            }
            BufferClass::Transient => {
                self.allocations.transient_live_bytes =
                    self.allocations.transient_live_bytes.saturating_add(bytes);
                self.allocations.peak_transient_live_bytes = self
                    .allocations
                    .peak_transient_live_bytes
                    .max(self.allocations.transient_live_bytes);
            }
            BufferClass::DiscardedOutput => {}
        }
        let sequence = self.next_buffer_sequence;
        self.next_buffer_sequence = self.next_buffer_sequence.saturating_add(1);
        if self.buffers.len() < self.capacity {
            self.buffers.push(BufferEvent {
                sequence,
                name: name.to_owned(),
                class,
                scope,
                bytes,
                allocated_at_ns: nanos(self.started.elapsed()),
                released_at_ns: None,
            });
        } else {
            self.dropped_events = self.dropped_events.saturating_add(1);
        }
    }

    /// Releases the earliest still-live matching transient buffer.
    pub fn release_transient(&mut self, name: &'static str, scope: ObservationScope, bytes: u64) {
        self.allocations.transient_live_bytes =
            self.allocations.transient_live_bytes.saturating_sub(bytes);
        if let Some(event) = self.buffers.iter_mut().find(|event| {
            event.name == name
                && event.class == BufferClass::Transient
                && event.scope == scope
                && event.released_at_ns.is_none()
        }) {
            event.released_at_ns = Some(nanos(self.started.elapsed()));
        }
    }

    /// Renames a live allocation when an internal scratch candidate becomes
    /// the prepared buffer owned by the batch plan.
    pub fn rename_live_transient(
        &mut self,
        old_name: &'static str,
        new_name: &'static str,
        scope: ObservationScope,
    ) {
        if let Some(event) = self.buffers.iter_mut().rev().find(|event| {
            event.name == old_name
                && event.class == BufferClass::Transient
                && event.scope == scope
                && event.released_at_ns.is_none()
        }) {
            new_name.clone_into(&mut event.name);
        }
    }

    /// Releases every tracked transient, used when an observed operation
    /// aborts and Rust drops all private intermediates.
    pub fn release_all_transients(&mut self) {
        let released_at = nanos(self.started.elapsed());
        for event in &mut self.buffers {
            if event.class == BufferClass::Transient && event.released_at_ns.is_none() {
                event.released_at_ns = Some(released_at);
            }
        }
        self.allocations.transient_live_bytes = 0;
    }

    /// Reclassifies allocated destinations when no result is returned.
    pub fn discard_retained_outputs(&mut self) {
        let released_at = nanos(self.started.elapsed());
        for event in &mut self.buffers {
            if event.class == BufferClass::RetainedOutput {
                event.class = BufferClass::DiscardedOutput;
                event.released_at_ns = Some(released_at);
            }
        }
        self.allocations.retained_final_output_bytes = 0;
    }

    /// Records one controlled byte-for-byte copy.
    pub fn record_copy(&mut self, name: &'static str, scope: ObservationScope, bytes: u64) {
        self.allocations.copy_count = self.allocations.copy_count.saturating_add(1);
        self.allocations.copied_bytes = self.allocations.copied_bytes.saturating_add(bytes);
        let sequence = self.next_copy_sequence;
        self.next_copy_sequence = self.next_copy_sequence.saturating_add(1);
        if self.copies.len() < self.capacity {
            self.copies.push(CopyEvent {
                sequence,
                name: name.to_owned(),
                scope,
                bytes,
            });
        } else {
            self.dropped_events = self.dropped_events.saturating_add(1);
        }
    }

    /// Mutable boundary counters for the binding and native executor.
    pub fn calls_mut(&mut self) -> &mut CallCounters {
        &mut self.calls
    }

    /// Marks the complete operation failed without requiring a `QwenError`.
    pub fn mark_error_category(&mut self, category: &str) {
        self.outcome = StageOutcome::Error;
        self.error_category = Some(category.to_owned());
    }

    /// Returns a complete snapshot. Repeated snapshots do not mutate state.
    #[must_use]
    pub fn report(&self) -> ObservationReport {
        let direct_child_durations = self
            .spans
            .iter()
            .map(|parent| {
                self.spans
                    .iter()
                    .filter(|child| child.parent_sequence == Some(parent.sequence))
                    .fold(0_u64, |total, child| {
                        total.saturating_add(child.duration_ns)
                    })
            })
            .collect::<Vec<_>>();
        let mut spans = self.spans.clone();
        for (span, child_duration) in spans.iter_mut().zip(direct_child_durations) {
            span.exclusive_duration_ns = span.duration_ns.saturating_sub(child_duration);
        }
        ObservationReport {
            schema_version: OBSERVATION_SCHEMA_VERSION.to_owned(),
            event_capacity: self.capacity,
            dropped_events: self.dropped_events,
            outcome: self.outcome,
            error_category: self.error_category.clone(),
            duration_ns: nanos(self.started.elapsed()),
            spans,
            buffers: self.buffers.clone(),
            copies: self.copies.clone(),
            allocations: self.allocations.clone(),
            calls: self.calls.clone(),
            counter_scope: "qwen-mm decoded/prepared RGB, observed still-image resize scratch, official output destinations, and controlled packed/source-clone copies; prompt/token/plan metadata plus codec and tokenizer internal scratch remain profile-only; language-binding input ownership is added by the binding recorder".to_owned(),
        }
    }
}

fn nanos(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_outputs_do_not_inflate_transient_peak() {
        let mut recorder = ObservationRecorder::new(8);
        let scope = ObservationScope::request(0);
        recorder.record_allocation("prepared_rgb", BufferClass::Transient, scope, 12);
        recorder.record_allocation("pixel_values", BufferClass::RetainedOutput, scope, 400);
        recorder.release_transient("prepared_rgb", scope, 12);
        let report = recorder.report();
        assert_eq!(report.allocations.peak_transient_live_bytes, 12);
        assert_eq!(report.allocations.transient_live_bytes, 0);
        assert_eq!(report.allocations.retained_final_output_bytes, 400);
    }

    #[test]
    fn event_capacity_is_hard_bounded_without_losing_aggregates() {
        let mut recorder = ObservationRecorder::new(1);
        let first = recorder.begin("one", ObservationScope::default(), 0);
        recorder.finish_success(first, 0, &[]);
        let second = recorder.begin("two", ObservationScope::default(), 0);
        recorder.finish_success(second, 0, &[]);
        recorder.record_allocation(
            "output",
            BufferClass::RetainedOutput,
            ObservationScope::default(),
            64,
        );
        recorder.record_copy("first", ObservationScope::default(), 7);
        recorder.record_copy("second", ObservationScope::request(0), 11);
        let report = recorder.report();
        assert_eq!(report.spans.len(), 1);
        assert_eq!(report.buffers.len(), 1);
        assert_eq!(report.copies.len(), 1);
        assert_eq!(report.copies[0].sequence, 0);
        assert_eq!(report.copies[0].name, "first");
        assert_eq!(report.copies[0].bytes, 7);
        assert_eq!(report.dropped_events, 2);
        assert_eq!(report.allocations.retained_final_output_bytes, 64);
        assert_eq!(report.allocations.copy_count, 2);
        assert_eq!(report.allocations.copied_bytes, 18);
    }

    #[test]
    fn final_error_category_overwrites_earlier_child_failures() {
        let mut recorder = ObservationRecorder::new(4);
        recorder.mark_error_category("media_geometry");
        recorder.mark_error_category("media_decode");
        let report = recorder.report();
        assert_eq!(report.outcome, StageOutcome::Error);
        assert_eq!(report.error_category.as_deref(), Some("media_decode"));
    }

    #[test]
    fn exclusive_durations_follow_parent_links_even_when_finish_is_not_lifo() {
        let mut recorder = ObservationRecorder::new(8);
        let outer = recorder.begin("outer", ObservationScope::default(), 0);
        let parent = recorder.begin("parent", ObservationScope::default(), 0);
        let child = recorder.begin("child", ObservationScope::default(), 0);
        recorder.finish_success(parent, 0, &[]);
        recorder.finish_success(child, 0, &[]);
        recorder.finish_success(outer, 0, &[]);
        let report = recorder.report();
        let outer = &report.spans[0];
        let parent = &report.spans[1];
        let child = &report.spans[2];
        assert_eq!(parent.parent_sequence, Some(outer.sequence));
        assert_eq!(child.parent_sequence, Some(parent.sequence));
        assert_eq!(
            outer.exclusive_duration_ns,
            outer.duration_ns.saturating_sub(parent.duration_ns)
        );
        assert_eq!(
            parent.exclusive_duration_ns,
            parent.duration_ns.saturating_sub(child.duration_ns)
        );
        assert_eq!(child.exclusive_duration_ns, child.duration_ns);
    }

    #[test]
    fn adjacent_sibling_intervals_do_not_overlap() {
        let mut recorder = ObservationRecorder::new(1024);
        let parent = recorder.begin("parent", ObservationScope::default(), 0);
        for _ in 0..512 {
            let child = recorder.begin("child", ObservationScope::default(), 0);
            let recorded_start = recorder.spans[child.index.expect("recorded child")].started_ns;
            assert_eq!(
                recorded_start,
                nanos(child.started.duration_since(recorder.started))
            );
            recorder.finish_success(child, 0, &[]);
        }
        recorder.finish_success(parent, 0, &[]);

        let report = recorder.report();
        let children = report
            .spans
            .iter()
            .filter(|span| span.parent_sequence == Some(parent.sequence))
            .collect::<Vec<_>>();
        for adjacent in children.windows(2) {
            assert!(
                adjacent[0].started_ns + adjacent[0].duration_ns <= adjacent[1].started_ns,
                "adjacent sibling spans overlap: {:?} then {:?}",
                adjacent[0],
                adjacent[1]
            );
        }
    }
}
