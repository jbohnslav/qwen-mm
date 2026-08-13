from __future__ import annotations

import copy
import hashlib
import json
import signal
import subprocess
import tempfile
import unittest
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import call, patch

from qwen_mm_reference.benchmark_protocol import (
    THREAD_ENVIRONMENT_NAMES,
    architecture_family,
    load_workload,
    materialize_case,
    select_cases,
    workload_provenance,
)
from qwen_mm_reference.profile_v1 import (
    SAMPLER_PROTOCOLS,
    ProfileArtifactError,
    _artifact_identity,
    _authenticate_sampler_after_capture,
    _is_qwen_binding_frame,
    _is_qwen_core_frame,
    _is_qwen_native_frame,
    _parse_collapsed,
    _parse_macos_sample,
    _parse_py_spy_summary,
    _run_py_spy_child,
    _source_identity,
    _source_tree_digest_at_revision,
    _source_tree_digest_excluding,
    _stack_rankings,
    merge_bundles,
    summarize_bundle,
    validate_bundle,
    validate_observation,
    validate_sampled_profile,
)

from scripts.modal_benchmark_support import committed_source_tree_digest, source_tree_digest


def scope(**overrides: int | None) -> dict[str, int | None]:
    value: dict[str, int | None] = {
        "request_index": None,
        "message_index": None,
        "content_item_index": None,
        "media_index": None,
        "input_index": None,
    }
    value.update(overrides)
    return value


def span(
    sequence: int,
    name: str,
    *,
    parent: int | None = 0,
    started: int,
    duration: int,
    output_bytes: int = 0,
    event_scope: dict[str, int | None] | None = None,
    input_bytes: int = 0,
    shape: list[int] | None = None,
) -> dict[str, object]:
    return {
        "sequence": sequence,
        "parent_sequence": parent,
        "name": name,
        "scope": event_scope or scope(),
        "started_ns": started,
        "duration_ns": duration,
        "exclusive_duration_ns": duration,
        "outcome": "success",
        "error_category": None,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "shape": shape or [],
    }


def observation() -> dict[str, object]:
    retained = 96
    spans = [
        span(
            0, "binding.prepare_batch", parent=None, started=0, duration=90, output_bytes=retained
        ),
        span(1, "binding.parse_requests", started=1, duration=5),
        span(2, "native.batch.plan", started=7, duration=35, output_bytes=retained),
        span(
            3,
            "native.chat.render",
            parent=2,
            started=8,
            duration=5,
            output_bytes=12,
            event_scope=scope(request_index=0),
        ),
        span(
            4,
            "native.chat.tokenize",
            parent=2,
            started=14,
            duration=5,
            output_bytes=16,
            shape=[4],
            event_scope=scope(request_index=0),
        ),
        span(
            5,
            "binding.destination.allocate",
            started=43,
            duration=5,
            output_bytes=retained,
            shape=[retained],
        ),
        span(
            6,
            "native.destination.execute",
            started=49,
            duration=20,
            output_bytes=retained,
        ),
        span(
            7,
            "binding.numpy.materialize",
            started=70,
            duration=10,
            output_bytes=retained,
            shape=[retained],
        ),
    ]
    spans[0]["exclusive_duration_ns"] = 15
    spans[2]["exclusive_duration_ns"] = 25
    buffers = [
        {
            "sequence": index,
            "name": name,
            "class": "retained_output",
            "scope": scope(),
            "bytes": 32,
            "allocated_at_ns": 45,
            "released_at_ns": None,
        }
        for index, name in enumerate(("input_ids", "attention_mask", "mm_token_type_ids"))
    ]
    return {
        "schema_version": "qwen-mm-observation-v1",
        "event_capacity": 64,
        "dropped_events": 0,
        "outcome": "success",
        "error_category": None,
        "duration_ns": 100,
        "spans": spans,
        "buffers": buffers,
        "copies": [],
        "allocations": {
            "allocation_count": 3,
            "allocated_bytes": retained,
            "copy_count": 0,
            "copied_bytes": 0,
            "transient_live_bytes": 0,
            "peak_transient_live_bytes": 0,
            "retained_final_output_bytes": retained,
        },
        "calls": {
            "public_python_calls": 1,
            "native_batch_calls": 1,
            "native_visual_calls": 0,
            "python_callbacks": 0,
            "hugging_face_calls": 0,
            "qwen_vl_utils_calls": 0,
            "pillow_calls": 0,
            "torchvision_calls": 0,
        },
        "counter_scope": "test material buffers",
    }


def thread_settings(budget: int) -> dict[str, object]:
    return {
        "budget": budget,
        "environment": {name: str(budget) for name in THREAD_ENVIRONMENT_NAMES},
        "torch": {},
    }


def complete_observation(media_scopes: list[tuple[int, int, int, int, int]], requests: int) -> dict:
    child_specs: list[tuple[str, dict[str, int | None], int, list[int]]] = [
        ("binding.parse_requests", scope(), 0, []),
        ("native.batch.plan", scope(), 0, []),
    ]
    child_specs.extend(
        (name, scope(request_index=request_index), 0, [])
        for request_index in range(requests)
        for name in ("native.chat.render", "native.chat.tokenize")
    )
    for media_scope in media_scopes:
        event_scope = scope(
            request_index=media_scope[0],
            message_index=media_scope[1],
            content_item_index=media_scope[2],
            media_index=media_scope[3],
            input_index=media_scope[4],
        )
        child_specs.extend(
            (name, event_scope, 1, [1])
            for name in (
                "native.media.plan",
                "native.media.decode_color",
                "native.media.resize",
                "native.media.normalize_patchify_layout",
            )
        )
    child_specs.extend(
        (name, scope(), 0, [])
        for name in (
            "binding.destination.allocate",
            "native.destination.execute",
            "binding.numpy.materialize",
        )
    )
    spans = [
        span(
            0,
            "binding.prepare_batch",
            parent=None,
            started=0,
            duration=len(child_specs) + 1,
            output_bytes=96,
        )
    ]
    spans[0]["exclusive_duration_ns"] = 1
    output_stages = {
        "binding.destination.allocate",
        "native.destination.execute",
        "binding.numpy.materialize",
    }
    for sequence, (name, event_scope, input_bytes, shape_value) in enumerate(child_specs, 1):
        spans.append(
            span(
                sequence,
                name,
                started=sequence,
                duration=1,
                output_bytes=96 if name in output_stages else 0,
                event_scope=event_scope,
                input_bytes=input_bytes,
                shape=shape_value,
            )
        )
    buffers = [
        {
            "sequence": index,
            "name": name,
            "class": "retained_output",
            "scope": scope(),
            "bytes": 32,
            "allocated_at_ns": 1,
            "released_at_ns": None,
        }
        for index, name in enumerate(("input_ids", "attention_mask", "mm_token_type_ids"))
    ]
    return {
        "schema_version": "qwen-mm-observation-v1",
        "event_capacity": 4096,
        "dropped_events": 0,
        "outcome": "success",
        "error_category": None,
        "duration_ns": len(child_specs) + 1,
        "spans": spans,
        "buffers": buffers,
        "copies": [],
        "allocations": {
            "allocation_count": 3,
            "allocated_bytes": 96,
            "copy_count": 0,
            "copied_bytes": 0,
            "transient_live_bytes": 0,
            "peak_transient_live_bytes": 0,
            "retained_final_output_bytes": 96,
        },
        "calls": {
            "public_python_calls": 1,
            "native_batch_calls": 1,
            "native_visual_calls": len(media_scopes),
            "python_callbacks": 0,
            "hugging_face_calls": 0,
            "qwen_vl_utils_calls": 0,
            "pillow_calls": 0,
            "torchvision_calls": 0,
        },
        "counter_scope": "synthetic complete-boundary fixture",
    }


class ObservationValidationTests(unittest.TestCase):
    def test_valid_observation_reconciles_complete_boundary(self) -> None:
        validate_observation(
            observation(), media_count=0, request_count=1, retained_output_bytes=96
        )

    def test_rejects_callback_timing_and_buffer_tampering(self) -> None:
        mutations = []

        def callback(value: dict[str, object]) -> None:
            value["calls"]["python_callbacks"] = 1  # type: ignore[index]

        mutations.append(callback)

        def exclusive(value: dict[str, object]) -> None:
            value["spans"][1]["exclusive_duration_ns"] = 99  # type: ignore[index]

        mutations.append(exclusive)

        def retained(value: dict[str, object]) -> None:
            value["allocations"]["retained_final_output_bytes"] = 95  # type: ignore[index]

        mutations.append(retained)

        def dropped(value: dict[str, object]) -> None:
            value["dropped_events"] = 1

        mutations.append(dropped)
        for mutate in mutations:
            with self.subTest(mutation=mutate.__name__):
                value = observation()
                mutate(value)
                with self.assertRaises(ProfileArtifactError):
                    validate_observation(
                        value, media_count=0, request_count=1, retained_output_bytes=96
                    )

    def test_transient_peak_reconciliation_sorts_lifetime_timestamps(self) -> None:
        value = observation()
        value["buffers"].extend(  # type: ignore[union-attr]
            [
                {
                    "sequence": 3,
                    "name": "first",
                    "class": "transient",
                    "scope": scope(),
                    "bytes": 5,
                    "allocated_at_ns": 10,
                    "released_at_ns": 50,
                },
                {
                    "sequence": 4,
                    "name": "second",
                    "class": "transient",
                    "scope": scope(),
                    "bytes": 7,
                    "allocated_at_ns": 20,
                    "released_at_ns": 30,
                },
            ]
        )
        allocations = value["allocations"]
        allocations["allocation_count"] = 5  # type: ignore[index]
        allocations["allocated_bytes"] = 108  # type: ignore[index]
        allocations["peak_transient_live_bytes"] = 12  # type: ignore[index]
        validate_observation(value, media_count=0, request_count=1, retained_output_bytes=96)


class SampleValidationTests(unittest.TestCase):
    def test_sampler_executable_identity_is_reauthenticated_after_capture(self) -> None:
        initial = {
            "binary_path": "/workspace/qwen-mm/.venv/bin/py-spy",
            "binary_sha256": "a" * 64,
            "binary_bytes": 123,
            "version": "py-spy 0.4.1",
        }
        with patch(
            "qwen_mm_reference.profile_v1._sampler_executable_identity",
            return_value=copy.deepcopy(initial),
        ) as authenticate:
            _authenticate_sampler_after_capture(initial, architecture="x86_64", py_spy="py-spy")
        authenticate.assert_called_once_with("x86_64", "py-spy")

        for field, replacement in (
            ("binary_path", "/tmp/replaced-py-spy"),
            ("binary_sha256", "b" * 64),
            ("binary_bytes", 124),
            ("version", "py-spy 0.4.2"),
        ):
            with self.subTest(field=field):
                changed = {**initial, field: replacement}
                with (
                    patch(
                        "qwen_mm_reference.profile_v1._sampler_executable_identity",
                        return_value=changed,
                    ),
                    self.assertRaisesRegex(ProfileArtifactError, "identity changed"),
                ):
                    _authenticate_sampler_after_capture(
                        initial, architecture="x86_64", py_spy="py-spy"
                    )

    def test_native_frame_classifiers_accept_exact_demangled_and_rust_v0_crates(self) -> None:
        core_frames = (
            "qwen_mm_core::processor::execute_plan_into",
            "<qwen_mm_core::Processor as core::fmt::Debug>::fmt",
            "_RNvMs2_NtCslYd21UUycFy_12qwen_mm_core9processorNtB5_18QwenImageProcessor",
            "_RINvXNtNtCslYd21UUycFy_12qwen_mm_core5value3ser9Serialize",
            "_RNvCs_12qwen_mm_core3foo",
            "_RNvC12qwen_mm_core9processor",
        )
        binding_frames = (
            "qwen_mm_python::binding::prepare_batch",
            "qwen_mm_native::output::run_batch_internal",
            "_RNvMsa_Csc8F1c33rCRZ_14qwen_mm_nativeNtB5_11PyProcessor",
            "_RINvXNtCsc8F1c33rCRZ_14qwen_mm_native6output18run_batch_internal",
            "_RNvCs_14qwen_mm_native3foo",
            "_RNvC14qwen_mm_native6output",
        )
        for frame in core_frames:
            with self.subTest(frame=frame):
                self.assertTrue(_is_qwen_core_frame(frame))
                self.assertTrue(_is_qwen_native_frame(frame))
        for frame in binding_frames:
            with self.subTest(frame=frame):
                self.assertTrue(_is_qwen_binding_frame(frame))
                self.assertTrue(_is_qwen_native_frame(frame))

    def test_native_frame_classifiers_reject_lookalike_crates(self) -> None:
        lookalikes = (
            "not_qwen_mm_core::processor::execute",
            "_qwen_mm_core::processor::execute",
            "not_qwen_mm_python::binding::prepare_batch",
            "_qwen_mm_native::output::run_batch_internal",
            "_RNvC19foo_12qwen_mm_core9processor",
            "_RNvC21foo_14qwen_mm_native6output",
            "prefixCslYd21UUycFy_12qwen_mm_core9processor",
            "prefixCsc8F1c33rCRZ_14qwen_mm_native6output",
        )
        for frame in lookalikes:
            with self.subTest(frame=frame):
                self.assertFalse(_is_qwen_core_frame(frame))
                self.assertFalse(_is_qwen_binding_frame(frame))
                self.assertFalse(_is_qwen_native_frame(frame))

    def test_py_spy_child_timeout_kills_authenticated_worker(self) -> None:
        worker_command = ["python", "-m", "qwen_mm_reference.profile_v1", "_sample_worker"]
        cleanup_done = False

        class TimedOutSampler:
            pid = 4321
            returncode = -9

            def __init__(self) -> None:
                self.killed = False
                self.communicate_timeouts: list[float | None] = []

            def communicate(self, timeout: float | None = None) -> tuple[str, str]:
                self.communicate_timeouts.append(timeout)
                if len(self.communicate_timeouts) == 1:
                    raise subprocess.TimeoutExpired(["py-spy"], timeout)
                if not cleanup_done:
                    raise AssertionError("pipe reaping ran before authenticated worker cleanup")
                return "", ""

            def kill(self) -> None:
                self.killed = True

        sampler = TimedOutSampler()

        def mark_cleanup(pid: int, requested_signal: int) -> None:
            nonlocal cleanup_done
            cleanup_done = True

        with (
            patch("qwen_mm_reference.profile_v1.subprocess.Popen", return_value=sampler),
            patch("qwen_mm_reference.profile_v1.Path.read_text", return_value="123"),
            patch(
                "qwen_mm_reference.profile_v1._linux_process_command",
                return_value=worker_command,
            ),
            patch("qwen_mm_reference.profile_v1.os.kill", side_effect=mark_cleanup) as kill,
        ):
            with self.assertRaisesRegex(ProfileArtifactError, "5 s timeout"):
                _run_py_spy_child(["py-spy", "record"], worker_command, timeout=5.0)
        self.assertTrue(sampler.killed)
        self.assertEqual(len(sampler.communicate_timeouts), 2)
        self.assertAlmostEqual(sampler.communicate_timeouts[0] or 0, 5.0, places=3)
        self.assertEqual(sampler.communicate_timeouts[1], 10.0)
        kill.assert_called_once_with(123, signal.SIGKILL)

    def test_py_spy_child_stops_sampler_after_authenticated_result(self) -> None:
        worker_command = ["python", "-m", "qwen_mm_reference.profile_v1", "_sample_worker"]
        sampler_signaled = False

        class SuccessfulSampler:
            pid = 4321
            returncode: int | None = None

            def poll(self) -> int | None:
                return self.returncode

            def communicate(self, timeout: float | None = None) -> tuple[str, str]:
                if not sampler_signaled:
                    raise AssertionError("sampler was reaped before its authenticated stop")
                self.returncode = 0
                return "Samples: 200 Errors: 0", ""

        def mark_signal(pid: int, requested_signal: int) -> None:
            nonlocal sampler_signaled
            self.assertEqual((pid, requested_signal), (4321, signal.SIGINT))
            sampler_signaled = True

        with tempfile.TemporaryDirectory() as temporary:
            ready_path = Path(temporary) / "ready.json"
            result_path = Path(temporary) / "result.json"
            ready_path.write_text(json.dumps({"pid": 321}), encoding="utf-8")
            result_path.write_text(json.dumps({"pid": 321}), encoding="utf-8")
            with (
                patch(
                    "qwen_mm_reference.profile_v1.subprocess.Popen",
                    return_value=SuccessfulSampler(),
                ),
                patch(
                    "qwen_mm_reference.profile_v1._linux_process_command",
                    side_effect=[worker_command, None],
                ),
                patch("qwen_mm_reference.profile_v1.os.kill", side_effect=mark_signal) as kill,
            ):
                result = _run_py_spy_child(
                    ["py-spy", "record"],
                    worker_command,
                    ready_path=ready_path,
                    result_path=result_path,
                )
        self.assertEqual(result.returncode, 0)
        self.assertIn("Samples: 200 Errors: 0", result.stdout)
        kill.assert_called_once_with(4321, signal.SIGINT)

    def test_py_spy_child_rejects_result_pid_command_mismatch(self) -> None:
        worker_command = ["python", "-m", "qwen_mm_reference.profile_v1", "_sample_worker"]

        class SuccessfulSampler:
            pid = 4321
            returncode: int | None = None

            def poll(self) -> int | None:
                return self.returncode

            def communicate(self, timeout: float | None = None) -> tuple[str, str]:
                self.returncode = 0
                return "Samples: 200 Errors: 0", ""

        with tempfile.TemporaryDirectory() as temporary:
            result_path = Path(temporary) / "result.json"
            result_path.write_text(json.dumps({"pid": 654}), encoding="utf-8")
            with (
                patch(
                    "qwen_mm_reference.profile_v1.subprocess.Popen",
                    return_value=SuccessfulSampler(),
                ),
                patch(
                    "qwen_mm_reference.profile_v1._linux_process_command",
                    return_value=["python", "unrelated.py"],
                ),
                patch("qwen_mm_reference.profile_v1.os.kill") as kill,
            ):
                with self.assertRaisesRegex(ProfileArtifactError, "exact worker command"):
                    _run_py_spy_child(["py-spy", "record"], worker_command, result_path=result_path)
        kill.assert_called_once_with(4321, signal.SIGINT)

    def test_py_spy_child_cleans_ready_worker_when_result_pid_mismatches(self) -> None:
        worker_command = ["python", "-m", "qwen_mm_reference.profile_v1", "_sample_worker"]

        class SuccessfulSampler:
            pid = 4321
            returncode: int | None = None

            def poll(self) -> int | None:
                return self.returncode

            def communicate(self, timeout: float | None = None) -> tuple[str, str]:
                self.returncode = 0
                return "Samples: 200 Errors: 0", ""

        def recorded_command(pid: int) -> list[str]:
            return worker_command if pid == 321 else ["python", "unrelated.py"]

        with tempfile.TemporaryDirectory() as temporary:
            ready_path = Path(temporary) / "ready.json"
            result_path = Path(temporary) / "result.json"
            ready_path.write_text(json.dumps({"pid": 321}), encoding="utf-8")
            result_path.write_text(json.dumps({"pid": 654}), encoding="utf-8")
            with (
                patch(
                    "qwen_mm_reference.profile_v1.subprocess.Popen",
                    return_value=SuccessfulSampler(),
                ),
                patch(
                    "qwen_mm_reference.profile_v1._linux_process_command",
                    side_effect=recorded_command,
                ),
                patch("qwen_mm_reference.profile_v1.os.kill") as kill,
            ):
                with self.assertRaisesRegex(ProfileArtifactError, "exact worker command"):
                    _run_py_spy_child(
                        ["py-spy", "record"],
                        worker_command,
                        ready_path=ready_path,
                        result_path=result_path,
                    )
        self.assertEqual(
            kill.call_args_list,
            [
                call(4321, signal.SIGINT),
                call(321, signal.SIGKILL),
            ],
        )

    def test_py_spy_child_does_not_signal_sampler_that_exited_before_result(self) -> None:
        worker_command = ["python", "-m", "qwen_mm_reference.profile_v1", "_sample_worker"]

        class ExitedSampler:
            pid = 4321
            returncode = 1

            def poll(self) -> int:
                return self.returncode

            def communicate(self, timeout: float | None = None) -> tuple[str, str]:
                return "", "child exited before result"

        with tempfile.TemporaryDirectory() as temporary:
            missing_result = Path(temporary) / "missing-result.json"
            with (
                patch(
                    "qwen_mm_reference.profile_v1.subprocess.Popen",
                    return_value=ExitedSampler(),
                ),
                patch("qwen_mm_reference.profile_v1.os.kill") as kill,
            ):
                with self.assertRaisesRegex(ProfileArtifactError, "before an authenticated"):
                    _run_py_spy_child(
                        ["py-spy", "record"], worker_command, result_path=missing_result
                    )
        kill.assert_not_called()

    def test_py_spy_child_cleans_ready_worker_when_sampler_exits_before_result(self) -> None:
        worker_command = ["python", "-m", "qwen_mm_reference.profile_v1", "_sample_worker"]

        class ExitedSampler:
            pid = 4321
            returncode = 0

            def poll(self) -> int:
                return self.returncode

            def communicate(self, timeout: float | None = None) -> tuple[str, str]:
                return "Samples: 200 Errors: 0", ""

        with tempfile.TemporaryDirectory() as temporary:
            ready_path = Path(temporary) / "ready.json"
            result_path = Path(temporary) / "missing-result.json"
            ready_path.write_text(json.dumps({"pid": 654}), encoding="utf-8")
            with (
                patch(
                    "qwen_mm_reference.profile_v1.subprocess.Popen",
                    return_value=ExitedSampler(),
                ),
                patch(
                    "qwen_mm_reference.profile_v1._linux_process_command",
                    return_value=worker_command,
                ),
                patch("qwen_mm_reference.profile_v1.os.kill") as kill,
            ):
                with self.assertRaisesRegex(ProfileArtifactError, "terminated 1 authenticated"):
                    _run_py_spy_child(
                        ["py-spy", "record"],
                        worker_command,
                        ready_path=ready_path,
                        result_path=result_path,
                    )
        kill.assert_called_once_with(654, signal.SIGKILL)

    def test_py_spy_child_rejects_and_kills_authenticated_worker_still_live(self) -> None:
        worker_command = ["python", "-m", "qwen_mm_reference.profile_v1", "_sample_worker"]

        class SuccessfulSampler:
            pid = 4321
            returncode: int | None = None

            def poll(self) -> int | None:
                return self.returncode

            def communicate(self, timeout: float | None = None) -> tuple[str, str]:
                self.returncode = 0
                return "Samples: 200 Errors: 0", ""

        with tempfile.TemporaryDirectory() as temporary:
            result_path = Path(temporary) / "result.json"
            result_path.write_text(json.dumps({"pid": 654}), encoding="utf-8")
            with (
                patch(
                    "qwen_mm_reference.profile_v1.subprocess.Popen",
                    return_value=SuccessfulSampler(),
                ),
                patch(
                    "qwen_mm_reference.profile_v1._linux_process_command",
                    side_effect=[worker_command, worker_command, worker_command],
                ),
                patch("qwen_mm_reference.profile_v1.os.kill") as kill,
            ):
                with self.assertRaisesRegex(ProfileArtifactError, "still live"):
                    _run_py_spy_child(["py-spy", "record"], worker_command, result_path=result_path)
        self.assertEqual(
            kill.call_args_list,
            [
                call(4321, signal.SIGINT),
                call(654, signal.SIGKILL),
            ],
        )

    def test_py_spy_nonzero_exit_kills_only_authenticated_recorded_worker(self) -> None:
        worker_command = ["python", "-m", "qwen_mm_reference.profile_v1", "_sample_worker"]

        class FailedSampler:
            returncode = 1

            def communicate(self, timeout: float | None = None) -> tuple[str, str]:
                return "", "failed"

        with tempfile.TemporaryDirectory() as temporary:
            ready_path = Path(temporary) / "ready.json"
            ready_path.write_text(json.dumps({"pid": 321}), encoding="utf-8")
            with (
                patch(
                    "qwen_mm_reference.profile_v1.subprocess.Popen",
                    return_value=FailedSampler(),
                ),
                patch(
                    "qwen_mm_reference.profile_v1._linux_process_command",
                    return_value=worker_command,
                ),
                patch("qwen_mm_reference.profile_v1.os.kill") as kill,
            ):
                result = _run_py_spy_child(
                    ["py-spy", "record"], worker_command, ready_path=ready_path
                )
        self.assertEqual(result.returncode, 1)
        kill.assert_called_once_with(321, signal.SIGKILL)

    def test_py_spy_nonzero_exit_never_kills_command_mismatch(self) -> None:
        worker_command = ["python", "-m", "qwen_mm_reference.profile_v1", "_sample_worker"]

        class FailedSampler:
            returncode = 1

            def communicate(self, timeout: float | None = None) -> tuple[str, str]:
                return "", "failed"

        with tempfile.TemporaryDirectory() as temporary:
            ready_path = Path(temporary) / "ready.json"
            ready_path.write_text(json.dumps({"pid": 654}), encoding="utf-8")
            with (
                patch(
                    "qwen_mm_reference.profile_v1.subprocess.Popen",
                    return_value=FailedSampler(),
                ),
                patch(
                    "qwen_mm_reference.profile_v1._linux_process_command",
                    return_value=["python", "unrelated.py"],
                ),
                patch("qwen_mm_reference.profile_v1.os.kill") as kill,
            ):
                result = _run_py_spy_child(
                    ["py-spy", "record"], worker_command, ready_path=ready_path
                )
        self.assertEqual(result.returncode, 1)
        kill.assert_not_called()

    def test_py_spy_summary_rejects_errors_and_ambiguity(self) -> None:
        self.assertEqual(_parse_py_spy_summary("Wrote raw data. Samples: 217 Errors: 0"), (217, 0))
        with self.assertRaisesRegex(ProfileArtifactError, "sampling errors"):
            _parse_py_spy_summary("Samples: 217 Errors: 1")
        with self.assertRaisesRegex(ProfileArtifactError, "unambiguous"):
            _parse_py_spy_summary("no summary")
        with self.assertRaisesRegex(ProfileArtifactError, "unambiguous"):
            _parse_py_spy_summary("Samples: 1 Errors: 0\nSamples: 2 Errors: 0")

    def test_macos_sample_call_graph_reduces_to_canonical_stacks(self) -> None:
        raw = """Analysis of sampling python every 1 milliseconds
Call graph:
    100 Thread_1
      100 _PyEval_EvalFrameDefault  (in Python) + 4  [0x1]
      + 100 qwen_mm_python::binding::prepare_batch  (in _native.so) + 4  [0x2]
          60 qwen_mm_core::processor::execute  (in _native.so) + 8  [0x3]

Total number in stack (recursive counted multiple, when >=5):
"""
        collapsed, count, unique = _parse_macos_sample(raw)
        self.assertEqual(count, 100)
        self.assertEqual(unique, 2)
        self.assertIn("qwen_mm_python::binding::prepare_batch", collapsed)

    def test_real_macos_sample_markers_addresses_and_offsets_are_canonicalized(self) -> None:
        fixture = Path(__file__).with_name("fixtures") / "macos-sample-real-callgraph.txt"
        collapsed, count, unique = _parse_macos_sample(fixture.read_text(encoding="utf-8"))
        self.assertEqual(count, 1_526)
        self.assertEqual(unique, 2)
        self.assertIn(
            "qwen_mm_python::binding::prepare_batch  (in _native.abi3.so);"
            "0x1040000a0  (in _native.abi3.so);"
            "qwen_mm_core::processor::execute_plan_into  (in _native.abi3.so) 1200\n",
            collapsed,
        )
        self.assertIn("qwen_mm_core::resize::resize_rgb  (in _native.abi3.so) 326\n", collapsed)
        self.assertNotIn("+ 172,460", collapsed)
        self.assertNotIn("[0x", collapsed)

    def test_real_rust_v0_sample_prefixes_and_source_locations_are_canonicalized(self) -> None:
        fixture = Path(__file__).with_name("fixtures") / "macos-sample-rust-v0-callgraph.txt"
        collapsed, count, unique = _parse_macos_sample(fixture.read_text(encoding="utf-8"))
        self.assertEqual(count, 100)
        self.assertEqual(unique, 2)
        self.assertIn("qwen_mm_native", collapsed)
        self.assertIn("qwen_mm_core", collapsed)
        self.assertNotIn("[0x", collapsed)
        self.assertNotIn("+ 172,460", collapsed)
        self.assertNotIn(".rs:", collapsed)
        rankings = _stack_rankings(collapsed)
        self.assertEqual(rankings["native_sample_count"], 100)
        self.assertEqual(rankings["native_sample_share"], 1.0)
        frames = [
            frame for line in collapsed.splitlines() for frame in line.rsplit(" ", 1)[0].split(";")
        ]
        fixture_core_frames = [frame for frame in frames if "qwen_mm_core" in frame]
        fixture_binding_frames = [frame for frame in frames if "qwen_mm_native" in frame]
        self.assertTrue(fixture_core_frames)
        self.assertTrue(fixture_binding_frames)
        self.assertTrue(all(map(_is_qwen_core_frame, fixture_core_frames)))
        self.assertTrue(all(map(_is_qwen_binding_frame, fixture_binding_frames)))

    def test_raw_reduction_and_rankings_are_authenticated(self) -> None:
        raw = (
            "profile_v1.setup;adapter.load 17\n"
            "qwen_mm_profile_iteration;qwen_mm.benchmark.Adapter.run 60\n"
            "qwen_mm_profile_iteration;qwen_mm_reference.profile_v1.normalize 40\n"
            "profile_v1.postcheck;adapter.run 11\n"
        )
        collapsed, count, unique = _parse_collapsed(raw, required_frame="qwen_mm_profile_iteration")
        self.assertEqual(count, 100)
        self.assertEqual(unique, 2)
        self.assertNotIn("setup", collapsed)
        self.assertNotIn("postcheck", collapsed)
        rankings = _stack_rankings(collapsed)
        self.assertEqual(rankings["sample_count"], 100)
        self.assertEqual(rankings["inclusive"][0]["frame"], "qwen_mm_profile_iteration")

    def test_validator_reopens_raw_and_recomputes_collapsed_rankings(self) -> None:
        root = Path(__file__).resolve().parents[2]
        raw = (
            "profile_v1.setup;adapter.load 17\n"
            "qwen_mm_profile_iteration;qwen_mm.benchmark.Adapter.run 60\n"
            "qwen_mm_profile_iteration;qwen_mm_reference.profile_v1.normalize 40\n"
            "profile_v1.postcheck;adapter.run 11\n"
        )
        collapsed, count, unique = _parse_collapsed(raw, required_frame="qwen_mm_profile_iteration")
        signature = {
            "input_ids": {
                "dtype": "int64",
                "shape": [1, 4],
                "strides": [32, 8],
                "nbytes": 32,
                "sha256": "a" * 64,
            }
        }
        runtime_identity = {
            "package": "qwen_mm",
            "version": "0.1.0",
            "package_artifact_sha256": "c" * 64,
            "native_module": "qwen_mm._native",
            "native_artifact_sha256": "d" * 64,
        }
        with tempfile.TemporaryDirectory(prefix=".profile-v1-test-", dir=root) as temporary:
            directory = Path(temporary)
            raw_path = directory / "capture.raw"
            collapsed_path = directory / "capture.collapsed"
            result_path = directory / "worker-result.json"
            raw_path.write_text(raw, encoding="utf-8")
            collapsed_path.write_text(collapsed, encoding="utf-8")
            worker_result = {
                "pid": 123,
                "iterations": 2,
                "input_fingerprint": "b" * 64,
                "logical_input_fingerprint": "b" * 64,
                "before_signature": signature,
                "after_signature": signature,
                "profile_alias": "qwen3-vl-8b",
                "architecture_family": "x86_64",
                "case_id": "text_short",
                "thread_budget": 1,
                "thread_settings": {
                    "budget": 1,
                    "environment": {name: "1" for name in THREAD_ENVIRONMENT_NAMES},
                    "torch": {},
                },
                "runtime_identity": runtime_identity,
                "measurement_window": {
                    "requested_duration_ns": 2_000_000_000,
                    "started_ns": 100,
                    "deadline_ns": 2_000_000_100,
                    "final_iteration_started_ns": 2_000_000_099,
                    "completed_ns": 2_000_000_110,
                    "postcheck_completed_ns": 2_000_000_120,
                },
            }
            result_path.write_text(json.dumps(worker_result), encoding="utf-8")
            worker_path = Path(__import__("qwen_mm_reference.profile_v1", fromlist=["x"]).__file__)
            sampled = {
                "profile_alias": "qwen3-vl-8b",
                "architecture_family": "x86_64",
                "case_id": "text_short",
                "thread_budget": 1,
                "thread_regime": "one",
                "input_fingerprint": "b" * 64,
                "logical_input_fingerprint": "b" * 64,
                "before_signature": signature,
                "after_signature": signature,
                "iterations": 2,
                "whole_boundary_marker": "qwen_mm_profile_iteration",
                "worker": {
                    "source": _artifact_identity(worker_path),
                    "result": _artifact_identity(result_path),
                    "command": "python -m qwen_mm_reference.profile_v1 _sample_worker --config x",
                    "target_pid": 123,
                    "runtime_identity": runtime_identity,
                },
                "sampler": {
                    "name": "py-spy",
                    "version": "py-spy 0.4.1",
                    "binary_path": SAMPLER_PROTOCOLS["x86_64"]["binary_path"],
                    "binary_sha256": hashlib.sha256(b"py-spy").hexdigest(),
                    "binary_bytes": 6,
                    "command": (
                        "/workspace/qwen-mm/.venv/bin/py-spy record --rate 99 --format raw "
                        "-o capture.raw -- python -m qwen_mm_reference.profile_v1 "
                        "_sample_worker --config x"
                    ),
                    "native": False,
                    "rate_hz": 99,
                    "duration_seconds": 2,
                    "sample_count": count,
                    "reported_sample_count": 128,
                    "unique_stacks": unique,
                    "errors": 0,
                    "install_provenance": "py-spy==0.4.1 test executable",
                },
                "artifacts": {
                    "raw": _artifact_identity(raw_path),
                    "collapsed": _artifact_identity(collapsed_path),
                },
                "rankings": _stack_rankings(collapsed),
            }
            protocol = {"samplers": SAMPLER_PROTOCOLS}
            coordinate = ("qwen3-vl-8b", "text_short", 1)
            validate_sampled_profile(
                sampled,
                protocol=protocol,
                observed_signatures={coordinate: signature},
            )
            self.assertEqual(sampled["rankings"]["native_sample_count"], 0)
            self.assertEqual(sampled["rankings"]["native_sample_share"], 0.0)

            relabeled_native = copy.deepcopy(sampled)
            relabeled_native["sampler"]["native"] = True
            with self.assertRaisesRegex(ProfileArtifactError, "frozen sampler capture"):
                validate_sampled_profile(
                    relabeled_native,
                    protocol=protocol,
                    observed_signatures={coordinate: signature},
                )

            attached = copy.deepcopy(sampled)
            attached["sampler"]["command"] = (
                "/workspace/qwen-mm/.venv/bin/py-spy record --rate 99 --duration 2 "
                "--format raw -o capture.raw --pid 123"
            )
            with self.assertRaisesRegex(ProfileArtifactError, "exactly launch"):
                validate_sampled_profile(
                    attached,
                    protocol=protocol,
                    observed_signatures={coordinate: signature},
                )

            duplicate_rate = copy.deepcopy(sampled)
            duplicate_rate["sampler"]["command"] = duplicate_rate["sampler"]["command"].replace(
                "--format raw", "--rate 99 --format raw"
            )
            with self.assertRaisesRegex(ProfileArtifactError, "exactly launch"):
                validate_sampled_profile(
                    duplicate_rate,
                    protocol=protocol,
                    observed_signatures={coordinate: signature},
                )

            relabeled_output = copy.deepcopy(sampled)
            relabeled_output["sampler"]["command"] = relabeled_output["sampler"]["command"].replace(
                "-o capture.raw", "--output capture.raw"
            )
            with self.assertRaisesRegex(ProfileArtifactError, "exactly launch"):
                validate_sampled_profile(
                    relabeled_output,
                    protocol=protocol,
                    observed_signatures={coordinate: signature},
                )

            extra_worker_token = copy.deepcopy(sampled)
            extra_worker_token["worker"]["command"] += " --extra"
            with self.assertRaisesRegex(ProfileArtifactError, "frozen authenticated shape"):
                validate_sampled_profile(
                    extra_worker_token,
                    protocol=protocol,
                    observed_signatures={coordinate: signature},
                )

            wrong_reported_total = copy.deepcopy(sampled)
            wrong_reported_total["sampler"]["reported_sample_count"] += 1
            with self.assertRaisesRegex(ProfileArtifactError, "reported sample count"):
                validate_sampled_profile(
                    wrong_reported_total,
                    protocol=protocol,
                    observed_signatures={coordinate: signature},
                )

            reported_errors = copy.deepcopy(sampled)
            reported_errors["sampler"]["errors"] = 1
            with self.assertRaisesRegex(ProfileArtifactError, "frozen sampler capture"):
                validate_sampled_profile(
                    reported_errors,
                    protocol=protocol,
                    observed_signatures={coordinate: signature},
                )

            wrong_runtime = copy.deepcopy(sampled)
            wrong_runtime["worker"]["runtime_identity"]["native_artifact_sha256"] = "e" * 64
            with self.assertRaisesRegex(ProfileArtifactError, "worker result"):
                validate_sampled_profile(
                    wrong_runtime,
                    protocol=protocol,
                    observed_signatures={coordinate: signature},
                )

            tampered_window = copy.deepcopy(worker_result)
            tampered_window["measurement_window"]["deadline_ns"] += 1
            result_path.write_text(json.dumps(tampered_window), encoding="utf-8")
            invalid_window = copy.deepcopy(sampled)
            invalid_window["worker"]["result"] = _artifact_identity(result_path)
            with self.assertRaisesRegex(ProfileArtifactError, "2 s protocol"):
                validate_sampled_profile(
                    invalid_window,
                    protocol=protocol,
                    observed_signatures={coordinate: signature},
                )
            result_path.write_text(json.dumps(worker_result), encoding="utf-8")

            tampered = copy.deepcopy(sampled)
            tampered["rankings"]["self"][0]["samples"] += 1
            with self.assertRaisesRegex(ProfileArtifactError, "rankings"):
                validate_sampled_profile(
                    tampered,
                    protocol=protocol,
                    observed_signatures={coordinate: signature},
                )

            collapsed_path.write_text(collapsed + "forged 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ProfileArtifactError, "hash/size"):
                validate_sampled_profile(
                    sampled,
                    protocol=protocol,
                    observed_signatures={coordinate: signature},
                )


class BundleValidationTests(unittest.TestCase):
    def _bundle(self, directory: Path) -> dict:
        root = Path(__file__).resolve().parents[2]
        workload_path = root / "benchmarks/workloads-v2.json"
        workload = load_workload(workload_path)
        cases = select_cases(
            workload,
            case_ids=("text_short", "text_long", "image1", "image24", "ragged24", "rgb24"),
        )
        protocol = {
            "workload": workload_provenance(workload_path),
            "observed_adapter": "qwen_mm.benchmark:create_observed_adapter",
            "paired_adapter": "qwen_mm.benchmark:create_adapter",
            "profiles": ["qwen3-vl-8b", "qwen3.5-9b"],
            "cases": [case["case_id"] for case in cases],
            "thread_budgets": [1, 4],
            "thread_regimes": {"one": 1, "production": 4},
            "repetitions": 3,
            "event_capacity": 4096,
            "samplers": copy.deepcopy(SAMPLER_PROTOCOLS),
            "build_labels": ["profiled-release"],
            "production_thread_budget": 4,
        }
        payloads = {case["case_id"]: materialize_case(case) for case in cases}

        def media_scopes(payload) -> list[tuple[int, int, int, int, int]]:
            values = []
            media_index = 0
            for request_index, messages in enumerate(payload.messages):
                local: dict[int, int] = {}
                for message_index, message in enumerate(messages):
                    content = message.get("content")
                    if not isinstance(content, list):
                        continue
                    for content_index, item in enumerate(content):
                        if item.get("type") != "image":
                            continue
                        global_index = item["buffer_index"]
                        input_index = local.setdefault(global_index, len(local))
                        values.append(
                            (
                                request_index,
                                message_index,
                                content_index,
                                media_index,
                                input_index,
                            )
                        )
                        media_index += 1
            return values

        def signature(coordinate: tuple[str, str, int]) -> dict:
            digest = hashlib.sha256(repr(coordinate).encode()).hexdigest()
            return {
                "input_ids": {
                    "dtype": "int64",
                    "shape": [3, 4],
                    "strides": [32, 8],
                    "nbytes": 96,
                    "sha256": digest,
                }
            }

        worker_source = Path(__import__("qwen_mm_reference.profile_v1", fromlist=["x"]).__file__)

        def capture(architecture: str) -> dict:
            suffix = "so" if architecture == "x86_64" else "dylib"
            native = f"native-{architecture}".encode()
            package = f"package-{architecture}".encode()
            runtime = {
                "package": "qwen_mm",
                "version": "0.1.0",
                "package_artifact_sha256": hashlib.sha256(package).hexdigest(),
                "native_module": "qwen_mm._native",
                "native_artifact_sha256": hashlib.sha256(native).hexdigest(),
            }
            candidate_identity = {
                "adapter_spec": "qwen_mm.benchmark:create_adapter",
                "kind": "module",
                "resolved": True,
                "runtime_identity": runtime,
            }
            wheel_path = directory / f"profile-{architecture}.whl"
            with zipfile.ZipFile(wheel_path, "w") as wheel:
                wheel.writestr("qwen_mm/__init__.py", package)
                wheel.writestr(f"qwen_mm/_native.{suffix}", native)
            log_path = directory / f"build-{architecture}.log"
            log_path.write_text("profiled-release build completed\n", encoding="utf-8")
            phase_path = directory / f"phase-c-{architecture}.json"
            phase_report = {
                "schema_id": "qwen-mm-phase-c-conformance-overlay-v2",
                "schema_version": 2,
                "status": "pass",
                "passed": True,
                "current_candidate": {
                    "revision": "test-revision",
                    "capture": {"runtime_identity": runtime},
                },
            }
            phase_path.write_text(json.dumps(phase_report), encoding="utf-8")
            phase_identity = _artifact_identity(phase_path)
            evidence = {
                "schema_id": "qwen-mm-phase-c-conformance-overlay-v2",
                "schema_version": 2,
                "candidate_runtime_identity": runtime,
            }
            gate = {
                "status": "pass",
                "reason_codes": [],
                "assets_root": "reference/testdata/assets",
                "report_sha256": phase_identity["sha256"],
                "evidence": evidence,
            }
            gate["gate_fingerprint"] = hashlib.sha256(
                json.dumps(
                    {"report_sha256": phase_identity["sha256"], "evidence": evidence},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            observations = []
            signatures: dict[tuple[str, str, int], dict] = {}
            fingerprints: dict[tuple[str, str, int], str] = {}
            logical_fingerprints: dict[tuple[str, str, int], str] = {}
            for profile in protocol["profiles"]:
                for case in cases:
                    payload = payloads[case["case_id"]]
                    scopes = media_scopes(payload)
                    for budget in (1, 4):
                        coordinate = (profile, case["case_id"], budget)
                        signatures[coordinate] = signature(coordinate)
                        fingerprints[coordinate] = payload.input_fingerprint
                        logical_fingerprints[coordinate] = payload.logical_input_fingerprint
                        for repetition in range(3):
                            observations.append(
                                {
                                    "profile_alias": profile,
                                    "case_id": case["case_id"],
                                    "release_name": case.get("release_name"),
                                    "boundary": case["boundary"],
                                    "thread_budget": budget,
                                    "thread_regime": "one" if budget == 1 else "production",
                                    "thread_settings": thread_settings(budget),
                                    "repetition": repetition,
                                    "media_count": len(payload.buffers),
                                    "input_fingerprint": payload.input_fingerprint,
                                    "logical_input_fingerprint": (
                                        payload.logical_input_fingerprint
                                    ),
                                    "output_signature": signatures[coordinate],
                                    "observation": complete_observation(
                                        scopes, len(payload.messages)
                                    ),
                                }
                            )
            sampled_profiles = []
            for coordinate, output in signatures.items():
                profile, case_id, budget = coordinate
                slug = f"{architecture}-{profile}-{case_id}-{budget}"
                raw_path = directory / f"{slug}.raw"
                if architecture == "x86_64":
                    raw = (
                        "qwen_mm_profile_iteration;"
                        "qwen_mm.benchmark.Adapter.run;"
                        "qwen_mm_reference.profile_v1.normalize 100\n"
                    )
                    version = "py-spy 0.4.1"
                    binary_path = SAMPLER_PROTOCOLS["x86_64"]["binary_path"]
                    command = (
                        f"{binary_path} record --rate 99 --format raw -o x -- "
                        "python -m qwen_mm_reference.profile_v1 _sample_worker --config x"
                    )
                    sampler_extra = {"rate_hz": 99}
                else:
                    raw = """Analysis of sampling python every 1 milliseconds
Call graph:
    1000 Thread_1
      1000 _PyEval_EvalFrameDefault  (in Python) + 4  [0x1]
      + 1000 qwen_mm_python::binding::prepare_batch  (in _native.dylib) + 4  [0x2]
          1000 qwen_mm_core::processor::execute  (in _native.dylib) + 8  [0x3]

Total number in stack (recursive counted multiple, when >=5):
"""
                    version = "PROGRAM:sample PROJECT:SamplingTools-test"
                    binary_path = "/usr/bin/sample"
                    command = f"{binary_path} 123 2 1 -mayDie -file x"
                    sampler_extra = {
                        "interval_ms": 1,
                        "conversion_version": "macos-sample-callgraph-v1",
                    }
                collapsed, count, unique = (
                    _parse_collapsed(raw) if architecture == "x86_64" else _parse_macos_sample(raw)
                )
                collapsed_path = directory / f"{slug}.collapsed"
                result_path = directory / f"{slug}.worker.json"
                raw_path.write_text(raw, encoding="utf-8")
                collapsed_path.write_text(collapsed, encoding="utf-8")
                worker_result = {
                    "pid": 123,
                    "iterations": 2,
                    "input_fingerprint": fingerprints[coordinate],
                    "logical_input_fingerprint": logical_fingerprints[coordinate],
                    "before_signature": output,
                    "after_signature": output,
                    "profile_alias": profile,
                    "case_id": case_id,
                    "thread_budget": budget,
                    "thread_settings": thread_settings(budget),
                    "runtime_identity": runtime,
                    **(
                        {
                            "measurement_window": {
                                "requested_duration_ns": 2_000_000_000,
                                "started_ns": 100,
                                "deadline_ns": 2_000_000_100,
                                "final_iteration_started_ns": 2_000_000_099,
                                "completed_ns": 2_000_000_110,
                                "postcheck_completed_ns": 2_000_000_120,
                            }
                        }
                        if architecture == "x86_64"
                        else {}
                    ),
                }
                result_path.write_text(json.dumps(worker_result), encoding="utf-8")
                sampled_profiles.append(
                    {
                        "profile_alias": profile,
                        "architecture_family": architecture,
                        "case_id": case_id,
                        "thread_budget": budget,
                        "thread_regime": "one" if budget == 1 else "production",
                        "input_fingerprint": fingerprints[coordinate],
                        "logical_input_fingerprint": logical_fingerprints[coordinate],
                        "before_signature": output,
                        "after_signature": output,
                        "iterations": 2,
                        "whole_boundary_marker": "qwen_mm_profile_iteration",
                        "worker": {
                            "source": _artifact_identity(worker_source),
                            "result": _artifact_identity(result_path),
                            "command": "python -m qwen_mm_reference.profile_v1 _sample_worker --config x",
                            "target_pid": 123,
                            "runtime_identity": runtime,
                        },
                        "sampler": {
                            "name": SAMPLER_PROTOCOLS[architecture]["name"],
                            "version": version,
                            "binary_path": binary_path,
                            "binary_sha256": hashlib.sha256(binary_path.encode()).hexdigest(),
                            "binary_bytes": 100,
                            "command": command,
                            "native": SAMPLER_PROTOCOLS[architecture]["native"],
                            "duration_seconds": 2,
                            "sample_count": count,
                            **(
                                {"reported_sample_count": count} if architecture == "x86_64" else {}
                            ),
                            "unique_stacks": unique,
                            "errors": 0,
                            "install_provenance": "synthetic authenticated sampler fixture",
                            **sampler_extra,
                        },
                        "artifacts": {
                            "raw": _artifact_identity(raw_path),
                            "collapsed": _artifact_identity(collapsed_path),
                        },
                        "rankings": _stack_rankings(collapsed),
                    }
                )
            pairs = []
            for coordinate, output in signatures.items():
                profile, case_id, budget = coordinate
                pairs.append(
                    {
                        "thread_budget": budget,
                        "implementations": {
                            "candidate": {
                                "profile_alias": profile,
                                "case_id": case_id,
                                "input_fingerprint": fingerprints[coordinate],
                                "logical_input_fingerprint": logical_fingerprints[coordinate],
                                "output_signature": output,
                            },
                            "reference": {},
                        },
                    }
                )
            benchmark = {
                "schema_id": "qwen-mm-benchmark-result-v3",
                "schema_version": 3,
                "architecture_family": architecture,
                "workload": copy.deepcopy(protocol["workload"]),
                "protocol": {
                    "candidate_adapter": "qwen_mm.benchmark:create_adapter",
                    "candidate_identity": candidate_identity,
                    "self_test_only": False,
                    "build_labels": ["profiled-release"],
                    "thread_regimes": ["one", "production"],
                    "thread_budget_mapping": {"one": 1, "production": 4},
                    "profiles": list(protocol["profiles"]),
                    "cases": list(protocol["cases"]),
                },
                "release_eligibility": {
                    "releasable": False,
                    "performance_status": "diagnostic_only",
                    "phase_c": gate,
                },
                "pairs": pairs,
            }
            benchmark_path = directory / f"benchmark-{architecture}.json"
            benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
            host = {
                "system": "Darwin" if architecture == "arm64" else "Linux",
                "release": "test",
                "machine": "arm64" if architecture == "arm64" else "x86_64",
                "architecture_family": architecture,
                "hostname": "test",
                "cpu_description": "Apple test" if architecture == "arm64" else "Intel test",
                "logical_cpu_count": 8,
                "affinity_cpus": None if architecture == "arm64" else list(range(8)),
                "effective_cpu_count": 8,
                "cgroup_v2": {
                    "cpu.max": None if architecture == "arm64" else "max 100000",
                    "cpuset.cpus.effective": None if architecture == "arm64" else "0-7",
                },
                "python": "3.11.15",
            }
            return {
                "host": host,
                "source": {
                    "revision": "test-revision",
                    "tree_sha256": "f" * 64,
                    "file_count": 1,
                    "total_bytes": 1,
                    "dirty": False,
                    "clean_attestation": "git_status_porcelain",
                },
                "build": {
                    "label": "profiled-release",
                    "mode": "release",
                    "command": "CARGO_PROFILE_RELEASE_DEBUG=1 CARGO_PROFILE_RELEASE_OPT_LEVEL=3 CARGO_PROFILE_RELEASE_STRIP=none RUSTFLAGS=-Cforce-frame-pointers=yes maturin build --release --locked",
                    "candidate_identity": candidate_identity,
                    "log": _artifact_identity(log_path),
                    "wheel": _artifact_identity(wheel_path),
                },
                "capture_command": "profile_v1 capture --synthetic-test",
                "profiler": {
                    "name": "qwen-mm-observation-v1",
                    "kind": "bounded_stage_instrumentation",
                    "timing": "inclusive and exclusive monotonic wall nanoseconds",
                },
                "phase_c_report": phase_identity,
                "paired_benchmark": {
                    **_artifact_identity(benchmark_path),
                    "architecture_family": architecture,
                    "schema_id": "qwen-mm-benchmark-result-v3",
                    "schema_version": 3,
                    "candidate_identity": candidate_identity,
                },
                "observations": observations,
                "sampled_profiles": sampled_profiles,
            }

        bundle = {
            "schema_id": "qwen-mm-profile-bundle-v2",
            "schema_version": 2,
            "created_at": datetime.now(UTC).isoformat(),
            "protocol": protocol,
            "captures": [capture("arm64"), capture("x86_64")],
            "summary": {},
        }
        excluded = {
            path.relative_to(root).as_posix() for path in directory.rglob("*") if path.is_file()
        }
        source_digest, source_files, source_bytes = _source_tree_digest_excluding(excluded)
        for item in bundle["captures"]:
            item["source"].update(
                tree_sha256=source_digest,
                file_count=source_files,
                total_bytes=source_bytes,
            )
        bundle["summary"] = summarize_bundle(bundle)
        return bundle

    @staticmethod
    def _git_output(*args: str) -> str:
        return "later-descendant" if args[:2] == ("rev-parse", "HEAD") else ""

    def test_source_identity_rejects_declared_fallback_when_git_metadata_is_present(self) -> None:
        with tempfile.TemporaryDirectory(prefix="profile-source-unreadable-") as temporary:
            root = Path(temporary)
            (root / ".git").mkdir()
            with (
                patch("qwen_mm_reference.profile_v1.repository_root", return_value=root),
                patch("qwen_mm_reference.profile_v1._git_output", return_value=None),
                self.assertRaisesRegex(ProfileArtifactError, "cannot be authenticated"),
            ):
                _source_identity(
                    revision="f" * 40,
                    source_digest="e" * 64,
                    declared_clean=True,
                )

    def test_recorded_source_digest_is_independent_of_current_head(self) -> None:
        with tempfile.TemporaryDirectory(prefix="profile-source-git-") as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
            subprocess.run(
                ["git", "config", "user.email", "profile-test@example.invalid"],
                cwd=root,
                check=True,
            )
            subprocess.run(["git", "config", "user.name", "Profile Test"], cwd=root, check=True)
            (root / ".gitignore").write_text(
                ".DS_Store\nreference/results/conformance-local/\n", encoding="utf-8"
            )
            (root / "source.txt").write_text("recorded\n", encoding="utf-8")
            subprocess.run(["git", "add", ".gitignore", "source.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "--quiet", "-m", "recorded"], cwd=root, check=True)
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            expected = committed_source_tree_digest(root, revision)
            with patch("qwen_mm_reference.profile_v1.repository_root", return_value=root):
                with (
                    patch.dict(
                        "qwen_mm_reference.profile_v1.os.environ",
                        {"QWEN_MM_SOURCE_REVISION": "f" * 40},
                        clear=False,
                    ),
                    self.assertRaisesRegex(ProfileArtifactError, "differs from observed"),
                ):
                    _source_identity()
                self.assertEqual(_source_tree_digest_at_revision(revision, set()), expected)

                (root / ".DS_Store").write_bytes(b"ignored metadata")
                generated = root / "reference/results/conformance-local/result.json"
                generated.parent.mkdir(parents=True)
                generated.write_text("{}\n", encoding="utf-8")
                self.assertNotEqual(source_tree_digest(root), expected)
                source = _source_identity()
                self.assertEqual(
                    (
                        source["tree_sha256"],
                        source["file_count"],
                        source["total_bytes"],
                    ),
                    expected,
                )
                self.assertFalse(source["dirty"])

                evidence = root / "benchmarks/profile-evidence-v1/capture.json"
                evidence.parent.mkdir(parents=True)
                evidence.write_text("{}\n", encoding="utf-8")
                subprocess.run(["git", "add", "--force", str(evidence)], cwd=root, check=True)
                subprocess.run(["git", "commit", "--quiet", "-m", "evidence"], cwd=root, check=True)
                descendant = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=root,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                self.assertNotEqual(descendant, revision)
                self.assertEqual(_source_tree_digest_at_revision(revision, set()), expected)
                self.assertEqual(_source_tree_digest_at_revision(descendant, set()), expected)

                (root / "source.txt").write_text("tampered\n", encoding="utf-8")
                subprocess.run(
                    ["git", "commit", "--all", "--quiet", "-m", "changed"], cwd=root, check=True
                )
                changed = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=root,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                self.assertNotEqual(_source_tree_digest_at_revision(changed, set()), expected)
                with self.assertRaisesRegex(ProfileArtifactError, "missing"):
                    _source_tree_digest_at_revision("0" * 40, set())

    @patch("qwen_mm_reference.benchmark_protocol._encode_rgb", return_value=b"encoded")
    def test_portable_cross_arch_bundle_and_tamper_rejection(self, _encode_rgb) -> None:
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix=".profile-bundle-test-", dir=root) as temporary:
            bundle = self._bundle(Path(temporary))
            source = bundle["captures"][0]["source"]
            source_tuple = (
                source["tree_sha256"],
                source["file_count"],
                source["total_bytes"],
            )
            with (
                patch("qwen_mm_reference.profile_v1.validate_result_portable"),
                patch(
                    "qwen_mm_reference.profile_v1.validate_result_authenticated_portable"
                ) as portable_auth,
                patch("qwen_mm_reference.profile_v1.validate_phase_c_overlay"),
                patch("qwen_mm_reference.profile_v1._git_output", side_effect=self._git_output),
                patch(
                    "qwen_mm_reference.profile_v1._source_tree_digest_at_revision",
                    return_value=source_tuple,
                ),
            ):
                validate_bundle(bundle, require_arm_x86=True)
                self.assertEqual(portable_auth.call_count, 2)
                merged = merge_bundles(
                    [
                        {
                            **copy.deepcopy(bundle),
                            "captures": [copy.deepcopy(bundle["captures"][0])],
                            "summary": summarize_bundle(
                                {"captures": [copy.deepcopy(bundle["captures"][0])]}
                            ),
                        },
                        {
                            **copy.deepcopy(bundle),
                            "captures": [copy.deepcopy(bundle["captures"][1])],
                            "summary": summarize_bundle(
                                {"captures": [copy.deepcopy(bundle["captures"][1])]}
                            ),
                        },
                    ]
                )
                self.assertEqual(
                    {capture["host"]["architecture_family"] for capture in merged["captures"]},
                    {"arm64", "x86_64"},
                )
                self.assertNotEqual(
                    bundle["captures"][0]["build"]["candidate_identity"]["runtime_identity"][
                        "native_artifact_sha256"
                    ],
                    bundle["captures"][1]["build"]["candidate_identity"]["runtime_identity"][
                        "native_artifact_sha256"
                    ],
                )
                self.assertEqual(len(bundle["captures"][0]["observations"]), 72)
                self.assertEqual(len(bundle["captures"][0]["sampled_profiles"]), 24)

                def relabel_phase_c_v1(value: dict) -> None:
                    capture = value["captures"][0]
                    benchmark_path = root / capture["paired_benchmark"]["path"]
                    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
                    benchmark["release_eligibility"]["phase_c"]["evidence"].update(
                        schema_id="qwen-mm-phase-c-conformance-report-v1",
                        schema_version=1,
                    )
                    relabeled_path = Path(temporary) / "phase-c-v1-relabeled-benchmark.json"
                    relabeled_path.write_text(json.dumps(benchmark), encoding="utf-8")
                    capture["paired_benchmark"].update(_artifact_identity(relabeled_path))

                mutations = {
                    "dirty_source": lambda value: value["captures"][0]["source"].update(dirty=True),
                    "source_digest": lambda value: value["captures"][0]["source"].update(
                        tree_sha256="0" * 64
                    ),
                    "source_revision": lambda value: value["captures"][0]["source"].update(
                        revision="spoofed"
                    ),
                    "wrong_host": lambda value: value["captures"][1]["host"].update(
                        system="Darwin"
                    ),
                    "workload_hash": lambda value: value["protocol"]["workload"].update(
                        sha256="0" * 64
                    ),
                    "phase_hash": lambda value: value["captures"][0]["phase_c_report"].update(
                        sha256="0" * 64
                    ),
                    "phase_c_v1_relabel": relabel_phase_c_v1,
                    "missing_observation": lambda value: value["captures"][0]["observations"].pop(),
                    "duplicate_sample_artifact": lambda value: value["captures"][0][
                        "sampled_profiles"
                    ][1]["artifacts"].update(
                        raw=value["captures"][0]["sampled_profiles"][0]["artifacts"]["raw"]
                    ),
                    "swapped_sampler": lambda value: value["captures"][0]["sampled_profiles"][0][
                        "sampler"
                    ].update(name="py-spy"),
                    "malformed_output_signature": lambda value: value["captures"][0][
                        "observations"
                    ][0].update(output_signature={"input_ids": None}),
                    "logical_input_fingerprint": lambda value: value["captures"][0]["observations"][
                        0
                    ].update(logical_input_fingerprint="0" * 64),
                    "malformed_workload_identity": lambda value: value["protocol"].update(
                        workload={}
                    ),
                }
                for name, mutate in mutations.items():
                    with self.subTest(tamper=name):
                        tampered = copy.deepcopy(bundle)
                        mutate(tampered)
                        with self.assertRaises(ProfileArtifactError):
                            validate_bundle(tampered, require_arm_x86=True)

                sampled_runtime_tamper = copy.deepcopy(bundle)
                sampled_worker = sampled_runtime_tamper["captures"][0]["sampled_profiles"][0][
                    "worker"
                ]
                wrong_runtime = {
                    **sampled_worker["runtime_identity"],
                    "native_artifact_sha256": "e" * 64,
                }
                original_result_path = root / sampled_worker["result"]["path"]
                tampered_result = json.loads(original_result_path.read_text(encoding="utf-8"))
                tampered_result["runtime_identity"] = wrong_runtime
                tampered_result_path = Path(temporary) / "runtime-tampered-worker.json"
                tampered_result_path.write_text(json.dumps(tampered_result), encoding="utf-8")
                sampled_worker["runtime_identity"] = wrong_runtime
                sampled_worker["result"] = _artifact_identity(tampered_result_path)
                with self.assertRaisesRegex(ProfileArtifactError, "profiled build runtime"):
                    validate_bundle(sampled_runtime_tamper, require_arm_x86=True)

                packaged = copy.deepcopy(bundle)
                for capture in packaged["captures"]:
                    capture["source"].update(
                        file_count=None,
                        total_bytes=None,
                        clean_attestation="packaged_source_from_clean_revision_and_digest",
                    )
                validate_bundle(packaged, require_arm_x86=True)

                missing_revision = copy.deepcopy(bundle)
                for capture in missing_revision["captures"]:
                    capture["source"]["revision"] = "0" * 40
                with (
                    patch(
                        "qwen_mm_reference.profile_v1._source_tree_digest_at_revision",
                        side_effect=ProfileArtifactError(
                            "profile source revision is missing or unreadable"
                        ),
                    ),
                    self.assertRaisesRegex(ProfileArtifactError, "missing"),
                ):
                    validate_bundle(missing_revision, require_arm_x86=True)

                tampered_revision = copy.deepcopy(bundle)
                for capture in tampered_revision["captures"]:
                    capture["source"]["revision"] = "1" * 40
                with (
                    patch(
                        "qwen_mm_reference.profile_v1._source_tree_digest_at_revision",
                        return_value=("0" * 64, source_tuple[1], source_tuple[2]),
                    ),
                    self.assertRaisesRegex(ProfileArtifactError, "recorded Git tree"),
                ):
                    validate_bundle(tampered_revision, require_arm_x86=True)
            with (
                patch("qwen_mm_reference.profile_v1.validate_result_portable"),
                patch(
                    "qwen_mm_reference.profile_v1.validate_phase_c_overlay",
                    side_effect=ValueError("semantic failure"),
                ),
                patch("qwen_mm_reference.profile_v1._git_output", side_effect=self._git_output),
                patch(
                    "qwen_mm_reference.profile_v1._source_tree_digest_at_revision",
                    return_value=source_tuple,
                ),
            ):
                with self.assertRaisesRegex(ProfileArtifactError, "invalid or stale"):
                    validate_bundle(bundle, require_arm_x86=True)

    @patch("qwen_mm_reference.benchmark_protocol._encode_rgb", return_value=b"encoded")
    def test_foreign_generated_encoded_exact_fingerprint_is_portable(self, _encode_rgb) -> None:
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix=".profile-portable-input-", dir=root) as temporary:
            temporary_path = Path(temporary)
            bundle = self._bundle(temporary_path)
            foreign = next(
                capture
                for capture in bundle["captures"]
                if capture["host"]["architecture_family"] != architecture_family()
            )
            foreign_exact = "e" * 64
            for operation in foreign["observations"]:
                if operation["case_id"] == "ragged24":
                    operation["input_fingerprint"] = foreign_exact
            for index, sampled in enumerate(foreign["sampled_profiles"]):
                if sampled["case_id"] != "ragged24":
                    continue
                sampled["input_fingerprint"] = foreign_exact
                result_path = root / sampled["worker"]["result"]["path"]
                result = json.loads(result_path.read_text(encoding="utf-8"))
                result["input_fingerprint"] = foreign_exact
                portable_result_path = temporary_path / f"foreign-ragged-worker-{index}.json"
                portable_result_path.write_text(json.dumps(result), encoding="utf-8")
                sampled["worker"]["result"] = _artifact_identity(portable_result_path)

            benchmark_path = root / foreign["paired_benchmark"]["path"]
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
            for pair in benchmark["pairs"]:
                candidate = pair["implementations"]["candidate"]
                if candidate["case_id"] != "ragged24":
                    continue
                logical = candidate["logical_input_fingerprint"]
                for implementation in pair["implementations"].values():
                    implementation["input_fingerprint"] = foreign_exact
                    implementation["logical_input_fingerprint"] = logical
            portable_benchmark_path = temporary_path / "foreign-benchmark.json"
            portable_benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
            foreign["paired_benchmark"].update(_artifact_identity(portable_benchmark_path))

            source = bundle["captures"][0]["source"]
            source_tuple = (
                source["tree_sha256"],
                source["file_count"],
                source["total_bytes"],
            )
            with (
                patch("qwen_mm_reference.profile_v1.validate_result_portable"),
                patch("qwen_mm_reference.profile_v1.validate_result_authenticated_portable"),
                patch("qwen_mm_reference.profile_v1.validate_phase_c_overlay"),
                patch("qwen_mm_reference.profile_v1._git_output", side_effect=self._git_output),
                patch(
                    "qwen_mm_reference.profile_v1._source_tree_digest_at_revision",
                    return_value=source_tuple,
                ),
            ):
                validate_bundle(bundle, require_arm_x86=True)

                mismatched = copy.deepcopy(bundle)
                mismatched_foreign = next(
                    capture
                    for capture in mismatched["captures"]
                    if capture["host"]["architecture_family"] != architecture_family()
                )
                sampled = next(
                    item
                    for item in mismatched_foreign["sampled_profiles"]
                    if item["case_id"] == "ragged24"
                )
                sampled["input_fingerprint"] = "f" * 64
                result_path = root / sampled["worker"]["result"]["path"]
                result = json.loads(result_path.read_text(encoding="utf-8"))
                result["input_fingerprint"] = "f" * 64
                mismatched_result_path = temporary_path / "mismatched-ragged-worker.json"
                mismatched_result_path.write_text(json.dumps(result), encoding="utf-8")
                sampled["worker"]["result"] = _artifact_identity(mismatched_result_path)
                with self.assertRaisesRegex(
                    ProfileArtifactError, "sampled and observed input fingerprints differ"
                ):
                    validate_bundle(mismatched, require_arm_x86=True)


if __name__ == "__main__":
    unittest.main()
