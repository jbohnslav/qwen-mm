"""Adapters from the paired benchmark payload to the installed native wheel.

The normal factory deliberately calls :meth:`Processor.prepare_batch`; profile
capture uses the observed factory so instrumentation is never silently added to
ordinary paired timings.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import Processor

_PROFILE_SNAPSHOTS = {
    "qwen3-vl-8b": (
        "models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
    ),
    "qwen3.5-9b": ("models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
}


def _assets_root() -> Path:
    configured = os.environ.get("QWEN_MM_ASSETS_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.cwd() / "reference" / ".cache" / "huggingface").resolve()


def _asset_directory(profile_alias: str) -> Path:
    try:
        relative = _PROFILE_SNAPSHOTS[profile_alias]
    except KeyError as error:
        raise ValueError(f"unsupported qwen-mm benchmark profile: {profile_alias}") from error
    directory = _assets_root() / relative
    if not directory.is_dir():
        raise FileNotFoundError(
            f"hash-pinned assets for {profile_alias} are missing at {directory}; "
            "set QWEN_MM_ASSETS_ROOT to the Hugging Face cache root"
        )
    return directory


def _binding_requests(payload: Any) -> list[dict[str, Any]]:
    """Map global immutable benchmark buffers to request-local binding slots."""

    requests: list[dict[str, Any]] = []
    for source_messages in payload.messages:
        messages = copy.deepcopy(list(source_messages))
        referenced: list[int] = []
        local_indices: dict[int, int] = {}
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if item.get("type") != "image":
                    continue
                global_index = item.pop("buffer_index")
                if global_index not in local_indices:
                    local_indices[global_index] = len(referenced)
                    referenced.append(global_index)
                item["input_index"] = local_indices[global_index]
        images = []
        for index in referenced:
            media = payload.buffers[index]
            if media.kind == "encoded":
                images.append(
                    {
                        "data": media.value,
                        "format": str(media.format).lower(),
                    }
                )
            elif media.kind == "rgb":
                images.append(media.value)
            else:
                raise ValueError(f"unsupported benchmark media kind: {media.kind}")
        request: dict[str, Any] = {
            "messages": messages,
            "options": {"add_generation_prompt": True},
        }
        if images:
            request["images"] = images
        requests.append(request)
    return requests


def _report_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise TypeError("native observation JSON must contain an object")
        return parsed
    for attribute in ("report", "observation", "observation_report"):
        candidate = getattr(value, attribute, None)
        if candidate is not None:
            return _report_mapping(candidate() if callable(candidate) else candidate)
    for method in ("to_json", "json"):
        candidate = getattr(value, method, None)
        if callable(candidate):
            return _report_mapping(candidate())
    raise TypeError("native observed call did not return a machine-readable observation report")


class NativeBenchmarkAdapter:
    """One-call installed-wheel adapter for the A5 benchmark protocol."""

    def __init__(self, context: Any, *, observed: bool = False, event_capacity: int = 4096) -> None:
        if event_capacity < 0:
            raise ValueError("event_capacity must be non-negative")
        self.name = f"qwen-mm-native-{context.build_label}"
        self._processor = Processor(
            context.profile_alias,
            _asset_directory(context.profile_alias),
        )
        self._observed = observed
        self._event_capacity = event_capacity
        self._metrics: dict[str, Any] = {}
        self._observation: dict[str, Any] | None = None

    def run(self, payload: Any) -> Mapping[str, Any]:
        requests = _binding_requests(payload)
        if self._observed:
            result = self._processor.prepare_batch_observed(
                requests,
                event_capacity=self._event_capacity,
            )
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError("prepare_batch_observed must return (PreparedBatch, report)")
            prepared, raw_report = result
            report = _report_mapping(raw_report)
            allocations = report.get("allocations", {})
            calls = report.get("calls", {})
            self._observation = report
            self._metrics = {
                "allocation_count": allocations.get("allocation_count"),
                "copy_count": allocations.get("copy_count"),
                "copy_count_scope": report.get("counter_scope"),
                "transient_live_bytes": allocations.get("peak_transient_live_bytes"),
                "cache_supported": False,
                "copied_bytes": allocations.get("copied_bytes"),
                "retained_final_output_bytes": allocations.get("retained_final_output_bytes"),
                "calls": calls,
            }
        else:
            prepared = self._processor.prepare_batch(requests)
            self._observation = None
            self._metrics = {
                "allocation_count": None,
                "copy_count": None,
                "copy_count_scope": "not collected on unobserved paired benchmark path",
                "transient_live_bytes": None,
                "cache_supported": False,
            }
        return prepared.arrays

    def metrics(self) -> Mapping[str, Any]:
        return self._metrics

    def observation_report(self) -> Mapping[str, Any]:
        if self._observation is None:
            raise RuntimeError("the adapter has not completed an observed call")
        return self._observation


def create_adapter(context: Any) -> NativeBenchmarkAdapter:
    """Create the instrumentation-free adapter used by paired A5 timings."""

    return NativeBenchmarkAdapter(context)


def create_observed_adapter(context: Any) -> NativeBenchmarkAdapter:
    """Create the bounded observed adapter used by D1 profile capture."""

    return NativeBenchmarkAdapter(context, observed=True)


__all__ = [
    "NativeBenchmarkAdapter",
    "create_adapter",
    "create_observed_adapter",
]
