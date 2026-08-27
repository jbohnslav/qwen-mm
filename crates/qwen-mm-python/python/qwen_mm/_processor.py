"""Public processor facade around the deterministic native implementation."""

from __future__ import annotations

from os import PathLike
from typing import Any

from . import _native
from ._media import normalize_requests
from ._pretrained import _download_pretrained_snapshot


class Processor:
    """Qwen-compatible processor with convenient Python media loading."""

    __slots__ = ("_limits", "_native")

    def __init__(
        self,
        profile: str,
        assets_directory: str | PathLike[str],
        *,
        limits: dict[str, int] | None = None,
        thread_budget: int | None = None,
    ) -> None:
        self._native = _native.Processor(
            profile,
            assets_directory,
            limits=limits,
            thread_budget=thread_budget,
        )
        self._limits = dict(limits or {})

    @classmethod
    def _from_native(
        cls,
        native: _native.Processor,
        *,
        limits: dict[str, int] | None,
    ) -> Processor:
        processor = cls.__new__(cls)
        processor._native = native
        processor._limits = dict(limits or {})
        return processor

    @classmethod
    def from_huggingface_cache(
        cls,
        profile: str,
        *,
        cache_directory: str | PathLike[str] | None = None,
        limits: dict[str, int] | None = None,
        thread_budget: int | None = None,
    ) -> Processor:
        native = _native.Processor.from_huggingface_cache(
            profile,
            cache_directory=cache_directory,
            limits=limits,
            thread_budget=thread_budget,
        )
        return cls._from_native(native, limits=limits)

    @classmethod
    def from_pretrained(
        cls,
        model: str,
        *,
        cache_dir: str | PathLike[str] | None = None,
        local_files_only: bool = False,
        limits: dict[str, int] | None = None,
        thread_budget: int | None = None,
    ) -> Processor:
        """Load a supported Qwen processor from its hash-pinned HF snapshot."""
        profile, snapshot = _download_pretrained_snapshot(
            _native.Processor,
            model,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        native = _native.Processor(
            profile.profile,
            snapshot,
            limits=limits,
            thread_budget=thread_budget,
        )
        return cls._from_native(native, limits=limits)

    @staticmethod
    def supported_profiles() -> list[dict[str, str]]:
        return _native.Processor.supported_profiles()

    def prepare(
        self,
        messages: Any,
        *,
        images: Any | None = None,
        videos: Any | None = None,
        add_generation_prompt: bool = False,
        add_vision_id: bool = False,
        tools: Any | None = None,
        enable_thinking: bool | None = None,
        padding_side: str = "right",
    ) -> _native.PreparedBatch:
        """Prepare one conversation without a one-element batch wrapper."""
        options: dict[str, Any] = {
            "add_generation_prompt": add_generation_prompt,
            "add_vision_id": add_vision_id,
        }
        if tools is not None:
            options["tools"] = tools
        if enable_thinking is not None:
            options["enable_thinking"] = enable_thinking

        request: dict[str, Any] = {"messages": messages, "options": options}
        if images is not None:
            request["images"] = images
        if videos is not None:
            request["videos"] = videos
        return self.prepare_batch([request], padding_side=padding_side)

    def prepare_batch(
        self,
        requests: Any,
        *,
        padding_side: str = "right",
    ) -> _native.PreparedBatch:
        """Prepare a heterogeneous batch, using right padding by default."""
        return self._native.prepare_batch(
            normalize_requests(requests, limits=self._limits),
            padding_side=padding_side,
        )

    def prepare_batch_observed(
        self,
        requests: Any,
        *,
        event_capacity: int = 4096,
        padding_side: str = "right",
    ) -> tuple[_native.PreparedBatch, dict[str, Any]]:
        return self._native.prepare_batch_observed(
            normalize_requests(requests, limits=self._limits),
            event_capacity=event_capacity,
            padding_side=padding_side,
        )

    @property
    def profile(self) -> str:
        return self._native.profile

    @property
    def profile_fingerprint(self) -> str:
        return self._native.profile_fingerprint

    @property
    def model_id(self) -> str:
        return self._native.model_id

    @property
    def revision(self) -> str:
        return self._native.revision

    @property
    def thread_budget(self) -> int:
        return self._native.thread_budget
