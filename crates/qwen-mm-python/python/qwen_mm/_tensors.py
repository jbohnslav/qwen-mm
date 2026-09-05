"""Optional Torch output; importing qwen-mm never imports Torch."""

from collections.abc import Iterator, Mapping
from typing import Any


def tensor_backend(return_tensors: str | None) -> Any:
    if return_tensors in (None, "np"):
        return None
    if return_tensors != "pt":
        raise ValueError("return_tensors must be 'np' or 'pt'")
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "return_tensors='pt' requires Torch; install torch or use 'np'"
        ) from error
    return torch


class TorchBatch(Mapping[str, Any]):
    """Model-input mapping with separate metadata and explicit device transfer."""

    def __init__(self, prepared: Any, torch: Any) -> None:
        self.arrays = {key: torch.from_numpy(value) for key, value in prepared.items()}
        self.metadata = prepared.metadata

    def __getitem__(self, key: str) -> Any:
        return self.arrays[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.arrays)

    def __len__(self) -> int:
        return len(self.arrays)

    def __getattr__(self, name: str) -> Any:
        try:
            return self.arrays[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def to(self, device: Any, *, non_blocking: bool = False) -> "TorchBatch":
        """Move model inputs to a device, preserving integer and pixel dtypes."""
        self.arrays = {
            key: value.to(device=device, non_blocking=non_blocking)
            for key, value in self.arrays.items()
        }
        return self
