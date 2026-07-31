"""Prepared-pixel vLLM processor registration for the A6 seam prototype."""

from .adapter import (
    CONTRACT_ID,
    QWEN3_VL_PROFILE_FINGERPRINT,
    QWEN35_PROFILE_FINGERPRINT,
    PreparedDummyInputsBuilder,
    PreparedQwen3VLInfo,
    PreparedQwen35Info,
    PreparedQwenProcessor,
    QwenMMPreparedImageItems,
    QwenMMPreparedMultiModalProcessor,
    QwenMMPreparedPixelParser,
    register,
)

__all__ = [
    "CONTRACT_ID",
    "PreparedDummyInputsBuilder",
    "PreparedQwen3VLInfo",
    "PreparedQwen35Info",
    "PreparedQwenProcessor",
    "QWEN35_PROFILE_FINGERPRINT",
    "QWEN3_VL_PROFILE_FINGERPRINT",
    "QwenMMPreparedImageItems",
    "QwenMMPreparedMultiModalProcessor",
    "QwenMMPreparedPixelParser",
    "register",
]
