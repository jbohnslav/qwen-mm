"""Pinned Hugging Face model resolution for the public Processor facade."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import huggingface_hub


@dataclass(frozen=True)
class _PretrainedProfile:
    profile: str
    model_id: str
    revision: str
    allow_patterns: tuple[str, ...]


_PUBLIC_SHORT_NAMES = {
    "Qwen3": "qwen3-vl-8b",
    "Qwen3.5": "qwen3.5-9b",
}

# These are exactly the processor/tokenizer artifacts hash-locked by
# reference/compatibility/v1.json. Keeping this list explicit makes it impossible
# for a model-weight filename to be selected by a broad glob.
_PROFILE_ALLOW_PATTERNS = {
    "qwen3-vl-8b": (
        "chat_template.json",
        "config.json",
        "merges.txt",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
    ),
    "qwen3.5-9b": (
        "chat_template.jinja",
        "config.json",
        "merges.txt",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
    ),
}


def _pretrained_profiles(native_processor_type: type[Any]) -> tuple[_PretrainedProfile, ...]:
    profiles = []
    for record in native_processor_type.supported_profiles():
        profile = record["profile"]
        profiles.append(
            _PretrainedProfile(
                profile=profile,
                model_id=record["model_id"],
                revision=record["revision"],
                allow_patterns=_PROFILE_ALLOW_PATTERNS[profile],
            )
        )
    return tuple(profiles)


def _resolve_pretrained_profile(native_processor_type: type[Any], model: str) -> _PretrainedProfile:
    profiles = _pretrained_profiles(native_processor_type)
    by_profile = {profile.profile: profile for profile in profiles}
    by_model_id = {profile.model_id: profile for profile in profiles}

    profile_name = _PUBLIC_SHORT_NAMES.get(model)
    if profile_name is not None:
        return by_profile[profile_name]
    if model in by_model_id:
        return by_model_id[model]

    supported = [*_PUBLIC_SHORT_NAMES, *by_model_id]
    choices = ", ".join(repr(identity) for identity in supported)
    raise ValueError(f"unsupported Qwen model {model!r}; use one of: {choices}")


def _download_pretrained_snapshot(
    native_processor_type: type[Any],
    model: str,
    *,
    cache_dir: str | PathLike[str] | None,
    local_files_only: bool,
) -> tuple[_PretrainedProfile, Path]:
    profile = _resolve_pretrained_profile(native_processor_type, model)
    snapshot = huggingface_hub.snapshot_download(
        repo_id=profile.model_id,
        revision=profile.revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        allow_patterns=list(profile.allow_patterns),
    )
    return profile, Path(snapshot)
