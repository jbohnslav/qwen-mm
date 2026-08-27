"""Installed-wheel tests for familiar pinned Hugging Face construction."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import huggingface_hub
from qwen_mm import Processor

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ASSETS_ROOT = REPOSITORY_ROOT / "reference" / ".cache" / "huggingface"
PROFILE_CASES = {
    "Qwen3": (
        "qwen3-vl-8b",
        "Qwen/Qwen3-VL-8B-Instruct",
        "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    ),
    "Qwen3.5": (
        "qwen3.5-9b",
        "Qwen/Qwen3.5-9B",
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    ),
}
EXPECTED_ASSETS = {
    "Qwen3": [
        "chat_template.json",
        "config.json",
        "merges.txt",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
    ],
    "Qwen3.5": [
        "chat_template.jinja",
        "config.json",
        "merges.txt",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
    ],
}


def test_aliases_model_ids_and_cache_hits() -> None:
    for short_name, (profile, model_id, revision, _) in PROFILE_CASES.items():
        for identity in (short_name, model_id):
            processor = Processor.from_pretrained(
                identity,
                cache_dir=ASSETS_ROOT,
                local_files_only=True,
                thread_budget=1,
            )
            assert processor.profile == profile
            assert processor.model_id == model_id
            assert processor.revision == revision


def test_download_is_pinned_and_filtered() -> None:
    calls: list[dict[str, object]] = []

    def snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        for _, model_id, _, relative_snapshot in PROFILE_CASES.values():
            if kwargs["repo_id"] == model_id:
                return str(ASSETS_ROOT / relative_snapshot)
        raise AssertionError(f"unexpected repository {kwargs['repo_id']!r}")

    with patch.object(huggingface_hub, "snapshot_download", snapshot_download):
        for short_name in PROFILE_CASES:
            processor = Processor.from_pretrained(
                short_name,
                cache_dir=ASSETS_ROOT,
                local_files_only=False,
                thread_budget=2,
            )
            assert processor.thread_budget == 2

    expected_calls = []
    for short_name, (_, model_id, revision, _) in PROFILE_CASES.items():
        expected_calls.append(
            {
                "repo_id": model_id,
                "revision": revision,
                "cache_dir": ASSETS_ROOT,
                "local_files_only": False,
                "allow_patterns": EXPECTED_ASSETS[short_name],
            }
        )
    assert calls == expected_calls
    for call in calls:
        assert all(
            not pattern.endswith((".bin", ".gguf", ".safetensors"))
            for pattern in call["allow_patterns"]
        )


def test_offline_miss_is_preserved() -> None:
    with tempfile.TemporaryDirectory() as empty_cache:
        try:
            Processor.from_pretrained(
                "Qwen3",
                cache_dir=empty_cache,
                local_files_only=True,
            )
        except huggingface_hub.errors.LocalEntryNotFoundError:
            pass
        else:
            raise AssertionError("an offline cache miss unexpectedly performed or found a download")


def test_unknown_model_is_actionable() -> None:
    for unknown in ("qwen3", "Qwen/Qwen3-VL-32B-Instruct"):
        try:
            Processor.from_pretrained(unknown, local_files_only=True)
        except ValueError as error:
            message = str(error)
            assert repr(unknown) in message
            for short_name, (_, model_id, _, _) in PROFILE_CASES.items():
                assert repr(short_name) in message
                assert repr(model_id) in message
        else:
            raise AssertionError(f"unsupported model {unknown!r} was accepted")


def test_existing_cache_constructor_remains_compatible() -> None:
    processor = Processor.from_huggingface_cache(
        "qwen3-vl-8b",
        cache_directory=ASSETS_ROOT,
        thread_budget=2,
    )
    assert processor.profile == "qwen3-vl-8b"
    assert processor.thread_budget == 2


def main() -> None:
    test_aliases_model_ids_and_cache_hits()
    test_download_is_pinned_and_filtered()
    test_offline_miss_is_preserved()
    test_unknown_model_is_actionable()
    test_existing_cache_constructor_remains_compatible()
    print("qwen-mm installed pretrained-construction tests passed")


if __name__ == "__main__":
    main()
