"""Guard the experimental measurements against mislabeled cache states."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from serve_experiment import payload  # noqa: E402


def image_urls(body):
    return [
        part["image_url"]["url"]
        for part in body["messages"][0]["content"]
        if part["type"] == "image_url"
    ]


def test_cold_single_inputs_do_not_reuse_warmup_or_other_repetitions():
    assert len({image_urls(payload("single", seed))[0] for seed in (0, 100, 101, 102)}) == 4


def test_repeated_images_are_equal_within_request_but_distinct_between_cases():
    first = image_urls(payload("repeat", 100))
    assert first[0] == first[1]
    assert first[0] != image_urls(payload("single", 100))[0]


def test_witness_refuses_missing_audit_fallback_or_encoder_bypass():
    import pytest
    from serve_experiment import verify_events

    processor = {
        "event": "processor",
        "grid": [[1, 16, 16]],
        "implementation": "NativeImageProcessor",
        "on_event_loop": False,
    }
    events = [processor, {"event": "vision"}, {"event": "vision_tower"}]
    assert verify_events(events, "native")["vision_tower_calls"] == 1
    for invalid in [
        [],
        events[:-1],
        events + [{"event": "vision_tower"}],
        [{**processor, "implementation": "Qwen3VLMultiModalProcessor"}, *events[1:]],
        [{**processor, "on_event_loop": True}, *events[1:]],
    ]:
        with pytest.raises(AssertionError):
            verify_events(invalid, "native")
