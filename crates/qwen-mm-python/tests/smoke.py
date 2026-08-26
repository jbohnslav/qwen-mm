"""Dependency and exception smoke executed against an installed wheel."""

import pickle

import numpy as np
import qwen_mm._native as qwen_mm_native
from qwen_mm import (
    InvalidRequestError,
    Processor,
    ProfileMismatchError,
    QwenMMError,
    __version__,
    native_version,
)


def main() -> None:
    assert __version__ == "0.1.0"
    assert native_version() == __version__
    assert np.__version__ == "2.4.6"
    assert not hasattr(qwen_mm_native, "_test_native_batch_active")
    profiles = Processor.supported_profiles()
    assert [profile["profile"] for profile in profiles] == ["qwen3-vl-8b", "qwen3.5-9b"]
    assert all(len(profile["revision"]) == 40 for profile in profiles)
    assert issubclass(InvalidRequestError, QwenMMError)
    assert InvalidRequestError.category == "invalid_request"
    assert InvalidRequestError.__module__ == "qwen_mm._native"
    try:
        Processor("not-a-profile", ".")
    except ProfileMismatchError as error:
        assert error.category == "profile_mismatch"
        assert error.context == {"alias": "not-a-profile"}
        restored = pickle.loads(pickle.dumps(error))
        assert type(restored) is ProfileMismatchError
        assert restored.category == error.category
        assert restored.context == error.context
    else:
        raise AssertionError("invalid profile did not raise its stable exception")
    print(f"qwen-mm wheel smoke passed ({__version__}, NumPy {np.__version__})")


if __name__ == "__main__":
    main()
