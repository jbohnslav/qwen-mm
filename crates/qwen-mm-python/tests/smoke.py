"""Smoke test executed against an installed development wheel."""

from qwen_mm import __version__, native_version


def main() -> None:
    assert __version__ == "0.1.0"
    assert native_version() == __version__
    print(f"qwen-mm wheel smoke passed ({__version__})")


if __name__ == "__main__":
    main()
