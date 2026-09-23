"""Require PyPI to serve exactly the wheels selected by the verified release jobs."""

import hashlib
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen


def main() -> None:
    for wheel in sorted(Path(sys.argv[1]).glob("*.whl")):
        name, version = wheel.name.split("-")[:2]
        url = f"https://pypi.org/pypi/{name.replace('_', '-')}/{version}/json"
        for attempt in range(6):
            try:
                with urlopen(url, timeout=30) as response:
                    data = json.load(response)
                matches = [entry for entry in data["urls"] if entry["filename"] == wheel.name]
                if matches:
                    break
            except HTTPError as error:
                if error.code != 404:
                    raise
            if attempt == 5:
                raise RuntimeError(f"published wheel unavailable: {wheel.name}")
            time.sleep(5)
        expected = hashlib.sha256(wheel.read_bytes()).hexdigest()
        if len(matches) != 1 or matches[0]["digests"]["sha256"] != expected:
            raise RuntimeError(f"published hash mismatch: {wheel.name}")
        print(f"Verified PyPI SHA-256: {wheel.name}")


if __name__ == "__main__":
    main()
