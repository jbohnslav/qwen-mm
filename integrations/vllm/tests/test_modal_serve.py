"""Modal imports controllers at /root on workers, without the local checkout."""

import sys
from pathlib import Path
from types import SimpleNamespace


def test_worker_import_does_not_resolve_local_repository(monkeypatch):
    class App:
        def __init__(self, *args):
            pass

        def function(self, **kwargs):
            return lambda fn: fn

        def local_entrypoint(self):
            return lambda fn: fn

    fake = SimpleNamespace(
        App=App, is_local=lambda: False, Volume=SimpleNamespace(from_name=lambda *a, **kw: object())
    )
    monkeypatch.setitem(sys.modules, "modal", fake)
    source = (Path(__file__).parents[1] / "scripts/modal_serve.py").read_text()
    namespace = {"__file__": "/root/modal_serve.py", "__name__": "modal_serve"}
    exec(compile(source, "/root/modal_serve.py", "exec"), namespace)
    assert callable(namespace["experiment"])
