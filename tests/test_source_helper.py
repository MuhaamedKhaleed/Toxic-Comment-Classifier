"""Offline contract checks for the explicit source helper."""

import importlib.util
import io
import sys
from pathlib import Path

import pytest


def helper():
    spec = importlib.util.spec_from_file_location(
        "source_helper", Path(__file__).parents[1] / "scripts/fetch_real_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_has_no_network_or_output(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Import must not download")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    monkeypatch.chdir(tmp_path)
    helper()
    assert not list(tmp_path.iterdir())


def test_existing_output_refused(monkeypatch, tmp_path):
    module = helper()
    monkeypatch.setattr(sys, "argv", ["fetch", "--output", str(tmp_path)])
    with pytest.raises(FileExistsError):
        module.main()


def test_changed_source_fails_closed(monkeypatch, tmp_path):
    module = helper()
    module.SOURCES = [("source.bin", "https://example.invalid/source", "0" * 64)]
    monkeypatch.setattr(
        module.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"changed source")
    )
    out = tmp_path / "new"
    monkeypatch.setattr(sys, "argv", ["fetch", "--output", str(out)])
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.main()
    assert not (out / "source.bin").exists()
    assert not (out / "source-manifest.json").exists()
