"""Standalone smoke and no-overwrite contract, no model-quality claims."""

import json
import subprocess
import sys
from pathlib import Path


def test_standalone_synthetic_demo(tmp_path: Path) -> None:
    import pytest

    pytest.importorskip("tensorflow" if False else "torch")
    output = tmp_path / "demo"
    script = Path(__file__).resolve().parents[1] / "scripts" / "demo.py"
    result = subprocess.run(
        [sys.executable, str(script), str(output)], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["synthetic_only"] is True
    assert report["quality_claim"] is False
    assert report["scope"] and report["result"]
    before = (output / "report.json").read_bytes()
    again = subprocess.run(
        [sys.executable, str(script), str(output)], capture_output=True, timeout=30
    )
    assert again.returncode != 0
    assert (output / "report.json").read_bytes() == before
