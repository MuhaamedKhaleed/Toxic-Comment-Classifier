"""The optional quality recipe must never train or download on import."""

import importlib.util
from pathlib import Path
import subprocess
import sys


def test_quality_import_has_no_optional_dependencies():
    path = Path(__file__).parents[1] / "scripts" / "quality_cpu.py"
    spec = importlib.util.spec_from_file_location("optional_quality", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def test_quality_refuses_existing_output_before_loading_data(tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "quality_cpu.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--data",
            str(tmp_path / "absent"),
            "--output",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Output exists" in result.stderr
    assert not list(tmp_path.iterdir())
