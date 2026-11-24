import subprocess
import sys
import tempfile
from pathlib import Path


def test_recipe_imports_and_help():
    code = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
    with tempfile.TemporaryDirectory() as td:
        for args in (
            ["-c", f"import runpy;runpy.run_path({str(code)!r},run_name='library')"],
            [str(code), "--help"],
        ):
            result = subprocess.run([sys.executable, *args], cwd=td, capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
            assert list(Path(td).iterdir()) == []
    code = Path(__file__).resolve().parents[1] / "scripts" / "source_random_split.py"
    with tempfile.TemporaryDirectory() as td:
        for args in (
            ["-c", f"import runpy;runpy.run_path({str(code)!r},run_name='library')"],
            [str(code), "--help"],
        ):
            result = subprocess.run([sys.executable, *args], cwd=td, capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
            assert list(Path(td).iterdir()) == []
