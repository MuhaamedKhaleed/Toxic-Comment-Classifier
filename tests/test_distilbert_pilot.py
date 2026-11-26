"""The DistilBERT pilot must never train or download on import."""

import importlib.util
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "distilbert_pilot.py"


def test_pilot_import_has_no_optional_dependencies():
    spec = importlib.util.spec_from_file_location("optional_distilbert_pilot", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def test_pilot_help_runs_without_data():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0


def test_pilot_refuses_existing_output_before_loading_data(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--data",
            str(tmp_path / "absent"),
            "--encoder",
            str(tmp_path / "absent-encoder"),
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


def _metric_fn():
    spec = importlib.util.spec_from_file_location("optional_distilbert_pilot", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.dev_metrics_from_logits


def test_metrics_use_softmax_across_both_logits():
    # Toxic row A has the higher raw logit1 (2 > 1) but a much higher logit0,
    # so softmax over both logits ranks non-toxic row B higher. A wrong
    # sigmoid(logit1) scorer would rank A higher and pass AUC 1.0 here.
    metrics = _metric_fn()
    import numpy as np

    logits = np.array([[5.0, 2.0], [0.0, 1.0]])
    truth = [1, 0]
    result = metrics(truth, logits)
    shifted = logits - logits.max(axis=1, keepdims=True)
    prob = np.exp(shifted)[:, 1] / np.exp(shifted).sum(axis=1)
    assert prob[1] > prob[0]
    assert result["roc_auc"] == 0.0
    assert result["confusion"] == {"tp": 0, "fp": 1, "fn": 1, "tn": 0}
    # Both logit1 values are positive: the legacy scorer predicted both toxic.
    assert (logits[:, 1] >= 0).tolist() == [True, True]


def test_metrics_match_standard_implementations_with_ties():
    pytest = __import__("pytest")
    pytest.importorskip("sklearn")
    metrics = _metric_fn()
    truth = [1, 1, 0, 1, 0, 0]
    probs = [0.9, 0.8, 0.8, 0.5, 0.5, 0.2]
    import math

    logits = [[0.0, math.log(p / (1 - p))] for p in probs]
    result = metrics(truth, logits)
    assert result["confusion"] == {"tp": 3, "fp": 2, "fn": 0, "tn": 1}
    assert result["f1"] == pytest.approx(0.75)
    assert result["precision"] == pytest.approx(0.6)
    assert result["recall"] == pytest.approx(1.0)
    assert result["accuracy"] == pytest.approx(2 / 3)
    # Tied scores across classes: standard sklearn values, hardcoded.
    assert result["roc_auc"] == pytest.approx(0.7777777777777778)
    assert result["average_precision"] == pytest.approx(0.7555555555555555)


def test_resume_refuses_legacy_or_changed_configuration():
    import pytest

    spec = importlib.util.spec_from_file_location("optional_distilbert_pilot", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="Legacy"):
        module.validate_resume_state({"history": []}, {})
    with pytest.raises(ValueError, match="configuration"):
        module.validate_resume_state(
            {"scorer_version": "two-logit-softmax-v1", "config": {"seed": 1}},
            {"seed": 2},
        )
    module.validate_resume_state(
        {"scorer_version": "two-logit-softmax-v1", "config": {"seed": 1}},
        {"seed": 1},
    )
