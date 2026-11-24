"""Regression checks use synthetic text/data and fake inference, never downloads."""

import json
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from toxic_comments.cli import app, classify
from toxic_comments.config import local_model_path
from toxic_comments.core import normalize_prediction, read_dataset, validate_rows
from toxic_comments.training import run_training, training_config


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{}],
        [{"comment_text": " ", "toxic": 0}],
        [{"comment_text": "text", "toxic": "yes"}],
        [{"comment_text": "text", "toxic": True}],
    ],
)
def test_bad_rows(rows):
    with pytest.raises(ValueError):
        validate_rows(rows)


def test_valid_data_and_cli(tmp_path):
    data = tmp_path / "sample.csv"
    data.write_text("comment_text,toxic\nhello,0\nangry,1\n", encoding="utf-8")
    assert validate_rows(read_dataset(data)) == {"rows": 2, "toxic": 1, "non_toxic": 1}
    result = CliRunner().invoke(app, ["check-data", "--data", str(data)])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["toxic"] == 1


@pytest.mark.parametrize(
    "member", ["../data-escape/proof.txt", "/absolute.csv", "C:/bad", "..\\bad"]
)
def test_unsafe_archive_rejected_before_read(tmp_path, member):
    path = tmp_path / "data.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, "synthetic")
        archive.writestr("train.csv", "comment_text,toxic\nhello,0\n")
    with pytest.raises(ValueError, match="unsafe"):
        read_dataset(path)
    assert not (tmp_path / "data-escape").exists()


def test_clean_zip_and_ambiguous_zip(tmp_path):
    path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data/train.csv", "comment_text,toxic\nhello,0\n")
    assert len(read_dataset(path)) == 1
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("other/train.csv", "comment_text,toxic\nhi,1\n")
    with pytest.raises(ValueError, match="exactly one"):
        read_dataset(path)


def test_missing_and_malformed_data(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        read_dataset(tmp_path / "missing.csv")
    path = tmp_path / "bad.csv"
    path.write_text("wrong,columns\nx,y\n")
    assert CliRunner().invoke(app, ["check-data", "--data", str(path)]).exit_code == 2


@pytest.mark.parametrize(
    "result",
    [
        [],
        [{}],
        [{"label": "LABEL_0", "score": 0.5}],
        [{"label": "toxic", "score": float("nan")}],
        [{"label": "toxic", "score": 2}],
        [{"label": "toxic", "score": "0.5"}],
    ],
)
def test_invalid_model_results(result):
    with pytest.raises(ValueError):
        normalize_prediction(result)


def test_fresh_local_inference(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    seen = []

    def factory(*args, **kwargs):
        assert kwargs["local_files_only"] is True

        def predict(text, **settings):
            seen.append(text)
            return [{"label": "toxic", "score": 0.75}]

        return predict

    assert classify("one", tmp_path, factory=factory) == {"label": "toxic", "score": 0.75}
    classify("two", tmp_path, factory=factory)
    assert seen == ["one", "two"]
    with pytest.raises(ValueError, match="empty"):
        classify(" ", tmp_path, factory=factory)
    with pytest.raises(ValueError, match="config.json"):
        classify("hello", tmp_path / "missing", factory=factory)


def test_external_failure_and_no_implicit_training(tmp_path, monkeypatch):
    from toxic_comments import cli

    def fail(*args, **kwargs):
        raise OSError("synthetic unavailable local model")

    monkeypatch.setattr(cli, "classify", fail)
    assert (
        CliRunner()
        .invoke(app, ["predict", "--text", "hello", "--model-dir", str(tmp_path)])
        .exit_code
        == 2
    )
    assert CliRunner().invoke(app, ["--help"]).exit_code == 0
    assert not (tmp_path / "results").exists()


def test_config_precedence_and_errors(tmp_path, monkeypatch):
    monkeypatch.delenv("TOXIC_MODEL_DIR", raising=False)
    with pytest.raises(ValueError, match="model-dir"):
        local_model_path(None)
    with pytest.raises(ValueError, match="not found"):
        local_model_path(None, tmp_path / "missing")
    env = tmp_path / "settings.env"
    env.write_text("TOXIC_MODEL_DIR=local-model\n")
    assert local_model_path(None, env) == Path("local-model")
    assert local_model_path(Path("explicit"), env) == Path("explicit")
    monkeypatch.delenv("TOXIC_MODEL_DIR")


def test_two_class_loss_and_training_validation(tmp_path):
    config = training_config()
    assert config["num_labels"] == 2
    assert config["problem_type"] == "single_label_classification"
    assert config["id2label"] == {0: "non_toxic", 1: "toxic"}
    rows = [{"comment_text": "hello", "toxic": 0}]
    with pytest.raises(ValueError, match="positive"):
        run_training(rows, tmp_path, tmp_path / "new", epochs=0)
    with pytest.raises(ValueError, match="Duplicate"):
        run_training(rows * 2, tmp_path, tmp_path / "new")
    with pytest.raises(ValueError, match="already exists"):
        run_training(rows, tmp_path, tmp_path)
    with pytest.raises(ValueError, match="Base model"):
        run_training(rows, tmp_path / "missing", tmp_path / "new")


def test_training_adapter_with_fake_backend(tmp_path, monkeypatch):
    """Exercise split/label/config wiring, not a claim of actual model training."""
    import sys
    from types import SimpleNamespace

    seen = {}

    class Tokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert kwargs["local_files_only"]
            return Tokenizer()

        def __call__(self, texts, **kwargs):
            return {"input_ids": [[1, 2] for _ in texts]}

        def save_pretrained(self, path):
            seen["tokenizer_saved"] = path

    class Model:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert kwargs["local_files_only"] and kwargs["num_labels"] == 2
            return Model()

        def save_pretrained(self, path):
            seen["model_saved"] = path

    class Trainer:
        def __init__(self, **kwargs):
            dataset = kwargs["train_dataset"]
            assert len(dataset) == 16
            assert dataset[0]["labels"][1] == "long"
            assert len(kwargs["eval_dataset"]) == 4

        def train(self):
            seen["train_called"] = True

        def evaluate(self):
            return {"eval_loss": 0.5}

    def split(rows, **kwargs):
        assert kwargs["stratify"] == [int(row["toxic"]) for row in rows]
        return rows[:16], rows[16:]

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            utils=SimpleNamespace(data=SimpleNamespace(Dataset=object)),
            tensor=lambda value, dtype=None: (value, dtype),
            long="long",
            cuda=SimpleNamespace(is_available=lambda: False),
        ),
    )
    monkeypatch.setitem(sys.modules, "sklearn", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "sklearn.model_selection", SimpleNamespace(train_test_split=split)
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=Tokenizer,
            AutoModelForSequenceClassification=Model,
            Trainer=Trainer,
            TrainingArguments=lambda **kwargs: kwargs,
            set_seed=lambda seed: None,
        ),
    )
    rows = [{"comment_text": f"synthetic-{i}", "toxic": i % 2} for i in range(20)]
    assert run_training(rows, tmp_path, tmp_path / "model") == {"eval_loss": 0.5}
    assert seen["train_called"] and seen["model_saved"] == seen["tokenizer_saved"]
    with pytest.raises(ValueError, match="five examples"):
        run_training(rows[:4], tmp_path, tmp_path / "small")


def test_train_cli_and_predict_validation(tmp_path, monkeypatch):
    from toxic_comments import cli, training

    path = tmp_path / "sample.csv"
    path.write_text("comment_text,toxic\nhello,0\n")
    monkeypatch.setattr(training, "run_training", lambda *args: {"eval_loss": 0.5})
    args = [
        "train",
        "--data",
        str(path),
        "--model-dir",
        str(tmp_path / "new"),
        "--base-model",
        str(tmp_path),
    ]
    assert CliRunner().invoke(app, args).exit_code == 0
    assert CliRunner().invoke(app, args + ["--epochs", "0"]).exit_code == 2
    args[4] = str(tmp_path)
    assert CliRunner().invoke(app, args).exit_code == 2
    monkeypatch.delenv("TOXIC_MODEL_DIR", raising=False)
    assert CliRunner().invoke(app, ["predict", "--text", "hi"]).exit_code == 2
    monkeypatch.setattr(cli, "classify", lambda *args: {"label": "non_toxic", "score": 0.8})
    result = CliRunner().invoke(app, ["predict", "--text", "hi", "--model-dir", str(tmp_path)])
    assert json.loads(result.stdout)["label"] == "non_toxic"
