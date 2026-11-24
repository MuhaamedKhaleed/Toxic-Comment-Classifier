"""Thin JSON CLI. Training is always an explicit subcommand."""

import json
import logging
import zipfile
from pathlib import Path
from typing import Any

import typer

from .config import local_model_path
from .core import normalize_prediction, read_dataset, validate_rows

app = typer.Typer(
    no_args_is_help=True, help="Check local toxicity data or use trusted local models."
)
logger = logging.getLogger(__name__)


def classify(text: str, model_dir: Path, *, factory: Any = None) -> dict[str, str | float]:
    """Run a named-label local classifier without fetching model artifacts."""
    if not text.strip():
        raise ValueError("Comment must not be empty.")
    if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
        raise ValueError(
            "Model directory needs config.json and compatible local model/tokenizer files."
        )
    if factory is None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ValueError("Install the ML extra with pip install -e '.[ml]'.") from exc
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    else:
        classifier = factory("text-classification", model=str(model_dir), local_files_only=True)
    return normalize_prediction(classifier(text, truncation=True, max_length=64))


@app.command("check-data")
def check_data(
    data: Path = typer.Option(..., help="Local UTF-8 CSV or ZIP containing train.csv."),
) -> None:
    """Validate text and binary labels, with no ML dependency or network call."""
    try:
        typer.echo(json.dumps(validate_rows(read_dataset(data))))
    except (OSError, UnicodeError, ValueError, zipfile.BadZipFile) as exc:
        logger.error("%s", exc)
        raise typer.Exit(2) from exc


@app.command()
def predict(
    text: str = typer.Option(..., help="Comment to classify."),
    model_dir: Path | None = typer.Option(None, help="Trusted local named-label classifier."),
    env_file: Path | None = typer.Option(None, help="Optional explicit dotenv file."),
) -> None:
    """Classify using existing local artifacts. Never train implicitly."""
    try:
        typer.echo(json.dumps(classify(text, local_model_path(model_dir, env_file))))
    except (OSError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        raise typer.Exit(2) from exc


@app.command()
def train(
    data: Path = typer.Option(..., help="Local labeled UTF-8 CSV or ZIP."),
    model_dir: Path = typer.Option(..., help="New output directory, must not already exist."),
    base_model: Path = typer.Option(
        ..., help="Existing trusted local DistilBERT checkpoint/tokenizer."
    ),
    epochs: int = typer.Option(2, min=1),
    batch_size: int = typer.Option(32, min=1),
    sample_frac: float = typer.Option(1.0, min=0.001, max=1.0),
) -> None:
    """Explicit local training. May be slow; no training is run by tests/help."""
    try:
        rows = read_dataset(data)
        if model_dir.exists():
            raise ValueError("Output directory already exists. Choose a new --model-dir.")
        if not base_model.is_dir():
            raise ValueError(
                "Provide an existing local --base-model directory; no downloads are started."
            )
        from .training import run_training

        results = run_training(rows, base_model, model_dir, epochs, batch_size, sample_frac)
        typer.echo(json.dumps(results))
    except ImportError as exc:
        logger.error("Install the ML extra with pip install -e '.[ml]'.")
        raise typer.Exit(2) from exc
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        logger.error("%s", exc)
        raise typer.Exit(2) from exc


def main() -> None:
    """Configure logs once, keeping JSON on stdout."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    app()
