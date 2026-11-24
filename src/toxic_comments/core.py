"""Offline validation and safe archive handling for binary toxicity data."""

import csv
import io
import math
import zipfile
from pathlib import Path
from typing import Any

MODEL_NAME = "distilbert-base-uncased"
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024


def validate_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Check required text/binary labels without silently coercing bad data."""
    if not rows:
        raise ValueError("Dataset is empty. Supply rows with comment_text and toxic columns.")
    counts = {"rows": len(rows), "toxic": 0, "non_toxic": 0}
    for index, row in enumerate(rows, start=1):
        if not {"comment_text", "toxic"}.issubset(row):
            raise ValueError("Dataset needs comment_text and toxic columns.")
        text = row["comment_text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Row {index}: comment_text must be non-empty text.")
        label = row["toxic"]
        if isinstance(label, bool) or str(label) not in {"0", "1"}:
            raise ValueError(f"Row {index}: toxic must be 0 or 1.")
        counts["toxic" if str(label) == "1" else "non_toxic"] += 1
    return counts


def read_dataset(path: Path) -> list[dict[str, Any]]:
    """Read UTF-8 CSV or one train.csv member, without extracting files."""
    if not path.is_file():
        raise ValueError("Dataset not found. Supply --data pointing to a local CSV or ZIP.")
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            for member in members:
                name = member.filename.replace("\\", "/")
                if name.startswith("/") or ".." in name.split("/") or ":" in name:
                    raise ValueError("Archive contains an unsafe path. Supply a clean CSV.")
                if member.file_size > MAX_ARCHIVE_BYTES:
                    raise ValueError("Archive member is too large. Supply a smaller local CSV.")
            candidates = [m for m in members if Path(m.filename).name == "train.csv"]
            if len(candidates) != 1:
                raise ValueError("ZIP must contain exactly one train.csv, not nested ZIPs.")
            with archive.open(candidates[0]) as source:
                text = source.read(MAX_ARCHIVE_BYTES + 1)
            if len(text) > MAX_ARCHIVE_BYTES:
                raise ValueError("Dataset exceeds the archive read limit.")
            rows = list(csv.DictReader(io.StringIO(text.decode("utf-8-sig"))))
    else:
        with path.open(encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
    validate_rows(rows)
    return rows


def normalize_prediction(result: Any) -> dict[str, str | float]:
    """Validate the model's named label and finite score, not generic LABEL_0."""
    if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], dict):
        raise ValueError("Model returned an invalid result. Check compatible local artifacts.")
    prediction = result[0]
    label = prediction.get("label")
    score = prediction.get("score")
    if label not in {"toxic", "non_toxic"}:
        raise ValueError(
            "Model must map labels to toxic/non_toxic. Old regression models are incompatible."
        )
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("Model score must be numeric.")
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Model score must be finite and between 0 and 1.")
    return {"label": label, "score": float(score)}
