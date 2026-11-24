"""Optional explicit local DistilBERT training, with no download fallback."""

import logging
from pathlib import Path
from typing import Any

from .core import validate_rows

logger = logging.getLogger(__name__)


def training_config() -> dict[str, Any]:
    """Named two-class head, using cross entropy rather than scalar regression."""
    return {
        "num_labels": 2,
        "problem_type": "single_label_classification",
        "id2label": {0: "non_toxic", 1: "toxic"},
        "label2id": {"non_toxic": 0, "toxic": 1},
    }


def run_training(
    rows: list[dict[str, Any]],
    base_model: Path,
    model_dir: Path,
    epochs: int = 2,
    batch_size: int = 32,
    sample_frac: float = 1.0,
) -> dict[str, float]:
    """Fit only when explicitly requested, using supplied local artifacts.

    Reports validation loss, not a fabricated accuracy metric. The split is
    stratified. Duplicate texts are rejected to avoid exact-text split leakage.
    """
    validate_rows(rows)
    if epochs < 1 or batch_size < 1 or not 0 < sample_frac <= 1:
        raise ValueError("Use positive epochs/batch size and a sample fraction in (0, 1].")
    texts = [row["comment_text"].strip() for row in rows]
    if len(set(texts)) != len(texts):
        raise ValueError(
            "Duplicate comments found. Deduplicate and resolve conflicting labels first."
        )
    if model_dir.exists():
        raise ValueError("Output directory already exists. Choose a new directory.")
    if not base_model.is_dir():
        raise ValueError("Base model must be an existing trusted local directory.")
    import random

    import torch
    from sklearn.model_selection import train_test_split
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(42)
    selected = random.Random(42).sample(rows, max(1, int(len(rows) * sample_frac)))
    labels = [int(row["toxic"]) for row in selected]
    if min(labels.count(0), labels.count(1)) < 5:
        raise ValueError(
            "The sampled data needs at least five examples of each class for the split."
        )
    train_rows, validation_rows = train_test_split(
        selected,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(base_model), local_files_only=True)

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, records: list[dict[str, Any]]) -> None:
            self.labels = [int(row["toxic"]) for row in records]
            self.encodings = tokenizer(
                [row["comment_text"] for row in records],
                truncation=True,
                padding=True,
                max_length=64,
            )

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int) -> dict[str, Any]:
            item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
            return item

    model = AutoModelForSequenceClassification.from_pretrained(
        str(base_model),
        local_files_only=True,
        **training_config(),
    )
    args = TrainingArguments(
        output_dir=str(model_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        warmup_steps=500,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        report_to="none",
        save_strategy="no",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=Dataset(train_rows),
        eval_dataset=Dataset(validation_rows),
    )
    logger.info("Training on %d rows, validation on %d rows", len(train_rows), len(validation_rows))
    trainer.train()
    results = trainer.evaluate()
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    return {key: float(value) for key, value in results.items() if isinstance(value, (int, float))}
