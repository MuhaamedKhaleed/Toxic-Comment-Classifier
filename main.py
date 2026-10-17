"""Toxic Comment Classifier fine-tuning and inference script."""

import argparse
import logging
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
    set_seed as hf_set_seed,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Default Configuration Constants
MODEL_NAME = "distilbert-base-uncased"
MODEL_DIR = "./toxic_comment_model"
OUTPUT_DIR = "./results"
LOGGING_DIR = "./logs"
ZIP_FILE_PATH = "jigsaw-toxic-comment-classification-challenge.zip"
EXTRACT_DIR = "jigsaw-toxic-comment-classification-challenge"
RANDOM_SEED = 42
SAMPLE_FRAC = 0.1
MAX_LENGTH = 64
BATCH_SIZE = 32
EPOCHS = 2

# Module-level aliases preserved for backward compatibility
model_name = MODEL_NAME
model_path = MODEL_DIR
zip_file_path = ZIP_FILE_PATH
extract_dir = EXTRACT_DIR
classifier = None


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seed across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


def safe_extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract zip archive safely, preventing path traversal attacks."""
    extract_to.mkdir(parents=True, exist_ok=True)
    resolved_target = extract_to.resolve()
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.namelist():
            member_path = (extract_to / member).resolve()
            if not str(member_path).startswith(str(resolved_target)):
                raise ValueError(f"Unsafe path detected in zip archive: {member}")
        zip_ref.extractall(extract_to)


def load_dataset(
    zip_path_str: Union[str, Path] = ZIP_FILE_PATH,
    extract_dir_str: Union[str, Path] = EXTRACT_DIR,
) -> pd.DataFrame:
    """Locate, extract if necessary, and load the training dataset into a DataFrame."""
    zip_path = Path(zip_path_str)
    target_dir = Path(extract_dir_str)

    # Check for extracted files first
    candidates = [
        target_dir / "train.csv.zip",
        target_dir / "train.csv",
        Path("train.csv.zip"),
        Path("train.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            logger.info("Loading dataset from %s", candidate)
            return pd.read_csv(candidate)

    # If not found directly, extract from zip archive if available
    if zip_path.exists():
        logger.info("Extracting %s to %s", zip_path, target_dir)
        safe_extract_zip(zip_path, target_dir)
        for candidate in [target_dir / "train.csv.zip", target_dir / "train.csv"]:
            if candidate.exists():
                logger.info("Files unzipped successfully! Loading %s", candidate)
                return pd.read_csv(candidate)

    raise FileNotFoundError(
        f"Dataset not found at '{zip_path}' or in '{target_dir}'. "
        "Please download the Jigsaw toxic comment dataset archive."
    )


class ToxicCommentDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for tokenized comments and binary toxicity labels."""

    def __init__(self, encodings: dict, labels: list):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Convert labels to float for sequence classification loss
        item["labels"] = torch.tensor(self.labels[idx]).float()
        return item

    def __len__(self) -> int:
        return len(self.labels)


def train_model(
    df: pd.DataFrame,
    model_name: str = MODEL_NAME,
    model_save_path: str = MODEL_DIR,
    output_dir: str = OUTPUT_DIR,
    logging_dir: str = LOGGING_DIR,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    sample_frac: float = SAMPLE_FRAC,
    max_length: int = MAX_LENGTH,
    seed: int = RANDOM_SEED,
) -> Tuple[Trainer, Dict[str, float]]:
    """Prepare data, fine-tune the sequence classification model, and save artifacts."""
    set_seed(seed)

    if "comment_text" not in df.columns or "toxic" not in df.columns:
        raise ValueError(
            f"Dataset missing required columns ('comment_text', 'toxic'). Found: {list(df.columns)}"
        )

    # Sample a smaller dataset for training speed
    df_sample = df.sample(frac=sample_frac, random_state=seed).copy()
    df_sample["toxic"] = df_sample["toxic"].astype(int)
    df_sample["comment_text"] = df_sample["comment_text"].fillna("").astype(str)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df_sample["comment_text"],
        df_sample["toxic"],
        test_size=0.2,
        random_state=seed,
    )

    logger.info(
        "Loaded %d training samples and %d validation samples.",
        len(train_texts),
        len(test_texts),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_encodings = tokenizer(
        list(train_texts), truncation=True, padding=True, max_length=max_length
    )
    test_encodings = tokenizer(
        list(test_texts), truncation=True, padding=True, max_length=max_length
    )

    train_dataset = ToxicCommentDataset(train_encodings, train_labels.tolist())
    test_dataset = ToxicCommentDataset(test_encodings, test_labels.tolist())

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=logging_dir,
        fp16=torch.cuda.is_available(),  # Enable mixed precision only when CUDA is available
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    logger.info("Starting model training...")
    trainer.train()

    logger.info("Evaluating model performance...")
    eval_results = trainer.evaluate()
    logger.info("Evaluation results: %s", eval_results)

    logger.info("Saving fine-tuned model and tokenizer to %s...", model_save_path)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    return trainer, eval_results


def classify_comment(comment: str) -> dict:
    """Classify a comment string as toxic or non-toxic using the fine-tuned model."""
    global classifier
    if not isinstance(comment, str):
        raise TypeError(f"Expected str for comment, got {type(comment).__name__}")
    if classifier is None:
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model directory '{model_path}' does not exist. "
                "Train the model first or provide a valid model directory."
            )
        classifier = pipeline("text-classification", model=model_path, tokenizer=model_path)
    result = classifier(comment)[0]
    return result


def parse_args() -> argparse.Namespace:
    """Parse CLI options for toxic comment classification."""
    parser = argparse.ArgumentParser(
        description="Train and infer with Toxic Comment Classifier."
    )
    parser.add_argument(
        "--data-zip",
        default=ZIP_FILE_PATH,
        help="Path to the Jigsaw toxic comment zip archive.",
    )
    parser.add_argument(
        "--extract-dir",
        default=EXTRACT_DIR,
        help="Directory to extract dataset archives into.",
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help="Hugging Face pre-trained transformer model identifier.",
    )
    parser.add_argument(
        "--model-dir",
        default=MODEL_DIR,
        help="Output directory for saved fine-tuned model artifacts.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Per-device training and evaluation batch size.",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=SAMPLE_FRAC,
        help="Fraction of dataset to sample for training.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=MAX_LENGTH,
        help="Maximum token length for tokenizer.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Force running the training pipeline.",
    )
    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        help="Optional comment string to classify.",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution entry point."""
    global model_path, classifier
    args = parse_args()
    set_seed(RANDOM_SEED)
    model_path = args.model_dir

    # If prediction was explicitly requested without training
    if args.predict and not args.train:
        result = classify_comment(args.predict)
        print(f"Comment: {args.predict}")
        print(f"Label: {result['label']}")
        print(f"Score: {result['score']}")
        return

    # Determine whether training should run
    should_train = args.train
    df: Optional[pd.DataFrame] = None
    if not should_train and not Path(args.model_dir).exists():
        should_train = True

    if should_train:
        try:
            df = load_dataset(args.data_zip, args.extract_dir)
        except FileNotFoundError as err:
            if args.train or not Path(args.model_dir).exists():
                logger.error("%s", err)
                return

        if df is not None:
            train_model(
                df=df,
                model_name=args.model_name,
                model_save_path=args.model_dir,
                output_dir=OUTPUT_DIR,
                logging_dir=LOGGING_DIR,
                epochs=args.epochs,
                batch_size=args.batch_size,
                sample_frac=args.sample_frac,
                max_length=args.max_length,
            )
            # Re-initialize classifier to point to the newly saved model
            classifier = pipeline(
                "text-classification",
                model=args.model_dir,
                tokenizer=args.model_dir,
            )

    # Inference demonstration
    if Path(args.model_dir).exists():
        comment_text = args.predict if args.predict is not None else "i hate u"
        result = classify_comment(comment_text)
        print(f"Comment: {comment_text}")
        print(f"Label: {result['label']}")
        print(f"Score: {result['score']}")
    else:
        logger.warning(
            "Model directory '%s' does not exist. Train the model first by providing the dataset.",
            args.model_dir,
        )


if __name__ == "__main__":
    main()
