"""Bounded original-path pretrained DistilBERT fine-tune on real comments.

The original project fine-tunes pretrained DistilBERT, but every recorded run
so far used either a tiny randomly initialized model or a TF-IDF alternative.
This script performs the first real pretrained-weight fine-tune of the
original neural path on Civil Comments official validation text, bounded to
free local CPU/RAM: embeddings and the lower transformer layers stay frozen,
only the top layers and the classification head train. It answers one
question: does the original neural path actually learn from real data under
no-cost constraints, and what quality signal does a small development slice
show? It does NOT produce an untouched benchmark, a calibrated risk score, or
moderation readiness, and its numbers cannot be ranked against the linear
baseline because data selection, size and split all differ.

Explicit local artifacts only: pinned local encoder directory (revision and
SHA-256 recorded in docs/contextual-provenance.json of the NER project; keep
its Apache 2.0 license) and the pinned local parquet from
scripts/fetch_real_data.py. No download, no import-time work, no overwrite.

Crash safety: the new output directory is created before training and each
completed epoch saves its development metrics and, when improved, the best
model. Rerun with --resume into that directory to continue or finish an
interrupted pilot without repeating completed epochs. This loads best weights
and resets AdamW, not the exact last optimizer state. Legacy scorer states and
changed run settings are rejected.
"""


def dev_metrics_from_logits(truth, logits):
    """Score a two-logit single-label classifier.

    Toxic probability is the softmax over both logits; the prediction is
    threshold >= 0.5 (a tied pair is assigned to the toxic class). ROC-AUC and average
    precision come from sklearn's standard implementations, not custom
    rank code.
    """
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    logits = np.asarray(logits, dtype=float)
    truth = np.asarray(truth)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    prob = exp[:, 1] / exp.sum(axis=1)
    pred = (prob >= 0.5).astype(int)
    tp = int(((pred == 1) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    tn = int(((pred == 0) & (truth == 0)).sum())
    return {
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "roc_auc": float(roc_auc_score(truth, prob)),
        "average_precision": float(average_precision_score(truth, prob)),
    }


def validate_resume_state(state, config):
    """Do not mix a legacy scorer or a different run into this continuation."""
    if state.get("scorer_version") != "two-logit-softmax-v1":
        raise ValueError("Legacy or unknown scoring state; use a new output directory.")
    if state.get("config") != config:
        raise ValueError("Resume configuration differs from the saved run.")


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--encoder", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-class", type=int, default=750)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--freeze-layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted pilot in its existing output directory.",
    )
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        parser.error("Output exists; use a new directory, or --resume it.")
    if min(args.per_class, args.epochs, args.batch_size, args.max_length) < 1:
        parser.error("Numeric options must be positive.")
    if not 0 <= args.freeze_layers <= 5:
        parser.error("Freeze layers must be within [0, 5].")
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["USE_TF"] = "0"
    os.environ["USE_JAX"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    import hashlib
    import json
    import random
    import resource
    import time

    start = time.time()
    parquet = args.data / "civil-validation.parquet"
    raw = parquet.read_bytes()
    config = {
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "encoder_path": str(args.encoder.resolve()),
        "per_class": args.per_class,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "freeze_layers": args.freeze_layers,
        "seed": args.seed,
    }
    state_path = args.output / "state.json"
    state = {"completed_epochs": 0, "history": []}
    if args.resume:
        if not state_path.is_file():
            raise ValueError("Resume requires a completed-epoch state file.")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        validate_resume_state(state, config)
    import pyarrow.parquet as pq

    seen, rows = set(), []
    source = pq.ParquetFile(parquet)
    for batch in source.iter_batches(batch_size=4096, columns=["text", "toxicity"]):
        for row in batch.to_pylist():
            text = " ".join(row["text"].split())
            if text and text not in seen:
                seen.add(text)
                rows.append((text, int(row["toxicity"] >= 0.5)))
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    by_class = {0: [], 1: []}
    for text, label in rows:
        if len(by_class[label]) < 2 * args.per_class:
            by_class[label].append((text, label))
    if min(len(v) for v in by_class.values()) < 2 * args.per_class:
        raise ValueError("Source lacks enough unique examples per class.")
    selected = by_class[0][: 2 * args.per_class] + by_class[1][: 2 * args.per_class]
    rng.shuffle(selected)
    train_rows = selected[: 2 * args.per_class]
    # Internal stratified carve-out: 80% fine-tune, 20% development signal.
    halves = {0: [], 1: []}
    for record in train_rows:
        halves[record[1]].append(record)
    dev_rows = halves[0][: args.per_class // 5] + halves[1][: args.per_class // 5]
    fit_rows = halves[0][args.per_class // 5 :] + halves[1][args.per_class // 5 :]
    rng.shuffle(fit_rows)
    rng.shuffle(dev_rows)

    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.encoder), local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.encoder),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        num_labels=2,
        problem_type="single_label_classification",
        id2label={0: "non_toxic", 1: "toxic"},
        label2id={"non_toxic": 0, "toxic": 1},
    )
    base = model.distilbert
    for parameter in base.embeddings.parameters():
        parameter.requires_grad = False
    for layer in base.transformer.layer[: args.freeze_layers]:
        for parameter in layer.parameters():
            parameter.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    def encode(records):
        encoded = tokenizer(
            [r[0] for r in records],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
            return_tensors="pt",
        )
        return encoded, torch.tensor([r[1] for r in records], dtype=torch.long)

    train_enc, train_y = encode(fit_rows)
    dev_enc, dev_y = encode(dev_rows)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5)
    model.train()
    args.output.mkdir(parents=True, exist_ok=True)
    history = state["history"]
    best_f1 = max([r["dev_f1"] for r in history], default=-1.0)
    if state["completed_epochs"] > 0:
        if not (args.output / "best-model").is_dir():
            raise ValueError("Resume needs the saved best model from epoch 1.")
        resumed = AutoModelForSequenceClassification.from_pretrained(
            args.output / "best-model",
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        )
        model.load_state_dict(resumed.state_dict())
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5)

    def dev_logits():
        model.eval()
        chunks = []
        with torch.inference_mode():
            for i in range(0, len(dev_rows), 32):
                part = {k: v[i : i + 32] for k, v in dev_enc.items()}
                chunks.append(model(**part).logits.numpy())
        model.train()
        return np.concatenate(chunks)

    for epoch in range(state["completed_epochs"] + 1, args.epochs + 1):
        order = torch.randperm(len(fit_rows))
        for i in range(0, len(order), args.batch_size):
            idx = order[i : i + args.batch_size]
            batch = {k: v[idx] for k, v in train_enc.items()}
            out = model(**batch, labels=train_y[idx])
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        logits = dev_logits()
        metrics = dev_metrics_from_logits(dev_y.numpy(), logits)
        row = {
            "epoch": epoch,
            "dev_f1": metrics["f1"],
            "dev_precision": metrics["precision"],
            "dev_recall": metrics["recall"],
        }
        history.append(row)
        print(row, flush=True)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            model.save_pretrained(args.output / "best-model")
            tokenizer.save_pretrained(
                args.output / "best-model"
            )  # needed by `toxic-comments predict`
        state = {
            "scorer_version": "two-logit-softmax-v1",
            "config": config,
            "completed_epochs": epoch,
            "history": history,
        }
        state_path.write_text(json.dumps(state), encoding="utf-8")
    if not (args.output / "best-model").is_dir():
        raise ValueError("No completed epoch produced a saved model state.")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.output / "best-model",
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    model.eval()
    logits = dev_logits()
    metrics = dev_metrics_from_logits(dev_y.numpy(), logits)
    report = {
        "interrupted_resumed": bool(args.resume),
        "resume_behavior": (
            "Resume reloads the best saved checkpoint and creates a fresh "
            "AdamW optimizer, so a continued run is an interrupted "
            "continuation, not an exact optimizer-state resume."
        ),
        "source": "https://www.tensorflow.org/datasets/catalog/civil_comments",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "encoder": {
            "repo": "distilbert/distilbert-base-uncased",
            "revision": "12040accade4e8a0f71eabdb258fecc2e7e948be",
            "license": "Apache-2.0",
        },
        "model": (
            "Original neural path: pretrained DistilBERT with two-class head, "
            f"embeddings and lower {args.freeze_layers} transformer layers "
            "frozen under free local resource limits; top layers and head "
            "fine-tuned."
        ),
        "parameters": {"total": total, "trainable": trainable},
        "data": {
            "unique_source_rows_scanned": len(rows),
            "fit_rows": len(fit_rows),
            "fit_class_counts": {
                "toxic": sum(1 for _, label in fit_rows if label == 1),
                "non_toxic": sum(1 for _, label in fit_rows if label == 0),
            },
            "dev_rows": len(dev_rows),
            "dev_class_counts": {
                "toxic": sum(1 for _, label in dev_rows if label == 1),
                "non_toxic": sum(1 for _, label in dev_rows if label == 0),
            },
            "selection": (
                f"Exact-text deduplicated, seeded Python RNG ({args.seed}). "
                f"Pool of {2 * args.per_class} per class shuffled, first "
                f"{2 * args.per_class} combined kept (near-balanced, actual "
                "counts recorded); development carve-out is exactly "
                f"{args.per_class // 5} per class, remainder fits. Only the "
                "data-selection RNG is seeded; training order and "
                "initialization are not deterministically seeded."
            ),
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "optimizer": "AdamW lr 2e-5",
            "history": history,
        },
        "dev_metrics_threshold_0.5": metrics,
        "elapsed_seconds": time.time() - start,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limits": (
            "Development slice carved from an already fully exposed source; "
            "not an untouched benchmark and not natural prevalence (balanced "
            "by construction). Partial fine-tune under 2 GB free RAM, not "
            "full fine-tuning. Not comparable in rank to the TF-IDF baseline: "
            "selection, size and split differ. Probabilities are not "
            "calibrated risk. Labels are averaged annotator judgments, no "
            "subgroup fairness evaluation. Do not automate punitive "
            "moderation decisions from this model."
        ),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["dev_metrics_threshold_0.5"]))


if __name__ == "__main__":
    main()
