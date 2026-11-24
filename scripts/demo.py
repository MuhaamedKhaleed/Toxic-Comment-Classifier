"""Explicit synthetic wiring demo. No real-data accuracy or quality claim."""

import json
import os
from pathlib import Path
import sys


def main() -> None:
    for key, value in {
        "CUDA_VISIBLE_DEVICES": "-1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }.items():
        os.environ.setdefault(key, value)

    out = Path(sys.argv[1] if len(sys.argv) == 2 else "synthetic-run")
    if len(sys.argv) > 2:
        raise SystemExit("Usage: python scripts/demo.py [NEW_OUTPUT_DIRECTORY]")
    out.mkdir(parents=True, exist_ok=False)
    import math
    import torch
    from transformers import (
        DistilBertTokenizerFast,
        DistilBertConfig,
        DistilBertForSequenceClassification,
    )
    from toxic_comments.cli import classify
    from toxic_comments.training import run_training, training_config

    torch.set_num_threads(1)
    torch.manual_seed(42)
    base = out / "base"
    base.mkdir()
    vocab = base / "vocab.txt"
    vocab.write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nkind\nrude\ncomment\n"
        + "\n".join(str(i) for i in range(20))
        + "\n"
    )
    tokenizer = DistilBertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)
    tokenizer.save_pretrained(base)
    model = DistilBertForSequenceClassification(
        DistilBertConfig(
            vocab_size=len(tokenizer),
            n_layers=1,
            dim=16,
            hidden_dim=32,
            n_heads=2,
            max_position_embeddings=128,
            dropout=0.0,
            attention_dropout=0.0,
            **training_config(),
        )
    )
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.save_pretrained(base, safe_serialization=True)
    rows = [
        {"comment_text": f"{'kind' if i % 2 == 0 else 'rude'} comment {i}", "toxic": i % 2}
        for i in range(20)
    ]
    scores = run_training(rows, base, out / "model", epochs=1, batch_size=4)
    trained = DistilBertForSequenceClassification.from_pretrained(
        out / "model", local_files_only=True
    )
    changed = any(not torch.equal(before[k], v) for k, v in trained.state_dict().items())
    prediction = classify("hello kind comment", out / "model")
    assert changed and math.isfinite(scores["eval_loss"]) and 0 <= prediction["score"] <= 1
    result = {"training": scores, "weights_changed": changed, "prediction": prediction}
    scope = "invented_comments_random_DistilBERT_not_toxicity_quality"
    report = {"synthetic_only": True, "scope": scope, "quality_claim": False, "result": result}
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({"synthetic_only": True, "report": str(out / "report.json")}, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
