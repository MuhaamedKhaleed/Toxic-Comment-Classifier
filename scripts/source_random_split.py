"""Separate fixed source-level random split, natural source class proportions."""

import argparse
import json
import time
import hashlib
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        average_precision_score,
        confusion_matrix,
        accuracy_score,
    )

    if a.output.exists():
        raise ValueError("Use new output directory.")
    seen = set()
    rows = []
    for r in json.loads(a.source_json.read_text()):
        t = r["text"].strip()
        if not t or t in seen:
            continue
        seen.add(t)
        rows.append((t, int(r["toxicity"] >= 0.5)))
    (tr, hold) = train_test_split(
        rows, test_size=0.3, random_state=20261125, stratify=[r[1] for r in rows]
    )
    (dev, test) = train_test_split(
        hold, test_size=0.5, random_state=20261126, stratify=[r[1] for r in hold]
    )
    start = time.time()
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=30000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
                ),
            ),
            ("classifier", LogisticRegression(C=8, max_iter=300, random_state=42)),
        ]
    )
    model.fit([r[0] for r in tr], [r[1] for r in tr])
    y = [r[1] for r in test]
    pred = model.predict([r[0] for r in test])
    probs = model.predict_proba([r[0] for r in test])[:, 1]
    r = {
        "source": "https://www.tensorflow.org/datasets/catalog/civil_comments",
        "train": len(tr),
        "dev_unused": len(dev),
        "test": len(test),
        "test_positive": sum(y),
        "f1": f1_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "roc_auc": roc_auc_score(y, probs),
        "average_precision": average_precision_score(y, probs),
        "accuracy": accuracy_score(y, pred),
        "confusion": confusion_matrix(y, pred).tolist(),
        "elapsed_seconds": time.time() - start,
        "split_sha256": {
            k: hashlib.sha256("\n".join((r[0] for r in v)).encode()).hexdigest()
            for (k, v) in [("train", tr), ("dev", dev), ("test", test)]
        },
        "limits": [
            "Same officialvalidation source exposed in previous balanced experiment, not independently heldout external data.",
            "Exactduplicate texts removed, no near-duplicate/author/thread separation.",
            "FixedC8and default.5probability threshold before new split. No balancing or threshold tuning.",
            "Dev split reserved but unused, no model selection on this experiment.",
            "Source prevalence, not deployment prevalence.",
            "Prior balanced/remainder results use different training samples, not controlled performance improvement.",
            "No demographic fairness or calibrated risk claim.",
        ],
    }
    a.output.mkdir(parents=True)
    (a.output / "report.json").write_text(json.dumps(r, indent=2, allow_nan=False))
    print(json.dumps(r))


if __name__ == "__main__":
    main()
