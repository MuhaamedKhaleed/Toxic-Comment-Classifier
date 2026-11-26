"""Freeze prior balanced recipe and evaluate a disjoint unbalanced source slice."""

import json
import time
import hashlib
import random
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        f1_score,
        roc_auc_score,
        accuracy_score,
        confusion_matrix,
        precision_score,
        recall_score,
        average_precision_score,
        brier_score_loss,
    )

    if args.output.exists():
        raise ValueError("Output exists. Choose a new directory.")
    source = args.source_json
    rows = json.loads(source.read_text())
    seen = set()
    bins = {0: [], 1: []}
    for r in rows:
        t = r["text"].strip()
        label = int(r["toxicity"] >= 0.5)
        if not t or t in seen or len(bins[label]) >= 5000:
            continue
        seen.add(t)
        bins[label].append((t, label))
        if min(map(len, bins.values())) >= 5000:
            break
    selected = bins[0] + bins[1]
    (train, hold) = train_test_split(
        selected, test_size=0.3, random_state=42, stratify=[r[1] for r in selected]
    )
    (dev, test) = train_test_split(
        hold, test_size=0.5, random_state=43, stratify=[r[1] for r in hold]
    )
    new = []
    used = set(seen)
    for r in rows:
        t = r["text"].strip()
        if not t or t in used:
            continue
        used.add(t)
        new.append((t, int(r["toxicity"] >= 0.5)))
    new = random.Random(20261125).sample(new, 10000)
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
    model.fit([r[0] for r in train], [r[1] for r in train])

    def evaluate(samples):
        labels = [r[1] for r in samples]
        pred = model.predict([r[0] for r in samples])
        prob = model.predict_proba([r[0] for r in samples])[:, 1]
        return {
            "n": len(samples),
            "positive": sum(labels),
            "prevalence": sum(labels) / len(labels),
            "f1": f1_score(labels, pred),
            "precision": precision_score(labels, pred),
            "recall": recall_score(labels, pred),
            "accuracy": accuracy_score(labels, pred),
            "roc_auc": roc_auc_score(labels, prob) if len(set(labels)) == 2 else None,
            "average_precision": average_precision_score(labels, prob),
            "brier": brier_score_loss(labels, prob),
            "confusion": confusion_matrix(labels, pred).tolist(),
            "predicted_positive": int(sum(pred)),
        }

    r = {
        "source": "https://www.tensorflow.org/datasets/catalog/civil_comments",
        "recipe": "Prior balanced10000unique source rows and7000/1500/1500split reconstructed exactly, fixedC8and default.5probability threshold, no new tuning.",
        "full_source_rows": len(rows),
        "full_source_positive": sum((r["toxicity"] >= 0.5 for r in rows)),
        "balanced_check": evaluate(test),
        "unbalanced_slice": evaluate(new),
        "training_text_sha256": hashlib.sha256(
            "\n".join((r[0] for r in train)).encode()
        ).hexdigest(),
        "new_slice_text_sha256": hashlib.sha256(
            "\n".join((r[0] for r in new)).encode()
        ).hexdigest(),
        "elapsed_seconds": time.time() - start,
        "limits": [
            "Official validation partition repurposed as research data, not officialtest.",
            "Fixedseed random10000remaining unique texts excluding all prior selected10000, but prior balanced selection alters remaining prevalence.",
            "Source prevalence not deployment prevalence.",
            "Thresholdtoxicity>=.5discards uncertainty.",
            "No demographic subgroups or fairness claim.",
            "No calibration or model changes from this evaluation.",
        ],
    }
    out = args.output
    out.mkdir(parents=True)
    (out / "report.json").write_text(json.dumps(r, indent=2, allow_nan=False))
    print(json.dumps(r))


if __name__ == "__main__":
    main()
