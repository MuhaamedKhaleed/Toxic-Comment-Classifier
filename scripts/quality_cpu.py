"Explicit measured CPU quality experiment. No import-time work or downloads."


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", type=Path, help="Translation only: trusted local Marian")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; use a new directory.")
    import json
    import time
    import resource
    from pathlib import Path
    import pyarrow.parquet as pq
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix

    root = Path(str(args.output))
    root.mkdir(exist_ok=False)
    seen = set()
    bins = {0: [], 1: []}
    for batch in pq.ParquetFile(str(args.data / "civil-validation.parquet")).iter_batches(
        batch_size=1024, columns=["text", "toxicity"]
    ):
        for r in batch.to_pylist():
            text = r["text"].strip()
            label = int(r["toxicity"] >= 0.5)
            if not text or text in seen or len(bins[label]) >= 5000:
                continue
            seen.add(text)
            bins[label].append((text, label))
        if min(map(len, bins.values())) >= 5000:
            break
    rows = bins[0] + bins[1]
    train, hold = train_test_split(
        rows, test_size=0.3, random_state=42, stratify=[r[1] for r in rows]
    )
    dev, test = train_test_split(
        hold, test_size=0.5, random_state=43, stratify=[r[1] for r in hold]
    )
    start = time.time()
    results = []
    best = -1
    for c in [0.5, 2, 8]:
        model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=30000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
                    ),
                ),
                ("classifier", LogisticRegression(C=c, max_iter=300, random_state=42)),
            ]
        )
        model.fit([r[0] for r in train], [r[1] for r in train])
        pred = model.predict([r[0] for r in dev])
        f = f1_score([r[1] for r in dev], pred)
        results.append({"C": c, "development_f1": f})
        if f > best:
            best = f
            chosen = model
            chosen_c = c
    labels = [r[1] for r in test]
    pred = chosen.predict([r[0] for r in test])
    prob = chosen.predict_proba([r[0] for r in test])[:, 1]
    report = {
        "source": "https://www.tensorflow.org/datasets/catalog/civil_comments",
        "dataset": (
            "Official validation repurposed into development-only "
            "train/dev/finalholdout, not officialtest"
        ),
        "selection": (
            "First5000exactunique each toxicity>=.5class, balanced artificial prevalence"
        ),
        "train": len(train),
        "dev": len(dev),
        "test": len(test),
        "dev_trials": results,
        "chosen_C": chosen_c,
        "test_f1": f1_score(labels, pred),
        "test_roc_auc": roc_auc_score(labels, prob),
        "test_accuracy": accuracy_score(labels, pred),
        "test_confusion": confusion_matrix(labels, pred).tolist(),
        "elapsed_seconds": time.time() - start,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limits": (
            "Transparent TFIDFlinearbaseline, not neural DistilBERT "
            "replacement; split from officialvalidation, "
            "artificialbalancedprevalence, no demographic "
            "fairness/realworldmoderation guarantee; finalholdout read "
            "once after devselection"
        ),
    }
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
