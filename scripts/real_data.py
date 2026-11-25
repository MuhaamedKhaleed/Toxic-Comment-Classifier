"""Explicit real-data CPU development recipe. Never runs on import."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Prepared local source directory")
    parser.add_argument("--base-model", help="Trusted compatible local DistilBERT directory")
    parser.add_argument("--manifest", help="Reviewed selected Open Images source manifest")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs (default: the small recorded run)")
    args = parser.parse_args()
    import os
    import pathlib
    import json
    import time

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from toxic_comments.training import run_training

    root = pathlib.Path(args.data)
    rows = json.loads((root / "civil-256.json").read_text())
    start = time.time()
    result = run_training(
        rows, pathlib.Path(args.base_model), root / "toxic-real-model", epochs=args.epochs, batch_size=16
    )
    report = {
        "real_data": True,
        "dataset": "Civil Comments",
        "source": "https://www.tensorflow.org/datasets/catalog/civil_comments",
        "license": "CC0 source dataset",
        "subset": "balanced128+128 from official validation partition repurposed as development; original train/test not used; toxicity>=.5",
        "base": "tiny random DistilBERT/local synthetic vocabulary from engineering demo, not pretrained language ability",
        "training": result,
        "elapsed_seconds": round(time.time() - start, 2),
        "limits": "real text training-path proof, not useful moderation quality; binary threshold discards label uncertainty; no fairness/accuracy claim",
    }
    (root / "toxic-real-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
