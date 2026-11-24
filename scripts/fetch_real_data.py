"""Explicit source download. No import-time network or data redistribution."""

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New source directory")
    parser.add_argument("--cached", type=Path, help="Use already downloaded files, without network")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = []
    for filename, url, sha in SOURCES:
        target = args.output / filename
        if args.cached:
            shutil.copyfile(args.cached / filename, target)
        else:
            with (
                urllib.request.urlopen(url, timeout=120) as source,
                target.open("wb") as out,
            ):
                shutil.copyfileobj(source, out)
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != sha:
            target.unlink()
            raise ValueError("Source checksum mismatch: " + filename)
        manifest.append({"file": filename, "url": url, "sha256": actual})
    (args.output / "source-manifest.json").write_text(json.dumps(manifest, indent=2))
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError(
            "Use uv run --with pyarrow==19.0.1 python scripts/fetch_real_data.py"
        ) from exc
    seen, bins = set(), {0: [], 1: []}
    # Stream record batches rather than materializing the entire validation corpus.
    parquet = pq.ParquetFile(args.output / "civil-validation.parquet")
    for batch in parquet.iter_batches(batch_size=1024, columns=["text", "toxicity"]):
        for row in batch.to_pylist():
            text = row["text"].strip()
            label = int(row["toxicity"] >= 0.5)
            if not text or text in seen or len(bins[label]) >= 128:
                continue
            seen.add(text)
            bins[label].append({"comment_text": text, "toxic": label})
            if min(map(len, bins.values())) == 128:
                break
        if min(map(len, bins.values())) == 128:
            break
    if min(map(len, bins.values())) != 128:
        raise ValueError("Source did not contain the expected balanced slice")
    (args.output / "civil-256.json").write_text(json.dumps(bins[0] + bins[1]))


SOURCES = [
    (
        "civil-validation.parquet",
        "https://huggingface.co/datasets/google/civil_comments/resolve/main/data/validation-00000-of-00001.parquet",
        "2e0eb65474e7e1290df8689fc93eea783160c898c09ae15382c0a9662367bd03",
    )
]

if __name__ == "__main__":
    main()
