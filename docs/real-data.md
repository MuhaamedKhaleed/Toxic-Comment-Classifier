# Real-data CPU development run

Source: https://www.tensorflow.org/datasets/catalog/civil_comments . Actual recorded result: `real-data-output.json`. This is a small development run, not full training, production readiness or an untouched benchmark. No data/model artifacts are committed. Run from the repository root after installing the locked ML/dev extras. This script starts real CPU training only when explicitly invoked. Outputs use fixed new names under the supplied data folder and refuse rerunning into an existing artifact. Use a new folder for another run.

```sh
uv sync --locked --extra dev --extra ml
uv run --no-sync python scripts/real_data.py --data /path/to/prepared-source
```

Download the official Civil Comments validation parquet from https://huggingface.co/datasets/google/civil_comments/tree/main/data (about21MB; source catalog CC0). Prepare256 exact-text unique comments, first128 each side of toxicity>=.5, into `civil-256.json` records comment_text/toxic. Original validation is repurposed as development data, not an independent benchmark. Supply `--base-model /path/to/trusted/local/DistilBERT`. Actual run used the tiny randomly initialized local demo model/vocabulary, not a pretrained semantic model. Real-data execution does not make that vocabulary useful. The actual eval loss is near chance. No offensive text, private labels or weights are republished. A future useful model needs representative pretrained tokenization and independent evaluation.

## Actual limits

real text training-path proof, not useful moderation quality; binary threshold discards label uncertainty; no fairness/accuracy claim

## Explicit primary-source download and preparation

```sh
uv sync --locked --extra dev --extra ml
uv run --with pyarrow==19.0.1 python scripts/fetch_real_data.py --output ./real-source
uv run --no-sync python scripts/demo.py ./tiny-base-demo
uv run --no-sync python scripts/real_data.py --data ./real-source --base-model ./tiny-base-demo/base
```

The explicit helper downloads about21MB from the official Google dataset repository, verifies the exact source SHA-256 and streams batches to select128 unique comments per thresholded label. It never runs at import/install. Its separate preparation dependency is pinned pyarrow19.0.1 through `uv run --with`, not included in the core lock. For offline preparation add `--cached /path/to/files` containing civil-validation.parquet. Output directory must be new. The tiny-base demo creates an untrained synthetic vocabulary and runs its own synthetic wiring check first; it is not a useful moderation model. Substitute a trusted compatible local DistilBERT only with its own rights/provenance checks. Actual teacher-forced training metrics and limitations above still apply.

## Docker without a registry account

This repository supplies a Dockerfile, not a published registry image. Build locally once. The default remains the small invented-input wiring demo. No account, paid service or automatic source download is required by the image.

```sh
docker build -f Dockerfile.ml -t project-ml .
docker run --rm --network none project-ml
```

The image also includes `/app/scripts`. After explicit source preparation on the host, mount a new writable working directory and invoke the real script with `--entrypoint python`. Keep model/source mounts read-only when the script does not write beside them. The image uses uid10001, so grant that uid write access to the output folder, do not run the container as root to bypass this. Data/preprocessing scripts currently place artifacts beside source files; mount the working copy read-write and keep your original untouched. The environment disables Hugging Face downloads. Network-none is suitable only after sources/models are local; an explicit fetch helper needs network by design. CPU/dependency builds were checked on GitHub's Linux runner, not every laptop OS.
