# Real-data CPU development run

Source: https://www.tensorflow.org/datasets/catalog/civil_comments . Actual recorded result: `real-data-output.json`. This is a small development run, not full training, production readiness or an untouched benchmark. No data/model artifacts are committed. Run from the repository root after installing the locked ML/dev extras. This script starts real CPU training only when explicitly invoked. Outputs use fixed new names under the supplied data folder and refuse rerunning into an existing artifact. Use a new folder for another run.

```sh
uv sync --locked --extra dev --extra ml
uv run --no-sync python scripts/real_data.py --data /path/to/prepared-source
```

Download the official Civil Comments validation parquet from https://huggingface.co/datasets/google/civil_comments/tree/main/data (about21MB; source catalog CC0). Prepare256 exact-text unique comments, first128 each side of toxicity>=.5, into `civil-256.json` records comment_text/toxic. Original validation is repurposed as development data, not an independent benchmark. Supply `--base-model /path/to/trusted/local/DistilBERT`. Actual run used the tiny randomly initialized local demo model/vocabulary, not a pretrained semantic model. Real-data execution does not make that vocabulary useful. The actual eval loss is near chance. No offensive text, private labels or weights are republished. A future useful model needs representative pretrained tokenization and independent evaluation.

## Actual limits

real text training-path proof, not useful moderation quality; binary threshold discards label uncertainty; no fairness/accuracy claim
