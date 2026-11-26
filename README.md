# Toxic Comment Classifier

A local binary-toxicity classifier with an offline data validator and explicit training/inference commands. This version fixes unsafe archive validation, implicit training on import, and a scalar regression head that did not match the binary task. The working model route is a fine-tuned DistilBERT.

## Results (honest)

- Fresh-clone DistilBERT fine-tune on public Civil Comments data (25 Nov 2026, one CPU core, ~25 min): **F1 0.8012, precision 0.7135, recall 0.9133, AUROC 0.8725** on 300 balanced development comments.
- A separate Kaggle CPU run (Python 3.13, ~15 min): F1 0.7754, precision 0.8492, recall 0.7133, AUROC 0.8822 on the same 300-row slice. Different unrecorded library versions, so treat it as a separate observation, not a confidence range.
- Both are **development-slice numbers on a small balanced subset** - not a held-out benchmark, not natural prevalence, and not a Jigsaw six-label evaluation (binary `toxic` only).
- 25 tests, 94% statement coverage, including a real optimizer epoch on a tiny DistilBERT with changed weights and safetensors reload. Coverage is not toxicity accuracy.

## Quickstart (fresh clone, about 2 minutes)

Needs Python 3.11 or 3.12 (locked wheels have no 3.13+ build), internet for the first install/downloads, ~2 GB free RAM.

```sh
uv sync --python 3.11 --locked --extra dev --extra ml
uv run --no-sync python scripts/fetch_encoder.py --output ./encoder                      # pinned DistilBERT, Apache-2.0
uv run --with pyarrow==19.0.1 python scripts/fetch_real_data.py --output ./real-source   # ~21 MB Civil Comments validation
uv run --with pyarrow==19.0.1 python scripts/distilbert_pilot.py --data ./real-source --encoder ./encoder --output ./toxic-out --per-class 200 --epochs 2
uv run --no-sync toxic-comments predict --text "have a lovely day" --model-dir ./toxic-out/best-model
```

A small 60-per-class, 1-epoch run finishes in ~25 s (dev F1 0.71, weak). More `--per-class`/`--epochs` is the main knob. Offline data checks need no ML stack: `toxic-comments check-data --data YOUR.csv` (also reads a ZIP containing exactly one `train.csv`, rejecting unsafe paths).

## Limits

- No stratified held-out evaluation, subgroup/fairness analysis, calibration or latency measurement. A returned score is not calibrated.
- Training uses your supplied local artifacts only; nothing is downloaded implicitly, and old one-output regression checkpoints are incompatible.
- On free Kaggle/Colab notebooks the locked environment does not install (no `torch==2.4.1+cpu` wheel there); use the preinstalled Python/torch/transformers route documented in `docs/distilbert-pilot.md`.
- Verified on Linux x86_64; the container is a data-check image without the ML stack. Free hosted CI runs tests, lint, CLI and the Docker build.
- No moderation policy, deployment, release or dataset/model redistribution.

## License

MIT for the new project-owned code (see `LICENSE` and `LICENSE-SCOPE.md`). Third-party models, datasets and historical code keep their own terms; rights review is still required before redistribution. Details and evidence: `docs/verification.md`, `docs/distilbert-pilot.md`, `docs/real-data.md`, `CHANGELOG.md`.
