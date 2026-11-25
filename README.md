# Toxic Comment Classifier

A local binary-toxicity experiment with an offline data validator and explicit training/inference commands. This engineering pilot fixes unsafe archive validation, implicit training and a scalar regression head that did not match the binary classification task.

Status: package and offline checks verified. No Jigsaw training, trained-checkpoint evaluation, accuracy, latency or production moderation claim is made. A random tiny DistilBERT forward pass checks the two-class loss wiring only.

## Real-data quickstart (about 2 minutes)

Needs Python 3.11 or 3.12 (the locked PyTorch/TensorFlow wheels have no 3.13+ build); `--python 3.11` makes uv fetch it. Needs internet for the first install and downloads, and roughly 2 GB of free RAM.

```sh
uv sync --python 3.11 --locked --extra dev --extra ml
uv run --no-sync python scripts/fetch_encoder.py --output ./encoder                      # pinned DistilBERT, Apache-2.0
uv run --with pyarrow==19.0.1 python scripts/fetch_real_data.py --output ./real-source   # ~21 MB Civil Comments validation
uv run --with pyarrow==19.0.1 python scripts/distilbert_pilot.py --data ./real-source --encoder ./encoder --output ./toxic-out --per-class 200 --epochs 2
uv run --no-sync toxic-comments predict --text "have a lovely day" --model-dir ./toxic-out/best-model
```
Output: fine-tuned `toxic-out/best-model` plus `report.json` (F1, precision, recall), then a JSON label and score. Tweak `--per-class`, `--epochs`, `--batch-size`, `--max-length`. A 60-per-class, 1-epoch run on 25 Nov 2026 took about 25 s and reached dev F1 0.71 (tiny and weak, mostly predicts `toxic`); more data and epochs are the knob. `scripts/real_data.py` is only a wiring check with a random tiny model and is not the useful route.

## Install and check

Python 3.11+:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m toxic_comments --help
toxic-comments --help
python -m pytest --cov=toxic_comments --cov-report=term-missing
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src
python -m pre_commit run --all-files
```

Alternatively, with uv installed, `uv sync --locked --extra dev` uses the committed Linux x86_64 lockfile. This lock targets the verified runner/CPU environment; other platforms can use pip but are unverified. Runtime dependencies are Typer, Click and python-dotenv; optional ML dependencies are Torch, Transformers, scikit-learn and Accelerate. No secret is required. Dependency installation requires internet; the runtime commands do not fetch models.

## Offline data example

Create a UTF-8 CSV with these synthetic rows, not a performance dataset:

```csv
comment_text,toxic
hello,0
angry,1
```

```sh
toxic-comments check-data --data /tmp/toxic-example.csv
```

Observed JSON:

```json
{"rows": 2, "toxic": 1, "non_toxic": 1}
```

`check-data` also reads a ZIP containing exactly one `train.csv` without extracting anything. Unsafe paths, ambiguous files, empty comments, nonbinary labels and oversized members are rejected. Nested `train.csv.zip` archives must be unpacked separately by the user. This intentionally tightens the old automatic extraction behavior.

## Local model commands

Help and missing-input errors were checked. The following actual ML operations are documented interfaces, **not verified end-to-end recipes** in this pilot:

```text
toxic-comments predict --text COMMENT --model-dir LOCAL_MODEL
toxic-comments train --data LOCAL_CSV --base-model LOCAL_DISTILBERT --model-dir NEW_OUTPUT
```

Use `python -m pip install -e '.[ml]'` for the optional ML backend. Training must be explicitly requested. It uses supplied local artifacts only, two named classes (`non_toxic`, `toxic`), integer labels and cross-entropy loss. It rejects duplicate comments and inadequate sampled class counts before splitting. Defaults preserve two epochs, batch size 32 and max length 64; the CLI defaults to using all supplied rows rather than silently sampling 10%. No pretrained model download, training or trained-checkpoint inference has been run here. A temporary tiny random model/tokenizer was saved and loaded through the actual local inference adapter to verify artifact wiring, not prediction quality. Old one-output regression checkpoints are incompatible, not repaired by changing the label text.

`predict` accepts `TOXIC_MODEL_DIR` from the process or an explicitly supplied `--env-file`; `--model-dir` wins. See `.env.example`. Only load trusted local model/tokenizer files. Generic `LABEL_0`/`LABEL_1` results are rejected rather than guessed. A returned score is not proof of calibration.

## Architecture and meaningful checks

```text
Typer commands -> path/config/data validation -> JSON output
                           |
                           +-> optional local ML adapter
                                train: stratified split -> two-class head
                                predict: named-label result validation
```

- `core.py`: offline CSV/ZIP validation and result validation.
- `config.py`: explicit dotenv/local-model path, no key collection.
- `training.py`: optional local-only training adapter.
- `cli.py`: thin commands and clear nonzero errors.
- `tests/`: synthetic data, hostile-path regression, fake model/backend wiring, CLI and config failures. Tests require no network or trained weights.

Observed: 24 tests pass, 91% package statement coverage. This is software coverage, not toxicity accuracy. Coverage includes fake training-backend wiring; it does not establish that a real model was trained. Ruff, format, mypy and pre-commit pass. See `docs/verification.md` for commands and limits.

## Container and CI

The multi-stage non-root image is a lightweight data-check/CLI image. It excludes the optional ML stack, datasets and weights. Local Docker is unavailable. Hosted build/smoke status must be checked in Actions before claiming success. CI uses only a free standard public-repository runner, no cache, uploaded artifact or image publication, and skips its job if the repository is private. No hosted inference service was deployed.

## Limits and next work

The source script advertised end-to-end classification but configured scalar regression, did not report classification metrics, and could start training implicitly. Those behaviors are corrected in this pilot. The API is intentionally explicit rather than preserving unsafe defaults or global cached model state.

Before making quality claims: obtain a licensed dataset and compatible local artifacts, verify actual training and trained-checkpoint inference, evaluate stratified held-out data separated from tuning, report precision/recall/F1 and class balance, assess subgroup errors and calibration, and review truncation beyond 64 tokens. Exact-duplicate rejection is not a semantic or author-level leakage audit. A binary `toxic` target does not implement all six Jigsaw labels. No automatic moderation policy or fairness guarantee follows.

No repository/model/data license was added. Rights and model/data terms require separate review. Main remains unchanged; this is a review branch, not a release.


Actual-backend follow-up: a locally initialized tiny DistilBERT and local DistilBERT tokenizer performed a real optimizer epoch on 20 invented unique comments. Tests confirmed changed network weights, finite validation loss, safetensors save, local reload, and prediction through classify. Full suite: 25 passed, 94% package statement coverage. This closes the earlier no-optimizer-execution gap, not real toxicity quality. No pretrained weights or external dataset were used.


## License scope

New project-owned package/engineering files use the owner's MIT default. See LICENSE and LICENSE-SCOPE.md. This is not permission to relicense historical notebooks, third-party code, datasets, models or weights. Their provenance and terms still need review. Earlier verification entries saying no license was added describe the initial pilot, not this follow-up.

## Checked synthetic wiring demo

See [commands, saved output and scope](docs/synthetic-demo.md). This is invented-input execution evidence, not model accuracy or a real-data gallery.

## Real-data CPU example

See [actual results, source terms and explicit download/preparation commands](docs/real-data.md). These are small development runs, not accuracy or production guarantees. No model/data is committed.

## Measured CPU quality follow-up

See [quality experiments, results and limits](docs/quality-cpu.md). These go beyond wiring smokes; some improve measured results, while others expose overfit or a stronger simple baseline. No quality guarantee.

## Fresh-clone path to a pretrained DistilBERT fine-tune

The quickstart in `docs/real-data.md` uses a tiny random base to prove wiring only (loss stays at 0.693). The original neural path with real pretrained weights is `scripts/distilbert_pilot.py`. Download the pinned, hash-checked `distilbert-base-uncased` encoder (Apache-2.0, revision 12040acca...) and the Civil Comments validation parquet, then run it:

```sh
uv sync --locked --extra dev --extra ml
uv run --with pyarrow==19.0.1 python scripts/fetch_real_data.py --output ./real-source
uv run --no-sync python scripts/fetch_encoder.py --output ./distilbert-encoder
uv run --with pyarrow==19.0.1 python scripts/distilbert_pilot.py --data ./real-source --encoder ./distilbert-encoder --output ./pilot-out
```

Recorded result (25 Nov 2026, one CPU core): F1 0.8012, precision 0.7135, recall 0.9133, AUROC 0.8725 on 300 development comments from a balanced subset, about 25 minutes, peak memory 1.77 GB. This is not a held-out benchmark or natural-prevalence estimate. Needs about 2 GB RAM, so use a 2 GB+ machine or a free Kaggle/Colab notebook. See `docs/distilbert-pilot.md`.

### Free Kaggle notebook route (verified 25 Nov 2026)

On a Kaggle CPU notebook with internet on, `uv sync --locked` failed ("torch==2.4.1+cpu has no wheel for the current platform"), so the locked environment does not install there. Use the notebook's preinstalled Python, torch, transformers and pyarrow instead:

```sh
git clone -b toxic-engineering-pilot --depth 1 https://github.com/MuhaamedKhaleed/Toxic-Comment-Classifier
cd Toxic-Comment-Classifier
export PYTHONPATH=$PWD/src
python scripts/fetch_real_data.py --output ./real-source
python scripts/fetch_encoder.py --output ./distilbert-encoder
python scripts/distilbert_pilot.py --data ./real-source --encoder ./distilbert-encoder --output ./pilot-out
```

One run on a Kaggle CPU session (Python 3.13, 4 CPU, 31 GB RAM) finished in about 15 minutes: F1 0.7754, precision 0.8492, recall 0.7133, accuracy 0.7933, AUROC 0.8822 on the same 300-row balanced development slice, 1,200 fit rows, 2 epochs. The exact torch and transformers versions of that run were not recorded, so it is a separate observation from the 0.8012 run above, not a confidence range. Training order and initialization are not seeded. The fine-tuned weights stay inside the notebook output; nothing is published.
