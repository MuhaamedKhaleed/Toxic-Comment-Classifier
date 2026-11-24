# Toxic Comment Classifier

A local binary-toxicity experiment with an offline data validator and explicit training/inference commands. This engineering pilot fixes unsafe archive validation, implicit training and a scalar regression head that did not match the binary classification task.

Status: package and offline checks verified. No Jigsaw training, trained-checkpoint evaluation, accuracy, latency or production moderation claim is made. A random tiny DistilBERT forward pass checks the two-class loss wiring only.

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
