# Verification and phase record

Python 3.11.18, CPU. No paid APIs, training, pretrained-weight download or deployed service.

## Observed local checks

- `uv sync --extra dev`: editable package installed and lock resolved.
- `uv sync --locked --extra dev`: locked install passed. The lock targets Linux x86_64, matching verification and CI; other platforms are unverified.
- `python -m toxic_comments --help`, `toxic-comments --help`, `toxic-comments train --help`: passed.
- `toxic-comments check-data --data /tmp/toxic-example.csv`: returned rows=2, toxic=1, non_toxic=1 for synthetic input documented in README.
- `toxic-comments predict --text hello`: exit 2 with missing model-path guidance.
- `toxic-comments check-data --data /missing.csv`: exit 2 with missing-data guidance.
- `python -m pytest --cov=toxic_comments --cov-report=term-missing`: 24 pass, 91% package statement coverage, offline synthetic/fake backend tests.
- `python -m ruff check src tests`, `python -m ruff format --check src tests`, `python -m mypy src`: passed.
- `python -m pre_commit run --all-files`: passed with the active environment on PATH.
- `command -v docker`: no executable. Local Docker build/run not claimed. Hosted result must be checked separately.

Real backend checks used Torch 2.4.1+cpu and Transformers 4.49.0 with offline environment flags. A tiny random DistilBERT (one layer, hidden size 16, vocabulary 32) produced shape (2, 2) and finite single-label cross-entropy loss for integer labels. A second temporary random tiny model/tokenizer was saved locally and loaded through `classify`, returning a named label and finite score. These are forward-only wiring checks. No optimizer/training or performance evaluation occurred. Fake backend tests verify dataset lengths, integer labels, stratified split arguments, config and save calls, not actual convergence.

## Original bugs and deliberate behavior changes

The old archive check accepted a sibling-prefix traversal member in a harmless synthetic reproduction. New code reads a validated train.csv member without extraction and rejects traversal. The original `num_labels=1` plus floating labels selects regression under Transformers, not binary classification. New code requests two classes, integer labels and explicit single-label loss. Source inspected: https://raw.githubusercontent.com/huggingface/transformers/v4.49.0/src/transformers/models/distilbert/modeling_distilbert.py .

Training formerly started implicitly when no model existed. Missing data could log an error and exit zero. Defaults now show help or fail explicitly; training has its own command. Global pipeline state is replaced by explicit local paths. The old main.py importing API is not preserved, but `python main.py --help` remains an entry point after install. The package has no guessed legacy-label conversion. CLI training defaults to all rows; 10% sampling must be explicit. Duplicate comments are rejected, not silently deduplicated or relabeled. Nested ZIP datasets now need user-controlled unpacking. These are safety/correctness changes, not silent refactor equivalence.

## Thirteen-phase status

1. Package: editable install and entry points verified.
2. Cleanup: offline validation/config/CLI and optional ML training separated; typed public helpers.
3. Errors/logging: once at entry point, JSON output separate; tested failure exits.
4. Config: explicit dotenv and local-model precedence, no required secret.
5. Dependencies: runtime/ML/dev extras, uv lock, locked install.
6. Tests: 24 offline cases, 91% software coverage, no accuracy inference.
7. Quality: Ruff/format/mypy/pre-commit pass.
8. CI: free public-runner-only workflow, no cache/artifacts to avoid fee scope creep; final hosted result needs live check.
9. Docker: non-root CLI/data-check image prepared, no local Docker; hosted verification tracked separately.
10. Interface: explicit check-data/predict/train commands. Actual trained-model execution unverified.
11. README: tested commands, architecture, synthetic output and limits; no invented performance or hosted demo.
12. Git: local bug-fix commit separate from package refactor; project-specific feature branch; no main merge, tag or release.
13. Presentation: proposed text in presentation.md, no profile/settings changes.

## Outstanding

Verify real local training with licensed data and trusted base artifacts. Evaluate held-out classification metrics, class balance, calibration and subgroup failures. No general quality/fairness/latency claims. Warmup/training defaults need empirical review. Exact-text duplicate rejection is not an author/semantic leakage audit. Existing trained regression artifacts are incompatible. ML deployment needs a separate image/runtime check; current Docker scope is CLI only. Licensing and PR words need review; no automatic main merge.


## Gap-closure follow-up (24 November 2026)

The CI source now uses uv 0.13.8 and the committed transitive lock, rather than a new pip resolution of direct pins. Dependency installer caching is keyed by uv.lock. The local matching command is `uv sync --locked --extra dev --extra ml`, followed by the existing tests/quality commands. This does not cache or publish private datasets, weights or model outputs. Public-repository-only guards remain. The changed workflow's hosted outcome must be checked after publication; prior green runs do not validate a changed workflow.

A complete copy-only project-focused profile README draft is appended to presentation.md. No profile account edits, main merge, PR, release or license change is implied by that text. Historical notebook/full-data/quality limitations remain open, not replaced by a passing synthetic test.


Actual-backend follow-up: a locally initialized tiny DistilBERT and local DistilBERT tokenizer performed a real optimizer epoch on 20 invented unique comments. Tests confirmed changed network weights, finite validation loss, safetensors save, local reload, and prediction through classify. Full suite: 25 passed, 94% package statement coverage. This closes the earlier no-optimizer-execution gap, not real toxicity quality. No pretrained weights or external dataset were used.
