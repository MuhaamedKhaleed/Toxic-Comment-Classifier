# Explicit synthetic demo

This is a reproducible wiring example, not a real-data benchmark. It creates invented inputs and a new output directory, calls the actual package, and writes `report.json` with `synthetic_only: true` and `quality_claim: false`. Existing output directories are refused. Numeric values may differ across CPU/library versions; output schema and safety checks, not exact loss values, are the contract.

```sh
uv sync --locked --extra ml
uv run --no-sync python scripts/demo.py /tmp/my-new-demo
```

The saved `docs/demo-output.json` is an actual local run, not an expected-quality target. Model files remain local and are not committed. No pretrained model download or service call occurs during the demo.

The demo trains a tiny randomly initialized model on invented inputs, saves and reloads it, and calls its inference path. Tiny architectures and one epoch/two GAN steps are deliberate CPU smoke limits, not full original-model training.

## Opt-in CPU execution container

```sh
docker build -f Dockerfile.ml -t project-synthetic-ml .
docker run --rm --network none project-synthetic-ml
```

The default `Dockerfile` is unchanged and lightweight. This opt-in image installs the committed lock and runtime ML extra, no pytest/dev tools. It is larger and slower to build. PyTorch projects use an explicit CPU-wheel index in the lock. Runtime is restricted to CPU execution. Build needs network for dependencies, runtime smoke uses `--network none`. No host data, weights, credentials, cloud endpoint or paid compute is required. The container prints the report path, runs assertions and exits nonzero on a failed backend check. Files live only in the disposable container unless you explicitly mount an empty writable output directory.

Hosted CI must build and run the opt-in container before container verification can be claimed. Local Python success alone is not container success.
