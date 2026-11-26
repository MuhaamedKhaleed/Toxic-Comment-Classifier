# Measured no-cost CPU quality experiment

Transparent alternative TFIDF-logistic baseline, not DistilBERT.10k exactunique comments selected from officialvalidation at balanced prevalence,7k/1.5k/1.5k split. Three C values selected by devF1, finalF1.7709/AUROC.8582/accuracy.7773. No officialtest/fairness/realworldprevalence claim.3.3sec259MiB. Extra pinned preparation dependency: pyarrow19.0.1; sklearn1.6.1 is locked MLdependency.

Run only explicitly, with previously prepared local permitted data and a new output. No automatic downloads or import-time execution. Exact measured result is quality-output.json. This is a reproducible development experiment, not a productionquality guarantee.

```sh
uv sync --locked --extra dev --extra ml
uv run --with pyarrow==19.0.1 python scripts/quality_cpu.py --data LOCAL_PREPARED_SOURCE --output NEW_OUTPUT
```

Test sets described as finalized have now been observed. Future choices must not be justified with their scores as if they remained unseen. Retain source terms/attribution from real-data.md. No model or raw text/photos committed.
