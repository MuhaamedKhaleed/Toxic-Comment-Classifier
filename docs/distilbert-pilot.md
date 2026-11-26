# Bounded pretrained DistilBERT fine-tune (original neural path)

25 November 2026. Development-slice signal from an exposed source, not a
benchmark and not moderation readiness.

The original project fine-tunes pretrained DistilBERT, but every recorded run
so far used a tiny randomly initialized model or a TF-IDF alternative. This
pilot performs the first real fine-tune of the original neural path with
pretrained weights on real Civil Comments text, bounded to free local
resources (one CPU core, 2 GB RAM): the pinned `distilbert-base-uncased`
encoder (revision 12040accade4e8a0f71eabdb258fecc2e7e948be, Apache 2.0, same
provenance and SHA-256 set as the NER project's contextual-provenance record)
with embeddings and the lower four transformer layers frozen; the top two
layers, pooler and classification head train.

## Actual measured results

Data: 96,805 exact-text-unique rows scanned from the pinned official
validation parquet (SHA-256 in the JSON). Seeded Python selection RNG
(20261125): a shuffled pool of 1500 rows per class, first 1500 combined kept
(near-balanced by construction; actual fit counts 587 toxic / 613
non-toxic), development carve-out exactly 150 per class. 1200 fit rows, 300
development rows. Only the data-selection RNG was seeded; training itself
was not deterministically seeded.

Training: AdamW lr 2e-5, batch 8, max length 96, 2 epochs, 14,767,874 of
66,955,010 parameters trainable. One external interruption: the run resumed
from the saved epoch-1 checkpoint with a fresh AdamW, so it is an
interrupted continuation, not an exact optimizer-state resume (the report
records `interrupted_resumed: true`).

Development metrics of the persisted epoch-1 checkpoint at threshold 0.5,
scored with the intended two-logit softmax and sklearn standard
implementations:

| Metric | Value |
|---|---:|
| F1 | .8012 |
| Precision | .7135 |
| Recall | .9133 |
| Accuracy | .7733 |
| ROC-AUC (sklearn) | .8725 |
| Average precision (sklearn) | .8302 |
| Confusion | TP 137 / FP 55 / FN 13 / TN 95 |

Peak RSS 1.77 GB on a 2 GB box; wall time roughly 25 minutes across the
interrupted attempt and the resume leg.

Conclusion: the original neural path with pretrained weights does learn from
real data under free local constraints, where the previous tiny randomly
initialized model stayed near chance. This is a feasibility signal with an
internal development slice, nothing more; see limits below. Exact numbers,
hashes and configuration: `distilbert-pilot.json`.

## Scoring correction (kept for audit)

The first version of this pilot scored the two-logit cross-entropy
classifier with `sigmoid(logit1)` and custom ROC/Average-Precision code
whose curve omitted the origin and handled ties non-standardly. Epoch
selection ran under that wrong scorer (epoch 1: F1 .8012; epoch 2: F1
.7762), so the persisted checkpoint was chosen under the wrong scorer and
the epoch-2 weights were not retained: re-selection under the correct
scorer is impossible, and the corrected metrics above evaluate the epoch-1
checkpoint as-is without implying it is the best checkpoint under the
correct rule.

Softmax probability >= 0.5 tests `logit1 >= logit0` while the legacy
`sigmoid(logit1) >= 0.5` tests `logit1 >= 0`; the two rules disagree
whenever `sign(logit1) != sign(logit1 - logit0)`. Checked empirically from
the persisted Nx2 logits: 0 of the 300 development rows meet that
condition, so the two scorers necessarily produce identical predictions,
confusion and threshold-based metrics on this checkpoint and slice (an
observed property, not a general equivalence); only the ranking metrics
changed (ROC-AUC .8763 -> .8725, average precision .8432 -> .8302).
Per-row logits and both scorers' predictions are preserved for audit. The post-hoc
rescore reproduced every legacy number exactly from the persisted
checkpoint and seeded selection before applying the correct scorer,
confirming the data slice and weights are unchanged. Legacy values are
kept in the JSON under `superseded_legacy_metrics`, explicitly marked
do-not-cite. The training script now scores with softmax and sklearn, with
tests covering ties, ranking disagreement and threshold disagreement.

## Limits

- Development slice drawn from the official validation parquet, which
  earlier experiments already used; this is not an untouched test and not
  natural prevalence (development balanced by construction).
- Internal fit/development carve-out only. Metrics are a feasibility signal
  for the original path, not production quality.
- Checkpoint selected under the wrong scorer (see above); epoch-2
  comparison under the correct scorer is not recoverable.
- Cannot be ranked against the TF-IDF baseline: data selection, size and
  split all differ. No threshold tuning, calibration or subgroup fairness
  evaluation was done. Probabilities are not calibrated risk.
- Labels are averaged annotator judgments, not facts about intent. Do not
  automate punitive moderation decisions from this model.
- Partial fine-tune under resource limits; full fine-tuning of all layers
  remains untested. No text, labels or weights are committed.
- Only the data-selection RNG was seeded; retraining produces different
  weights.

## Reproduce

Python 3.11, Torch 2.4.1+cpu, Transformers 4.49.0, scikit-learn 1.6.1,
pyarrow 19.0.1. Local pinned encoder directory and local parquet only (see
docs/real-data.md for the pinned download helper); no download at import or
test time; new output directory required; interrupted runs resume with
`--resume` from saved best weights (fresh optimizer, see above). The corrected script rejects legacy scorer states and changed run configurations.

```sh
uv sync --locked --extra dev --extra ml
uv run --with pyarrow==19.0.1 python scripts/distilbert_pilot.py \
  --data LOCAL_CIVIL_SOURCE --encoder LOCAL_PINNED_ENCODER --output NEW_DIRECTORY
python -m pytest tests/test_distilbert_pilot.py
```
