# Toxicity classifier: prevalence sensitivity check

25 November2026. Evaluation-only warning, no model/threshold tuning.

The existing TF-IDF logistic baseline was developed on an artificially balanced sample. We reconstructed that exact recipe: first5,000unique text rows per toxicity>=.5class,7,000train/1,500dev/1,500test using the original seeds, C8chosen previously. The balanced holdout reproduces F1.770919exactly, confirming the comparison uses the same recipe.

Then sampled10,000remaining exact-unique texts with seed20261125, excluding all10,000previously selected texts. There are304positive labels,3.04%. Results:

|Metric|Balanced prior holdout|Unbalanced remainder|
|---|---:|---:|
|F1|.7709|.1779|
|Precision|.7938|.1013|
|Recall|.7493|.7303|
|ROC-AUC|.8582|.8393|
|Average precision|.8581|.2962|
|Accuracy|.7773|.7948|

On the remainder:7,726true negatives,1,970false positives,82false negatives,222true positives. The majority-only negative rule would have96.96%accuracy there but zero positive recall. Accuracy is therefore misleading by itself. Model probabilities are not established calibrated risk scores.

## Important selection limits

This is NOT natural population prevalence. The full source file has97,320rows and7,671positive labels before exact-text filtering. Earlier balanced selection removed many positives. The remainder consequently has altered prevalence. The source is Civil Comments officialvalidation, repurposed for development research, not officialtest. Source order is structured: an initial first10,000remaining-row diagnostic had zero positives, so it cannot support an unbalanced prevalence comparison. That preliminary diagnostic was abandoned for that purpose, and the fixed-seed random remainder check is reported instead. Neither permits production generalization.

No threshold sweep, calibration fit, model choice or tuning used these new answers. A later experiment should freeze a fresh source-disjoint test before training and validate precision/recall under the actual intended prevalence and cost of false accusations. Do not automate punitive decisions from this classifier. Labels are annotator toxicity judgments averaged into a threshold, not facts about a person's intent. No subgroup fairness evaluation.

## Reproduce

Requires the existing scikit-learn/NumPy environment and authorized local Civil Comments officialvalidation parquet. Export only text and toxicity rows to a private JSON list, preserving source order, then:

    python scripts/evaluate.py --source-json YOUR_PRIVATE_ROWS.json --output NEW_DIRECTORY

No dataset text or weights are bundled. Script is an explicit executable experiment, import-safe until its explicit main entrypoint. It rejects existing output directories, performs no network call, and saves metrics only. Portable invocation reproduced all previous result fields exactly except runtime. SHA256of training and remainder concatenated text is in report.json for audit, not a substitute for image/text-source permission.

Source: https://www.tensorflow.org/datasets/catalog/civil_comments . Source catalog lists CC0. This evaluation does not change model/threshold or portfolio score. Existing balanced score must retain its artificial-prevalence caveat. New code MIT, third-party data rights separate. No paid work, main merge or release.

## Separate source-proportion pilot

To avoid treating the depleted remainder as natural data, ran a separate fixed stratified random split over all96,846exact-unique nonempty source texts:67,792train,14,527reserved-unused development,14,527test. FixedC8and.5threshold, no balancing and no tuning.1,145test positives,7.88%source prevalence. F1.5006, precision.7611, recall.3729, ROC-AUC.9101, average precision.6180, accuracy.9414. Confusion13,248TN/134FP/718FN/427TP.

This differs in training size, class distribution and split, so it is NOT a controlled improvement over the balanced model, and F1values cannot be ranked without those contexts. The source was already exposed in prior experiments. This is a fresh exact-text-disjoint partition within an exposed source, not an untouched external benchmark. No author/thread/near-duplicate separation. The result makes the tradeoff concrete: fewer false positives, many missed positive labels. No automatic moderation readiness follows.

Reproduce `scripts/source_random_split.py` with the same --source-json and new --output arguments. Full metrics and split hashes are in source-proportion-report.json. Development split remains unused, but the test is now exposed and must not be reused as an untouched test after further tuning.


Numerical result JSON and any generated samples are supplied in the private experiment ZIP, not committed here. Run the commands from the repository root with authorized local inputs. These scripts are separate research recipes, not changes to the production model or alert policy.
