# Official PROCESS-2 TEST Evaluation Record

## Status

**OFFICIAL TEST STATUS: CONSUMED FOR FINAL EVALUATION.**

The predefined PROCESS-2 TEST partition was evaluated once on 2026-10-10 after the deployment configuration was frozen. This is a research-prototype, preliminary speech-based screening result; it is not a diagnosis, clinical validation result, or medical probability.

The authoritative evaluation record is the finalized, internally consistent persisted bundle in `artifacts/results/official_test_evaluation/`. Inspect that bundle directly. Do **not** rerun `python .\scripts\evaluate_official_test.py`: its consumed-evaluation guard must refuse a subsequent run.

## Frozen configuration evaluated

- ASR: faster-whisper `base.en`, CPU, `int8`, English (`en`), 4 CPU threads, 1 worker.
- Representation: `ASR_RATE_PLUS_CTD`, with 51 ordered predictors: 32 ASR linguistic, 6 ASR-word-count-derived speech-rate, and 13 CTD waveform-only acoustic predictors.
- Pipeline: `SimpleImputer(strategy="median")`, `StandardScaler`, then Logistic Regression with `C=0.1`, `max_iter=2000`, and `random_state=42`.
- Score: raw Logistic Regression impaired-class classifier score; display score is raw score × 100.
- Operating rule: the frozen, TRAIN-derived research threshold is `0.53`. Scores at least `0.53` map to Possible impairment-like speech pattern; lower scores map to Healthy-like speech pattern.
- Calibration: raw score; no additional sigmoid calibration.

The pipeline, schema, metadata, and frozen ASR configuration hashes were validated before the evaluation. The TEST partition contributed no model fit, preprocessing fit, feature selection, threshold selection, calibration, classifier selection, or ASR-model selection.

## Authoritative persisted result

The persisted bundle contains 80 participant predictions from 240 recordings: 40 Healthy and 40 Impaired participants (30 MCI and 10 Dementia within the impaired class).

| Metric | Value |
|---|---:|
| Accuracy | 0.8000 |
| Balanced accuracy | 0.8000 |
| Precision | 0.8158 |
| Sensitivity | 0.7750 |
| Specificity | 0.8250 |
| F1 | 0.7949 |
| ROC-AUC | 0.8250 |
| Brier score | 0.1739 |
| Log loss | 0.5532 |

At the frozen research threshold of `0.53`, the confusion matrix is:

| True class | Predicted Healthy | Predicted Impaired |
|---|---:|---:|
| Healthy | 33 | 7 |
| Impaired | 9 | 31 |

Equivalently: TN 33, FP 7, FN 9, TP 31.

## Descriptive 95% bootstrap confidence intervals

The persisted bootstrap file uses 2,000 participant-level resamples with `random_state=42`. It estimates uncertainty around the fixed predictions only; it does not retrain, tune, calibrate, or select a threshold.

| Metric | 95% percentile CI |
|---|---:|
| Accuracy | 0.7125–0.8875 |
| Balanced accuracy | 0.7084–0.8860 |
| Sensitivity | 0.6410–0.8974 |
| Specificity | 0.6977–0.9388 |
| F1 | 0.6875–0.8842 |
| ROC-AUC | 0.7260–0.9110 |

## Original-diagnosis descriptive breakdown

This is a descriptive breakdown of the frozen binary classifier, not a new three-class model and not evidence for diagnosis-specific thresholds.

| Original diagnosis | Participants | Healthy-like | Possible impairment-like |
|---|---:|---:|---:|
| HC | 40 | 33 | 7 |
| MCI | 30 | 7 | 23 |
| Dementia | 10 | 2 | 8 |

## Concurrent-execution anomaly and record precedence

During the first authorized held-out evaluation, two evaluator processes were inadvertently active concurrently before atomic single-run protection was introduced. Both used the same frozen model, feature schema, preprocessing, ASR configuration, and threshold. No development decision changed between them.

The duplicate process reported F1 = 0.800 and ROC-AUC = 0.848. It is retained as an execution anomaly for transparency; it is neither another independent validation cohort nor an alternative result to select. The results were not averaged, and the more favorable duplicate output was not chosen.

The finalized, internally consistent persisted bundle was designated the official evaluation record by a procedural artifact-precedence rule: it is the complete, auditable on-disk bundle whose predictions, metrics, confidence intervals, diagnosis breakdown, metadata, and plots agree. This choice is not performance-based.

An atomic evaluation lock was subsequently added before any TEST work can begin. It remains after interruption or completion, and the evaluator also refuses any run once `evaluation_metadata.json` exists. This prevents concurrent or repeat TEST processing. An authorized developer may inspect the saved CSV, JSON, and plots in the persisted bundle, but must not rerun the evaluator or reprocess TEST data.

## Limitations and next stage

The held-out sample has 80 participants, including only 10 with original Dementia labels; the confidence intervals and subgroup counts should therefore be interpreted cautiously. The result applies to this predefined research corpus and frozen pipeline only. It does not establish clinical validity, diagnosis, medical risk, or causal biomarkers.

The TEST partition is consumed and cannot support a future unbiased re-evaluation of an altered model. Any future model-development work must remain TRAIN-only or, preferably, use a new external validation cohort.

The next project stage is production inference / FastAPI integration using the already-frozen artifacts. It must not use the consumed TEST results to redesign the model.
