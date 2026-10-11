# Faculty Quick Reference

## Project

- **Title:** Speech-Based Cognitive Decline Screening
- **Status:** Research and educational screening prototype; not a medical diagnosis.
- **Dataset:** PROCESS-2
- **Participants:** 400
- **TRAIN:** 320 participants
- **TEST:** 80 participants; consumed once for final held-out evaluation
- **Diagnosis composition:** HC 200, MCI 150, Dementia 50
- **Binary target:** Healthy = HC; Impaired = MCI + Dementia
- **Speech tasks:** SFT semantic fluency, PFT phonemic fluency, CTD picture description

## Frozen deployment pipeline

- **ASR:** faster-whisper `base.en`, CPU/int8, English, four CPU threads, one worker
- **Predictors:** 51 total
  - 32 ASR-compatible linguistic
  - 6 ASR-derived speech-rate
  - 13 CTD waveform acoustic
- **Preprocessing:** TRAIN-fitted median imputer and StandardScaler
- **Classifier:** Logistic Regression
- **C:** `0.1`
- **Research operating threshold:** `0.53`
- **Below 0.53:** Healthy-like speech pattern
- **At or above 0.53:** Possible impairment-like speech pattern
- **Display score:** raw impaired-class classifier score × 100; not a disease probability

## Authoritative official held-out TEST result

| Metric | Result |
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

Confusion counts: TN 33, FP 7, FN 9, TP 31. The TEST partition is consumed and must not be rerun or used for iterative development.

## Explanation and result interpretation

- **Research explainability:** SHAP on the Logistic Regression log-odds scale.
- **Runtime explanation:** exact standardized Logistic Regression log-odds contributions; not probability changes and not labelled runtime SHAP.
- **TRAIN profile:** Healthy TRAIN and Impaired TRAIN aggregate references across the same 51 frozen predictors.
- **Standardized chart:** zero is the frozen TRAIN preprocessing center; positive does not automatically mean impaired.
- **Coefficient direction:** positive pushes the fitted model toward Possible impairment-like; negative pushes it toward Healthy-like.
- **Scientific limit:** model behavior and cohort associations do not establish medical causes.

**A live demonstration score is an individual model output, not performance evidence. Use the official held-out metrics above when discussing system performance.**
