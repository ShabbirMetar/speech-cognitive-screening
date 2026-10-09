# Speech-Based Cognitive Decline Screening — Durable Handoff

## Project objective

Develop a research/screening prototype that analyzes recorded speech for possible cognitive-decline signals. The intended system combines ASR-derived linguistic features with acoustic pause, timing, prosodic, and energy features. It must never be presented as a medical diagnosis.

## Dataset and split

- Dataset: external controlled-access PROCESS-2 at `D:\minorProject\PROCESS-2`; raw files are not committed or modified.
- Participants: 400 total: HC 200, MCI 150, Dementia 50.
- Binary label: Healthy = HC; Impaired = MCI + Dementia.
- Official split: TRAIN 320 (160 Healthy, 160 Impaired); TEST 80 (40 Healthy, 40 Impaired).
- Participant—not the 1,200 recordings—is the unit of independence.

## No-leakage policy

Feature extraction was applied deterministically and uniformly to all PROCESS-2 samples before model fitting. Feature selection, preprocessing, cross-validation, tuning, model/representation comparison, and threshold selection must use only the predefined TRAIN partition. The TEST feature rows are isolated for one final evaluation after the complete pipeline and threshold are frozen.

Development code must call `ml.data.development_guard.assert_train_only`; it must fail if a TEST or mixed-split table reaches modelling. Do not load `*_TEST_LOCKED.csv` for development work. Do not transcribe or evaluate TEST now.

## Environment

- Windows 11, Intel i5 (4 physical / 8 logical cores), 16 GB RAM, no GPU.
- One environment only: `D:\minorProject\Project\speech-cognitive-screening\.venv`.
- Use `\.venv\Scripts\python.exe` or activate that environment; never create `backend\.venv`.

## Completed stages

1. Dataset validation and metadata EDA.
2. Manual transcript parser and deterministic linguistic features (44 predictors across SFT/PFT/CTD); manual pause annotations are part of the research reference.
3. Frozen acoustic extraction: 1,200/1,200 WAV files successful. Pure acoustic subset is 13 features/task, 39 total: duration, voiced/silence timing, pause statistics, F0 mean/std/range, RMS mean/std. Legacy `recording_word_rate_wpm` and `articulation_rate_wpm` in the acoustic table are transcript-derived, not pure acoustic.
4. TRAIN-only linguistic/classical and acoustic/modality experiments.
5. faster-whisper ASR pilot: `base.en` is frozen for deployment. On 30 pilot WAVs it achieved overall WER 0.342 and mean runtime 3.45 s/WAV; `small.en` is not a current option.
6. Full TRAIN-only `base.en` transcription: 960/960 successful, 0 failed. Outputs include ASR transcripts, ASR linguistic features, and deployment features.
7. TRAIN-only deployment representation comparison completed on 2026-10-02.
8. Focused TRAIN-only fusion strategy comparison and freeze of the 51-feature `ASR_RATE_PLUS_CTD` representation.
9. TRAIN-only calibration, threshold, and explainability analysis completed. No TEST rows were loaded, transcribed, or evaluated.

## Frozen decisions

- Use faster-whisper `base.en`, CPU, `int8`, English, 4 CPU threads, one worker for deployment-oriented ASR. Do not introduce Whisper medium or revisit small.en unless explicitly asked.
- Preserve acoustic features as a core project modality even when their standalone or fused performance is weaker.
- Frozen deployment representation: `ASR_RATE_PLUS_CTD`, with 51 predictors: 32 ASR linguistic features, 6 ASR-count-derived speech-rate features, and 13 CTD pure acoustic pause/prosodic/energy features.
- Frozen classifier family: Logistic Regression with fold-local median imputation, standard scaling, and inner-CV selection from `C = [0.1, 1.0, 10.0]`.
- Raw Logistic Regression classifier scores are retained; sigmoid calibration is not adopted.
- Primary research operating threshold: `0.53`, derived from TRAIN out-of-fold predictions because both maximum balanced accuracy and maximum Youden J selected it. This is not a clinical, medical, or validated diagnostic threshold. The lower sensitivity-oriented thresholds remain descriptive alternatives only.
- Keep age, gender, and MMSE out of the primary predictive model.
- Handle missing Honoré statistics with fold-local median imputation; do not drop them without a separately specified ablation.
- Use interpretable/regularized models before complex models. The all-TRAIN serialized production artifact has not yet been created.

## Current artifacts

- Manual TRAIN features: `artifacts/features/linguistic_features_train.csv` — 320 rows, 44 predictors.
- ASR linguistic TRAIN features: `artifacts/features/asr_linguistic_features_train.csv` — 320 rows, 32 predictors.
- Acoustic TRAIN features: `artifacts/features/acoustic_features_train.csv` — 320 rows, 150 predictors.
- Deployment TRAIN features: `artifacts/features/deployment_features_train.csv` — 320 rows, 77 predictors = 32 ASR linguistic + 6 ASR-count-derived rate + 39 pure acoustic.
- Full ASR transcripts: `artifacts/transcripts/asr_train/base_en_train_transcripts.csv` — 960 rows.
- ASR quality: `artifacts/results/asr_train_feature_quality.csv`. `pft_honore_statistic` has 67 missing and `sft_honore_statistic` has 55 missing; no ASR linguistic feature is infinite or constant.
- Latest comparison: `artifacts/results/deployment_representation_comparison/`.
- Explainability outputs: `artifacts/results/explainability/` — global SHAP/standardized-coefficient tables, feature-group reliance, deterministic TRAIN-only example explanations, figures, and provenance metadata.

## Actual observed metrics

All values below are five-fold outer TRAIN CV, Logistic Regression with inner three-fold F1 tuning of `C = [0.1, 1.0, 10.0]`. Imputation and scaling occur inside the pipeline. These are not TEST results and do not establish clinical performance.

| Representation | Features | F1 (mean ± SD) | ROC-AUC (mean ± SD) | Train–validation F1 gap |
|---|---:|---:|---:|---:|
| MANUAL_FULL | 44 | 0.674 ± 0.064 | 0.762 ± 0.048 | 0.068 |
| MANUAL_ASR_COMPATIBLE | 32 | 0.647 ± 0.047 | 0.733 ± 0.055 | 0.063 |
| ASR_LINGUISTIC | 32 | 0.635 ± 0.084 | 0.704 ± 0.074 | 0.083 |
| ASR_LINGUISTIC_PLUS_RATE | 38 | 0.640 ± 0.088 | 0.713 ± 0.079 | 0.081 |
| DEPLOYMENT_MULTIMODAL | 77 | 0.616 ± 0.068 | 0.697 ± 0.066 | 0.137 |

Paired fold-wise F1 deltas (mean ± SD; descriptive only, no significance claims):

- ASR linguistic minus manual ASR-compatible: -0.013 ± 0.044.
- ASR linguistic plus rate minus ASR linguistic: +0.006 ± 0.022.
- Deployment multimodal minus ASR linguistic plus rate: -0.024 ± 0.039.
- Deployment multimodal minus ASR linguistic: -0.019 ± 0.050.
- Deployment multimodal minus manual full: -0.058 ± 0.036.

Earlier reference results: manual linguistic/classical Logistic Regression had F1 0.676 and ROC-AUC 0.762; CTD pure acoustic Logistic Regression had F1 0.605 ± 0.076 and ROC-AUC 0.668 ± 0.084. These are research-reference experiments, not deployment-model choices.

## Focused fusion-strategy results

The final planned TRAIN-only fusion-strategy experiment compared the 38-feature ASR-rate representation, the 13-feature CTD pure-acoustic subset, a 51-feature compact early fusion, and separate probability-level late fusion. All use the same fixed outer folds. Late fusion tunes ASR/CTD Logistic Regression `C` values and `alpha ∈ {0.6, 0.7, 0.8, 0.9}` only inside each outer fold's inner CV; class metrics use the fixed 0.5 threshold.

| Representation | Features/mode | F1 (mean ± SD) | ROC-AUC (mean ± SD) | Train–validation F1 gap |
|---|---|---:|---:|---:|
| ASR_RATE | 38 | 0.640 ± 0.088 | 0.713 ± 0.079 | 0.081 |
| CTD_ACOUSTIC | 13 | 0.605 ± 0.076 | 0.668 ± 0.084 | 0.056 |
| ASR_RATE_PLUS_CTD | 51 compact early fusion | 0.650 ± 0.085 | 0.717 ± 0.079 | 0.073 |
| LATE_FUSION_ASR_CTD | separate 38 + 13 pipelines | 0.632 ± 0.079 | 0.721 ± 0.090 | 0.081 |
| DEPLOYMENT_MULTIMODAL_REFERENCE | 77-feature earlier early fusion, not rerun | 0.616 ± 0.068 | 0.697 ± 0.066 | 0.137 |

Paired F1 deltas (mean ± SD; descriptive only):

- Compact early fusion minus ASR_RATE: +0.010 ± 0.042.
- Late fusion minus ASR_RATE: -0.008 ± 0.023.
- Late fusion minus compact early fusion: -0.018 ± 0.027.

Late-fusion inner selections by outer fold were: alpha 0.7, 0.7, 0.8, 0.9, and 0.7; ASR-rate C values 1.0, 1.0, 0.1, 0.1, and 0.1; CTD-acoustic C values 10.0, 0.1, 0.1, 0.1, and 10.0. Compact early fusion has the highest observed mean F1 in this focused comparison, but its small fold-wise improvement is inconsistent. These results do not justify removing acoustic analysis or making significance claims.

Artifacts: `artifacts/results/fusion_strategy_comparison/`.

## TRAIN-only calibration and threshold analysis

The frozen 51-feature Logistic Regression representation was evaluated with one outer-fold OOF classifier score per TRAIN participant. Each outer fold tuned `C` only within its outer-TRAIN partition. The raw score is a **classifier probability-like screening score**, not a disease or dementia probability.

Calibration metrics below pool all 320 OOF predictions. Therefore the raw pooled OOF ROC-AUC (0.7137) is not expected to equal the earlier mean of five fold-level ROC-AUC values (0.7172); the underlying 320 OOF scores, labels, folds, and predictions are identical.

| Score source | Brier score | Log loss | ROC-AUC |
|---|---:|---:|---:|
| Raw Logistic Regression | 0.2167 | 0.6559 | 0.7137 |
| Sigmoid-calibrated Logistic Regression | 0.2179 | 0.6357 | 0.7115 |

Sigmoid calibration was cross-fitted within each outer TRAIN fold as a diagnostic comparator. It improved log loss but slightly worsened Brier score and ROC-AUC, so raw Logistic Regression scores remain the operating-score source; no automatic calibration change was made.

Objective raw-score threshold candidates (TRAIN OOF only) are:

| Rule | Threshold | Sensitivity | Specificity | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| Max F1 | 0.27 | 0.919 | 0.300 | 0.702 | 0.609 |
| Max balanced accuracy / Youden J | 0.53 | 0.594 | 0.744 | 0.642 | 0.669 |
| Sensitivity at least 0.70 (descriptive) | 0.43 | 0.700 | 0.556 | 0.653 | 0.628 |
| Sensitivity at least 0.75 (descriptive) | 0.39 | 0.769 | 0.475 | 0.670 | 0.622 |
| Sensitivity at least 0.80 (descriptive) | 0.37 | 0.806 | 0.444 | 0.683 | 0.625 |

The primary research operating threshold is now frozen at `0.53` because maximum balanced accuracy and maximum Youden J independently selected it from the TRAIN OOF sweep. It is not clinically validated and must be described only as a TRAIN-derived research operating point. Sensitivity-target rows remain descriptive and do not imply a clinical target. Artifacts: `artifacts/results/screening_threshold/`.

## TRAIN-only explainability

The frozen 51-feature Logistic Regression representation was fitted once on all 320 TRAIN participants after a five-fold TRAIN-only F1 resolution of `C ∈ {0.1, 1.0, 10.0}`. `C = 0.1` was selected (mean CV F1 0.650 ± 0.076); this resolution is not a new model-family or representation comparison. The all-TRAIN fit is used for explanation only and does not provide held-out performance.

`shap.LinearExplainer` receives the same median-imputed and standardized matrix used by the classifier. SHAP values are on the model log-odds scale, not the probability scale, and reconstruct the fitted decision function with maximum numerical error `1.776e-15`. The global mean-absolute-SHAP group shares are ASR linguistic 63.9%, ASR rate 9.6%, and CTD acoustic 26.5%. These are descriptive model-reliance summaries, not causal or clinical biomarker importance.

The leading global features were PFT P-initial word count, PFT P-initial ratio, CTD mean voice energy, SFT repeated-word count, PFT average word length, PFT Honoré vocabulary richness, SFT Brunet vocabulary richness, and CTD silence ratio. Their directions describe this fitted model only; they must not be interpreted as causal effects. Deterministic low-score, near-threshold, and high-score TRAIN examples are included solely for demonstration in `artifacts/results/explainability/example_explanations.csv` and are not held-out evidence.

## Current task and immediate next task

The calibration, threshold, and explainability stages are complete. The next task is final all-TRAIN fitting, artifact serialization, and complete deployment-pipeline freeze: ASR configuration, feature schema/order, imputer, scaler, classifier, raw-score terminology, and the 0.53 research operating threshold. Only after that freeze may one official TEST evaluation be performed. Do not begin another representation/model search.

## Explicitly do not do

- Do not run Whisper again for this completed TRAIN batch or process TEST.
- Do not train on manual features and deploy untested ASR features as though they were interchangeable.
- Do not use TEST for feature selection, tuning, thresholds, model choice, SHAP, or comparison.
- Do not call the current multimodal result superior; its observed F1 and ROC-AUC were lower than ASR linguistic and its F1 gap was larger.
- Do not remove acoustic features from the project objective based solely on this result.
- Do not treat the compact early-fusion F1 increase as definitive; it is small and inconsistent across five folds.
- Do not call any TRAIN-derived threshold clinically validated or present a classifier score as probability of dementia, disease, or diagnostic confidence.
- Do not claim diagnosis, clinical validation, or significance from five folds.
