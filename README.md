# Speech-Based Cognitive Decline Screening

This repository contains a research prototype for preliminary speech-based cognitive-decline screening. It is not a clinical diagnostic tool and must not be presented as a diagnosis, medical probability, or clinically validated assessment.

## Architecture

- `frontend/`: React application built with Vite and JavaScript.
- `backend/`: FastAPI service for application APIs and configuration.
- `ml/`: Data interfaces, audio/NLP/ASR work, feature engineering, models, and reusable explainability helpers.

## Research data

PROCESS-2 is an external controlled-access research dataset. It remains outside this Git repository and must never be committed, copied, uploaded, or placed in the application source tree. Configure its local location through `PROCESS2_DATASET_PATH` in a local `.env` file based on `.env.example`.

The application reads the local dataset location from `PROCESS2_DATASET_PATH`. Dataset files remain read-only and external to this repository. Metadata validation and descriptive EDA read the CSV without changing it. Any approved audio or transcript processing is read-only and writes only derived local artifacts inside this repository; raw dataset files are never copied or altered.

## Local environment

Use the single virtual environment at the project root. Do not create or use a separate `backend\.venv`.

```powershell
cd D:\minorProject\Project\speech-cognitive-screening
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

## Metadata EDA

Run the read-only metadata EDA with:

```powershell
cd D:\minorProject\Project\speech-cognitive-screening
.\.venv\Scripts\python.exe .\scripts\run_eda.py
```

Figures and a compact summary CSV are written to `artifacts/results/eda/`. The EDA creates `screening_label` only in memory: `HC` is `Healthy`, while `MCI` and `Dementia` are `Impaired`.

## Data-split rule

The official TEST split must remain untouched during model development. Metadata-level descriptive plots may summarize the complete dataset for documentation, but predictive feature analysis, feature selection, model fitting, and hyperparameter tuning must operate only on TRAIN participants.

Feature extraction was applied deterministically and uniformly to all PROCESS-2 samples before model fitting. All feature selection, cross-validation, hyperparameter tuning, model comparison and threshold selection are restricted to the predefined TRAIN partition. The predefined TEST feature rows are isolated and reserved for one final evaluation after the complete pipeline is frozen.

## Manual transcript processing

PROCESS-2 transcripts can contain speaker labels, pause annotations, disfluencies, and contributions from other speakers. The manual transcript pipeline therefore extracts linguistic features from `Pat:` speech when that label is available and excludes `Oth:` speech. Pause annotations are recorded before cleaning and do not count as words.

Some transcripts have no speaker labels. For those files only, the parser uses the complete unlabeled text as a participant-text candidate and records the explicit `unlabeled_fallback` state. If speaker labels are present but no `Pat:` label is present, no participant speaker identity is inferred and the participant-text field is empty.

Run structural inspection (TRAIN by default) without displaying participant speech:

```powershell
cd D:\minorProject\Project\speech-cognitive-screening
.\.venv\Scripts\python.exe .\scripts\inspect_transcripts.py
```

Build deterministic manual-transcript features for all participants:

```powershell
cd D:\minorProject\Project\speech-cognitive-screening
.\.venv\Scripts\python.exe .\scripts\build_manual_transcript_features.py
```

The resulting table is written to `artifacts/features/manual_transcript_features.csv`. Its `diagnosis`, `screening_label`, participant ID, and split columns are labels/bookkeeping only; age, gender, and MMSE are not included. The official TEST split is never used for learned preprocessing, feature selection, or model development.

When a participant transcript is unavailable or cannot be attributed to `Pat:`, its task-specific linguistic measurements are recorded as missing (`NaN`), not as genuine zero linguistic activity. In contrast, a successfully parsed participant transcript with no detected pause annotation has pause mean and maximum values of `0`. Honoré's statistic remains `NaN` whenever its formula is mathematically undefined.

## Linguistic Logistic Regression baseline

The initial baseline uses only task-prefixed linguistic features and the `screening_label` target (`Healthy=0`, `Impaired=1`). It uses the official TRAIN participants only with five-fold stratified cross-validation. Median imputation and standard scaling are inside the sklearn pipeline, so each validation fold is transformed only with its corresponding training-fold information. TEST participants are not evaluated.

```powershell
cd D:\minorProject\Project\speech-cognitive-screening
.\.venv\Scripts\python.exe .\scripts\train_linguistic_baseline.py
```

Outputs are written to `artifacts/results/baseline/`; out-of-fold predictions contain only anonymized participant identifiers.

## Acoustic feature pilot

The initial acoustic pipeline is a small, read-only TRAIN-only pilot. It uses librosa to calculate interpretable duration, energy, zero-crossing, spectral, MFCC, F0, energy-based silence/pause, and transcript-word-rate features for a reproducibly selected ten-participant subset. It does not train a model or evaluate the official TEST split.

```powershell
cd D:\minorProject\Project\speech-cognitive-screening
.\.venv\Scripts\python.exe .\scripts\pilot_acoustic_features.py
```

The pilot writes derived features to `artifacts/features/acoustic_pilot_features.csv` and sanity-check plots to `artifacts/results/acoustic_pilot/`.

Important limitations:

- Librosa energy-based silence detection is an engineering baseline, not a clinically validated final VAD method.
- Real-world background noise can affect silence and pause estimates.
- Pitch extraction can fail; unavailable F0 values remain `NaN`.
- Recording duration and transcript-derived word-rate estimates can be affected by other-speaker contributions.
- Silero VAD or another speech detector may be evaluated later.

## Frozen full acoustic extraction

After the pilot parameters are approved, run the frozen extractor to create one participant row with task-prefixed acoustic features. It applies the same deterministic extractor to TRAIN and TEST recordings because no parameters are learned during extraction. It must not use TEST labels to alter thresholds, pitch settings, feature definitions, feature selection, model tuning, or model comparison.

```powershell
cd D:\minorProject\Project\speech-cognitive-screening
.\.venv\Scripts\python.exe .\scripts\build_acoustic_features.py
```

Derived outputs are stored only under `artifacts/features/`: `acoustic_features.csv`, a participant-free frozen configuration record, and checkpoint files that allow interrupted processing to resume. A checkpoint is reused only when its configuration fingerprint matches the frozen extractor settings. Raw PROCESS-2 WAV files are never modified or resaved.

## Current status

Completed research-pipeline stages are:

- PROCESS-2 validation and metadata EDA.
- Manual transcript parsing and linguistic feature extraction.
- Frozen full acoustic feature extraction and TRAIN/locked-TEST feature partitioning.
- TRAIN-only nested-CV linguistic/classical, acoustic-baseline, and pure-acoustic modality-comparison experiments.
- faster-whisper `base.en` pilot and full TRAIN-only transcription (320 participants, 960 recordings), followed by ASR linguistic and deployment-feature generation.
- TRAIN-only deployment representation comparison across manual, ASR, ASR-rate, and pure-acoustic multimodal representations.
- Focused TRAIN-only fusion comparison. The frozen deployment representation is `ASR_RATE_PLUS_CTD`: 32 ASR-compatible linguistic features, 6 ASR-count-derived speech-rate features, and 13 CTD waveform-only acoustic features (51 total). Logistic Regression is the frozen classifier family.
- TRAIN-only calibration and threshold analysis. Raw Logistic Regression classifier scores are retained; the primary research operating threshold is `0.53`, selected independently by maximum balanced accuracy and Youden J on TRAIN out-of-fold scores. It is not a clinical or validated diagnostic threshold.
- TRAIN-only SHAP explainability for the frozen 51-feature Logistic Regression model. SHAP values are calculated on the standardized model log-odds scale; they describe model behaviour, not causal or clinical biomarkers. Outputs are in `artifacts/results/explainability/`.

The current explainability fit selected `C=0.1` with five-fold TRAIN CV F1 selection (`0.650 ± 0.076` across folds). Its strongest global mean-absolute-SHAP features include PFT P-initial word count/ratio, CTD mean voice energy, SFT repeated-word count, and CTD silence ratio. The exact tables and figures are saved with the artifact outputs.

No final all-TRAIN serialized deployment artifact has been created, and the official TEST partition has not been evaluated. FastAPI ML integration and the React microphone workflow remain future stages.

Run the reproducible, TRAIN-only explainability analysis with:

```powershell
cd D:\minorProject\Project\speech-cognitive-screening
.\.venv\Scripts\python.exe .\scripts\analyze_model_explainability.py
```

The display-only Cognitive Speech Screening Score is `raw classifier score × 100`. It is not a probability of dementia, disease, or clinical risk percentage. At the TRAIN-derived research operating point, a raw score of at least `0.53` maps to “Possible impairment-like speech pattern”; lower scores map to “Healthy-like speech pattern.”
