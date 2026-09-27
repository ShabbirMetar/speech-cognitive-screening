# Speech-Based Cognitive Decline Screening

This repository is the initial skeleton for a research prototype exploring speech-based cognitive decline screening. It is not a clinical diagnostic tool and makes no claims about model performance or accuracy.

## Architecture

- `frontend/`: React application built with Vite and JavaScript.
- `backend/`: FastAPI service for application APIs and configuration.
- `ml/`: Reserved locations for future data interfaces, audio/NLP/ASR work, feature engineering, models, and explainability.

## Research data

PROCESS-2 is an external controlled-access research dataset. It remains outside this Git repository and must never be committed, copied, uploaded, or placed in the application source tree. Configure its local location through `PROCESS2_DATASET_PATH` in a local `.env` file based on `.env.example`.

The application reads the local dataset location from `PROCESS2_DATASET_PATH`. Dataset files remain read-only and external to this repository. Metadata validation and descriptive EDA read the CSV without changing it; raw audio and transcripts are neither copied nor processed.

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

## Current status

The repository contains a health-check FastAPI service, a minimal React landing page, read-only PROCESS-2 validation, and metadata EDA. Cognitive screening, audio/NLP processing, feature extraction, and machine-learning models have not been implemented.
