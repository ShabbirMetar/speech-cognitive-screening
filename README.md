# Speech-Based Cognitive Decline Screening

This repository is the initial skeleton for a research prototype exploring speech-based cognitive decline screening. It is not a clinical diagnostic tool and makes no claims about model performance or accuracy.

## Architecture

- `frontend/`: React application built with Vite and JavaScript.
- `backend/`: FastAPI service for application APIs and configuration.
- `ml/`: Reserved locations for future data interfaces, audio/NLP/ASR work, feature engineering, models, and explainability.

## Research data

PROCESS-2 is an external controlled-access research dataset. It remains outside this Git repository and must never be committed, copied, uploaded, or placed in the application source tree. Configure its local location through `PROCESS2_DATASET_PATH` in a local `.env` file based on `.env.example`.

The backend configuration can verify that the configured path exists, but this initial skeleton does not inspect or process any dataset content.

## Current status

The repository currently contains only the initial project structure, a health-check FastAPI service, and a minimal React landing page. Cognitive screening, dataset analysis, audio/NLP processing, and machine-learning models have not been implemented.
