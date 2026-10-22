# Faculty Demo Checklist

Use this checklist on the actual laptop and browser intended for the faculty demonstration. Use only newly recorded demonstration speech—never PROCESS-2 TEST audio. The application is a research screening prototype, not a medical diagnostic system.

## Before the demo

- [ ] Connect and test the intended microphone in the operating system and browser.
- [ ] Use a quiet room and close CPU-heavy applications; faster-whisper runs locally on CPU.
- [ ] Confirm the repository is on the intended committed revision and `git status` is clean.
- [ ] Open the CTD step once and confirm the bundled Picture Description stimulus loads clearly. Use `VITE_CTD_STIMULUS_URL` only when an authorized override is required.
- [ ] Start the backend from the repository root:

  ```powershell
  cd D:\minorProject\Project\speech-cognitive-screening
  .\.venv\Scripts\Activate.ps1
  uvicorn backend.app.main:app --reload
  ```

- [ ] Open `http://127.0.0.1:8000/api/health` and wait for `screening_model_ready: true`.
- [ ] Keep `http://127.0.0.1:8000/docs` open as the authorized/demo-WAV fallback interface.
- [ ] Start the frontend in a second terminal:

  ```powershell
  cd D:\minorProject\Project\speech-cognitive-screening\frontend
  Copy-Item .env.example .env.local
  npm install
  npm run dev
  ```

- [ ] Open `http://localhost:5173`, grant microphone permission, and confirm the service indicator reports ready.
- [ ] Have the faculty dossier/report and [FACULTY_QUICK_REFERENCE.md](FACULTY_QUICK_REFERENCE.md) available.
- [ ] Confirm the official metrics remain the documented held-out results; do not run `scripts/evaluate_official_test.py`.

## Demo flow

1. Explain the screening problem and the research-only, non-diagnostic scope.
2. Explain PROCESS-2: 400 participants, predefined 320 TRAIN / 80 TEST participant split.
3. Show the local architecture: browser WAV → FastAPI → base.en ASR/features → frozen pipeline → result.
4. Explain SFT, PFT, and CTD without giving response examples that could prime the speaker.
5. Complete the three recordings using new demonstration speech.
6. Review playback, duration, and completion state for every recording.
7. Submit once with **Analyze speech**.
8. Explain that CPU ASR is local and progress is indeterminate; do not repeatedly submit.
9. Show the Cognitive Speech Screening Score as a display transformation, not a disease probability.
10. Explain `53 / 100` as the TRAIN-derived research operating point, not a medical cutoff.
11. Show the three automatic transcripts and note that ASR may contain recognition errors.
12. Show the selected neutral speech measurements.
13. Show the complete standardized 51-feature profile.
14. Explain that Healthy TRAIN and Impaired TRAIN references are descriptive aggregates, not clinical normal/disease ranges.
15. Expand the all-51 feature table, including raw values, imputation status, standardized values, and model direction.
16. Show local Logistic Regression log-odds contributors and distinguish them from research SHAP analysis.
17. Expand model details: Logistic Regression, base.en, 51 predictors, research threshold `0.53`.
18. Show the research/medical disclaimer.
19. Present the official held-out metrics separately from the live result.

**One live person's score is not performance evidence. The frozen pipeline's official held-out TEST metrics are the performance evidence.**

## Failure fallback

- **Microphone permission fails:** allow microphone access for `localhost`, confirm the correct input device, then reload. Do not bypass browser security warnings.
- **Frontend cannot start or record:** use Swagger with newly created authorized/demo WAV files; never use consumed TEST recordings.
- **Backend unavailable:** confirm the virtual environment, backend terminal, and `/api/health`; do not proceed until `screening_model_ready` is true.
- **ASR is slow:** wait for the current request. Do not repeatedly submit or start parallel inference jobs.
- **CTD image fails to load:** reload the frontend and verify `frontend/public/assets/cookie-theft-picture.png` is present; remove an invalid `VITE_CTD_STIMULUS_URL` override so the bundled image is used.
- **Reference visualization unavailable:** explain that it is optional visualization metadata. The core score, classification, transcripts, measurements, and local contributors should still work.
- **Artifact integrity/model startup fails:** stop the demo inference path and use the saved project documentation. Do not regenerate or replace frozen model artifacts during the demonstration.
- **No fake fallback:** never substitute a mocked prediction or present fixture data as real inference.

## One final real-microphone trial

Perform once on the actual faculty-demo laptop/browser with new non-PROCESS-2 speech.

- [ ] Record trial date, browser/version, and microphone used.
- [ ] Start a timer at **Start assessment** and record total assessment duration: `____ min ____ s`.
- [ ] Verify microphone permission and visible recording state.
- [ ] Record, stop, play back, and accept SFT, PFT, and CTD independently.
- [ ] Confirm Review shows three recordings and realistic durations.
- [ ] Start a second timer when selecting **Analyze speech**.
- [ ] Confirm exactly one real multipart API request is submitted.
- [ ] Record Analyze → Results duration: `____ min ____ s`.
- [ ] Verify score, backend-returned classification, threshold marker, and disclaimer.
- [ ] Verify all three automatic transcripts and word counts.
- [ ] Verify selected measurements and the three-series 51-feature profile.
- [ ] Test all four profile filters and expand the all-51 table.
- [ ] Verify model-direction wording and local log-odds contributors.
- [ ] Select **Start new assessment** and confirm old playback URLs/results are cleared and the app returns home.
- [ ] Record any operational issue without changing model, schema, ASR, preprocessing, or threshold.
