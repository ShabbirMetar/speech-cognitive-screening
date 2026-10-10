"""ONE-TIME OFFICIAL HELD-OUT TEST EVALUATION.

No model, preprocessing, feature, ASR, calibration, or threshold tuning is
permitted from these results.  The script loads the already frozen pipeline
and uses only its persisted preprocessing and classifier parameters.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
for import_root in (PROJECT_ROOT, BACKEND_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


from ml.asr.asr_linguistic_features import (
    BOOKKEEPING_COLUMNS,
    asr_feature_columns,
    build_asr_feature_tables_for_split,
)
from ml.asr.whisper_transcriber import WhisperTranscriptionConfig, load_whisper_model, transcribe_audio
from ml.models.baseline_logistic import encode_screening_labels
from ml.models.explainability import RESEARCH_OPERATING_THRESHOLD
from ml.models.frozen_pipeline import load_frozen_artifacts, predict_screening
from scripts.transcribe_train_base_en import (
    TRANSCRIPT_COLUMNS,
    _atomic_csv_write,
    _atomic_json_write,
    build_directory_index,
    build_split_worklist,
    format_timing_messages,
    get_dataset_root,
    load_checkpoint,
    non_hidden_directories,
    participant_directory_for_id,
    remaining_worklist,
    task_wav_paths,
    validate_transcript_frame,
    write_or_validate_frozen_config,
)


TEST_ACOUSTIC_PATH = PROJECT_ROOT / "artifacts" / "features" / "acoustic_features_test_LOCKED.csv"
ASR_CONFIG_SOURCE_PATH = PROJECT_ROOT / "artifacts" / "transcripts" / "asr_train" / "asr_config.json"
TEST_TRANSCRIPT_DIRECTORY = PROJECT_ROOT / "artifacts" / "transcripts" / "asr_test_final"
TEST_ASR_CONFIG_PATH = TEST_TRANSCRIPT_DIRECTORY / "asr_config.json"
TEST_CHECKPOINT_PATH = TEST_TRANSCRIPT_DIRECTORY / "base_en_test_checkpoint_FINAL.csv"
TEST_TRANSCRIPT_PATH = TEST_TRANSCRIPT_DIRECTORY / "base_en_test_transcripts_FINAL.csv"
TEST_ASR_FEATURE_PATH = PROJECT_ROOT / "artifacts" / "features" / "asr_linguistic_features_test_FINAL.csv"
TEST_DEPLOYMENT_FEATURE_PATH = PROJECT_ROOT / "artifacts" / "features" / "deployment_features_test_FINAL.csv"
TEST_FEATURE_PATH = PROJECT_ROOT / "artifacts" / "features" / "official_test_features_FINAL.csv"
OUTPUT_DIRECTORY = PROJECT_ROOT / "artifacts" / "results" / "official_test_evaluation"
PREDICTIONS_PATH = OUTPUT_DIRECTORY / "test_predictions.csv"
METRICS_PATH = OUTPUT_DIRECTORY / "final_metrics.csv"
BOOTSTRAP_PATH = OUTPUT_DIRECTORY / "bootstrap_confidence_intervals.csv"
DIAGNOSIS_BREAKDOWN_PATH = OUTPUT_DIRECTORY / "diagnosis_prediction_breakdown.csv"
METADATA_PATH = OUTPUT_DIRECTORY / "evaluation_metadata.json"
EVALUATION_LOCK_PATH = OUTPUT_DIRECTORY / ".official_test_evaluation.lock"
CONFUSION_MATRIX_PATH = OUTPUT_DIRECTORY / "confusion_matrix.png"
ROC_PATH = OUTPUT_DIRECTORY / "roc_curve.png"
SCORE_DISTRIBUTION_PATH = OUTPUT_DIRECTORY / "test_score_distribution.png"
EXPECTED_TEST_PARTICIPANTS = 80
EXPECTED_TEST_RECORDINGS = 240
FROZEN_THRESHOLD = 0.53
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_RANDOM_STATE = 42


def claim_official_evaluation_lock(lock_path: Path = EVALUATION_LOCK_PATH) -> None:
    """Atomically reserve the one-time held-out evaluation before any TEST work.

    The lock deliberately remains after either completion or interruption. A
    human must review an interrupted run before permitting any further TEST
    processing, which prevents two evaluators from racing before metadata is
    written at the end of the run.
    """

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "process_id": os.getpid(),
        "purpose": "one-time official held-out TEST evaluation",
    }
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        raise RuntimeError(
            "Official TEST evaluation lock already exists; refusing concurrent or repeat TEST processing. "
            "Review the recorded run before any manual recovery."
        ) from error
    try:
        os.write(descriptor, json.dumps(payload, indent=2).encode("utf-8"))
    finally:
        os.close(descriptor)


def assert_official_test_not_already_recorded(metadata_path: Path = METADATA_PATH) -> None:
    """Reject any attempt to run a consumed official TEST evaluation again."""

    if metadata_path.exists():
        raise RuntimeError("Official TEST evaluation is already recorded; refusing a second evaluation run.")


def _frozen_asr_config(artifacts: Any) -> WhisperTranscriptionConfig:
    """Build the transcriber config only from the hash-validated frozen record."""

    if not ASR_CONFIG_SOURCE_PATH.is_file():
        raise FileNotFoundError(f"Frozen TRAIN ASR configuration is missing: {ASR_CONFIG_SOURCE_PATH}")
    source_config = json.loads(ASR_CONFIG_SOURCE_PATH.read_text(encoding="utf-8"))
    metadata_config = artifacts.metadata.get("asr_deployment_configuration")
    if source_config != metadata_config:
        raise ValueError("Frozen ASR source configuration differs from model metadata.")
    required = {"model_size", "device", "compute_type", "language", "cpu_threads", "num_workers"}
    missing = required.difference(source_config)
    if missing:
        raise ValueError(f"Frozen ASR configuration is missing: {sorted(missing)}")
    config = WhisperTranscriptionConfig(
        model_size=str(source_config["model_size"]),
        device=str(source_config["device"]),
        compute_type=str(source_config["compute_type"]),
        language=str(source_config["language"]),
        cpu_threads=int(source_config["cpu_threads"]),
        num_workers=int(source_config["num_workers"]),
    )
    expected = ("base.en", "cpu", "int8", "en", 4, 1)
    observed = (config.model_size, config.device, config.compute_type, config.language, config.cpu_threads, config.num_workers)
    if observed != expected:
        raise ValueError(f"Frozen ASR configuration differs from the approved base.en settings: {observed}")
    return config


def specificity(target: np.ndarray, prediction: np.ndarray) -> float:
    """Return Healthy-class recall for fixed binary screening predictions."""

    true_negative, false_positive, _, _ = confusion_matrix(target, prediction, labels=[0, 1]).ravel()
    denominator = true_negative + false_positive
    return float(true_negative / denominator) if denominator else float("nan")


def fixed_threshold_metrics(target: np.ndarray, scores: np.ndarray, threshold: float = FROZEN_THRESHOLD) -> dict[str, float | int]:
    """Calculate pre-specified held-out metrics without modifying the threshold."""

    y_true = np.asarray(target, dtype=int)
    raw_scores = np.asarray(scores, dtype=float)
    prediction = (raw_scores >= threshold).astype(int)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        y_true, prediction, labels=[0, 1]
    ).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "sensitivity": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": specificity(y_true, prediction),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, raw_scores)),
        "brier_score": float(brier_score_loss(y_true, raw_scores)),
        "log_loss": float(log_loss(y_true, raw_scores, labels=[0, 1])),
        "tn": int(true_negative),
        "fp": int(false_positive),
        "fn": int(false_negative),
        "tp": int(true_positive),
    }


def bootstrap_confidence_intervals(
    target: np.ndarray,
    scores: np.ndarray,
    threshold: float = FROZEN_THRESHOLD,
    replicates: int = BOOTSTRAP_REPLICATES,
    random_state: int = BOOTSTRAP_RANDOM_STATE,
) -> pd.DataFrame:
    """Estimate descriptive participant-level percentile intervals with no refitting."""

    y_true = np.asarray(target, dtype=int)
    raw_scores = np.asarray(scores, dtype=float)
    if len(y_true) != len(raw_scores) or not len(y_true):
        raise ValueError("Bootstrap requires equally sized non-empty targets and scores.")
    point = fixed_threshold_metrics(y_true, raw_scores, threshold)
    metric_names = ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "roc_auc")
    samples: dict[str, list[float]] = {name: [] for name in metric_names}
    generator = np.random.default_rng(random_state)
    for _ in range(replicates):
        indices = generator.integers(0, len(y_true), size=len(y_true))
        sample_target = y_true[indices]
        sample_scores = raw_scores[indices]
        sample_prediction = (sample_scores >= threshold).astype(int)
        non_auc = {
            "accuracy": float(accuracy_score(sample_target, sample_prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(sample_target, sample_prediction)),
            "sensitivity": float(recall_score(sample_target, sample_prediction, zero_division=0)),
            "specificity": specificity(sample_target, sample_prediction),
            "f1": float(f1_score(sample_target, sample_prediction, zero_division=0)),
        }
        for name, value in non_auc.items():
            if np.isfinite(value):
                samples[name].append(value)
        if np.unique(sample_target).size == 2:
            samples["roc_auc"].append(float(roc_auc_score(sample_target, sample_scores)))
    rows: list[dict[str, object]] = []
    for name in metric_names:
        values = np.asarray(samples[name], dtype=float)
        rows.append(
            {
                "metric": name,
                "point_estimate": float(point[name]),
                "ci_lower_95": float(np.percentile(values, 2.5)) if len(values) else float("nan"),
                "ci_upper_95": float(np.percentile(values, 97.5)) if len(values) else float("nan"),
                "valid_replicates": int(len(values)),
                "requested_replicates": int(replicates),
                "bootstrap_random_state": int(random_state),
            }
        )
    return pd.DataFrame(rows)


def diagnosis_prediction_breakdown(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize fixed binary predictions by original diagnosis without new modelling."""

    rows: list[dict[str, object]] = []
    for diagnosis in ("HC", "MCI", "Dementia"):
        subset = predictions.loc[predictions["original_diagnosis"].eq(diagnosis)]
        rows.append(
            {
                "original_diagnosis": diagnosis,
                "participant_count": int(len(subset)),
                "predicted_healthy_like": int(subset["predicted_binary_label"].eq(0).sum()),
                "predicted_possible_impairment_like": int(subset["predicted_binary_label"].eq(1).sum()),
            }
        )
    return pd.DataFrame(rows)


def _plot_confusion(target: np.ndarray, prediction: np.ndarray, destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(5, 4))
    display = ConfusionMatrixDisplay(confusion_matrix(target, prediction, labels=[0, 1]), display_labels=["Healthy", "Impaired"])
    display.plot(ax=axis, cmap="Blues", colorbar=False)
    axis.set_title("Official held-out TEST: fixed-threshold classification")
    figure.tight_layout()
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_roc(target: np.ndarray, scores: np.ndarray, destination: Path) -> None:
    false_positive_rate, true_positive_rate, _ = roc_curve(target, scores)
    auc = roc_auc_score(target, scores)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(false_positive_rate, true_positive_rate, color="#4C78A8", linewidth=2, label=f"Frozen model (AUC = {auc:.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="#666666", label="Chance")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate", ylabel="True positive rate")
    axis.set_title("Official held-out TEST ROC curve (descriptive)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=True, loc="lower right")
    figure.tight_layout()
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_score_distribution(predictions: pd.DataFrame, destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    for label, color in ((0, "#4C78A8"), (1, "#DD8452")):
        values = predictions.loc[predictions["true_binary_label"].eq(label), "raw_model_score"]
        name = "Healthy" if label == 0 else "Impaired"
        axis.hist(values, bins=np.linspace(0, 1, 21), alpha=0.58, label=name, color=color, edgecolor="white")
    axis.axvline(FROZEN_THRESHOLD, color="#222222", linestyle="--", linewidth=2, label="Frozen research threshold = 0.53")
    axis.set(xlim=(0, 1), xlabel="Raw classifier score", ylabel="Participants")
    axis.set_title("Official held-out TEST score distributions")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _transcribe_test(
    test_metadata: pd.DataFrame, config: WhisperTranscriptionConfig
) -> pd.DataFrame:
    """Run only unfinished TEST ASR work with a resume-safe final-evaluation checkpoint."""

    worklist = build_split_worklist(test_metadata, "TEST")
    if len(worklist) != EXPECTED_TEST_RECORDINGS or worklist.duplicated(["participant_id", "task"]).any():
        raise AssertionError("Official TEST worklist must contain exactly 240 unique participant/task rows.")
    write_or_validate_frozen_config(TEST_ASR_CONFIG_PATH, config)
    completed = load_checkpoint(TEST_CHECKPOINT_PATH, worklist)
    if completed["transcript_text"].astype("string").str.strip().eq("").any():
        raise ValueError("TEST ASR checkpoint contains an empty transcript; no content will be fabricated.")
    remaining = remaining_worklist(worklist, completed)
    print(f"TEST ASR checkpoint rows: {len(completed)}; remaining recordings: {len(remaining)}")
    if len(remaining):
        model = load_whisper_model(config)
        directory_index = build_directory_index(non_hidden_directories(get_dataset_root()))
        started_at = perf_counter()
        failures = 0
        for progress, item in enumerate(remaining.itertuples(index=False), start=1):
            participant_directory = participant_directory_for_id(directory_index, item.participant_id)
            wavs = task_wav_paths(participant_directory)[item.task] if participant_directory else []
            if len(wavs) != 1:
                failures += 1
                print(f"[{progress}/{len(remaining)}] {item.task}: failed (expected one WAV)", flush=True)
                continue
            try:
                transcription = transcribe_audio(model, wavs[0], config)
                if not transcription.transcript_text.strip():
                    raise ValueError("empty frozen-ASR transcript")
                row = pd.DataFrame(
                    [{
                        "participant_id": item.participant_id,
                        "task": item.task,
                        "transcript_text": transcription.transcript_text,
                        "audio_duration_seconds": transcription.audio_duration_seconds,
                        "transcription_runtime_seconds": transcription.transcription_runtime_seconds,
                        "model": config.model_size,
                    }]
                )
                completed = pd.concat([completed, row], ignore_index=True)
                validate_transcript_frame(completed, worklist)
                _atomic_csv_write(completed.loc[:, TRANSCRIPT_COLUMNS], TEST_CHECKPOINT_PATH)
                print(f"[{progress}/{len(remaining)}] {item.task}: complete", flush=True)
            except (OSError, RuntimeError, ValueError) as error:
                failures += 1
                print(f"[{progress}/{len(remaining)}] {item.task}: failed ({error.__class__.__name__})", flush=True)
        elapsed = perf_counter() - started_at
        loop_message, _, _ = format_timing_messages(elapsed, 0.0, 1, failures)
        print(loop_message)
    validate_transcript_frame(completed, worklist)
    if len(completed) != len(worklist):
        raise RuntimeError("Official TEST ASR is incomplete; checkpoint retained without fabricated transcripts.")
    completed = completed.sort_values(["participant_id", "task"]).reset_index(drop=True)
    if completed["transcript_text"].astype("string").str.strip().eq("").any():
        raise RuntimeError("Official TEST ASR contains an empty transcript.")
    if completed["task"].value_counts().to_dict() != {"SFT": 80, "PFT": 80, "CTD": 80}:
        raise AssertionError("Official TEST ASR task counts must be 80 per task.")
    _atomic_csv_write(completed.loc[:, TRANSCRIPT_COLUMNS], TEST_TRANSCRIPT_PATH)
    return completed


def main() -> int:
    """Perform the single authorized held-out evaluation and permanently record it."""

    assert_official_test_not_already_recorded()
    claim_official_evaluation_lock()
    if not TEST_ACOUSTIC_PATH.is_file():
        print(f"Required locked TEST acoustic feature table not found: {TEST_ACOUSTIC_PATH}")
        return 1

    artifacts = load_frozen_artifacts()
    threshold = float(artifacts.metadata["research_threshold"])
    if (len(artifacts.schema["ordered_feature_names"]), artifacts.metadata["training_participants"], threshold) != (51, 320, FROZEN_THRESHOLD):
        raise AssertionError("Frozen schema, training count, or threshold differs from the approved final configuration.")
    if artifacts.metadata["official_test_used_during_development"] is not False:
        raise AssertionError("Frozen model metadata incorrectly records TEST use during development.")
    if artifacts.pipeline.named_steps["classifier"].C != 0.1:
        raise AssertionError("Frozen Logistic Regression C must equal 0.1.")
    config = _frozen_asr_config(artifacts)
    print("Official Held-Out TEST Evaluation")
    print("==================================")
    print("Frozen artifact integrity: PASS")

    acoustic_test = pd.read_csv(TEST_ACOUSTIC_PATH)
    if len(acoustic_test) != EXPECTED_TEST_PARTICIPANTS or not acoustic_test["Split"].eq("TEST").all():
        raise AssertionError("Locked acoustic table must contain exactly 80 TEST participants.")
    if acoustic_test["participant_id"].duplicated().any():
        raise AssertionError("Locked acoustic TEST table contains duplicate participant IDs.")
    expected_diagnoses = {"HC": 40, "MCI": 30, "Dementia": 10}
    if acoustic_test["diagnosis"].value_counts().to_dict() != expected_diagnoses:
        raise AssertionError("Official TEST diagnosis distribution differs from the predefined split.")
    metadata = acoustic_test.loc[:, list(BOOKKEEPING_COLUMNS)].copy()
    transcripts = _transcribe_test(metadata, config)
    asr_linguistic, deployment, _ = build_asr_feature_tables_for_split(
        transcripts, metadata, acoustic_test, "TEST"
    )
    if len(asr_feature_columns(asr_linguistic)) != 32:
        raise AssertionError("Official TEST ASR linguistic schema must contain exactly 32 predictors.")
    _atomic_csv_write(asr_linguistic, TEST_ASR_FEATURE_PATH)
    _atomic_csv_write(deployment, TEST_DEPLOYMENT_FEATURE_PATH)

    feature_names = list(artifacts.schema["ordered_feature_names"])
    missing = [feature for feature in feature_names if feature not in deployment.columns]
    if missing:
        raise AssertionError(f"Official TEST deployment table is missing frozen features: {missing}")
    forbidden = {"age", "gender", "MMSE", "participant_id", "diagnosis", "screening_label", "Split"}
    if forbidden.intersection(feature_names) or any("pause_annotation" in feature for feature in feature_names):
        raise AssertionError("Frozen feature schema includes prohibited predictor fields.")
    features = deployment.loc[:, feature_names]
    if list(features.columns) != feature_names or len(features.columns) != 51 or features.columns.duplicated().any():
        raise AssertionError("Official TEST features do not exactly match the frozen 51-feature schema.")
    final_feature_table = pd.concat([deployment.loc[:, list(BOOKKEEPING_COLUMNS)], features], axis=1)
    _atomic_csv_write(final_feature_table, TEST_FEATURE_PATH)

    screening = pd.DataFrame(predict_screening(features, artifacts))
    labels = encode_screening_labels(deployment["screening_label"]).to_numpy()
    predictions = pd.DataFrame(
        {
            "participant_id": deployment["participant_id"],
            "true_binary_label": labels,
            "original_diagnosis": deployment["diagnosis"],
            "raw_model_score": screening["raw_model_score"],
            "screening_score": screening["cognitive_speech_screening_score"],
            "predicted_binary_label": (screening["raw_model_score"] >= threshold).astype(int),
            "predicted_screening_class": screening["classification"],
        }
    )
    if len(predictions) != EXPECTED_TEST_PARTICIPANTS or predictions["participant_id"].duplicated().any():
        raise AssertionError("Official TEST predictions must have one row per participant.")
    metrics = fixed_threshold_metrics(labels, predictions["raw_model_score"].to_numpy(), threshold)
    bootstrap = bootstrap_confidence_intervals(labels, predictions["raw_model_score"].to_numpy(), threshold)
    breakdown = diagnosis_prediction_breakdown(predictions)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _atomic_csv_write(predictions, PREDICTIONS_PATH)
    _atomic_csv_write(pd.DataFrame([metrics]), METRICS_PATH)
    _atomic_csv_write(bootstrap, BOOTSTRAP_PATH)
    _atomic_csv_write(breakdown, DIAGNOSIS_BREAKDOWN_PATH)
    _plot_confusion(labels, predictions["predicted_binary_label"].to_numpy(), CONFUSION_MATRIX_PATH)
    _plot_roc(labels, predictions["raw_model_score"].to_numpy(), ROC_PATH)
    _plot_score_distribution(predictions, SCORE_DISTRIBUTION_PATH)
    evaluation_metadata = {
        "evaluation_type": "ONE-TIME OFFICIAL HELD-OUT TEST EVALUATION",
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "participant_count": int(len(predictions)),
        "recording_count": int(len(transcripts)),
        "class_counts": {"Healthy": int((labels == 0).sum()), "Impaired": int((labels == 1).sum())},
        "diagnosis_counts": {key: int(value) for key, value in deployment["diagnosis"].value_counts().items()},
        "frozen_artifact_sha256": artifacts.manifest["artifact_sha256"],
        "frozen_model": {
            "feature_representation": artifacts.metadata["feature_representation"],
            "feature_count": len(feature_names),
            "logistic_C": artifacts.metadata["logistic_C"],
            "preprocessing": [artifacts.metadata["imputer"], artifacts.metadata["scaler"]],
            "asr_configuration": artifacts.metadata["asr_deployment_configuration"],
            "score_source": artifacts.metadata["score_type"],
            "research_threshold": threshold,
            "calibration": artifacts.metadata["calibration"],
        },
        "no_fitting_or_tuning_performed_on_test": True,
        "threshold_was_not_modified_from_test_results": True,
        "test_status": "CONSUMED FOR FINAL EVALUATION",
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "random_state": BOOTSTRAP_RANDOM_STATE,
            "purpose": "descriptive uncertainty estimation only; no retraining, tuning, or threshold selection",
        },
    }
    _atomic_json_write(evaluation_metadata, METADATA_PATH)

    print(f"\nTEST participants: {len(predictions)}")
    print(f"Healthy: {(labels == 0).sum()}")
    print(f"Impaired: {(labels == 1).sum()}")
    print(f"Recordings processed: {len(transcripts)}")
    print("\nModel:\nLogistic Regression\nC = 0.1")
    print("\nFeatures:\n51")
    print("\nASR:\nbase.en")
    print(f"\nThreshold:\n{threshold:.2f}")
    print("\nFinal TEST metrics:")
    for name in ("accuracy", "balanced_accuracy", "precision", "sensitivity", "specificity", "f1", "roc_auc", "brier_score", "log_loss"):
        print(f"{name}: {float(metrics[name]):.4f}")
    print("\nConfusion Matrix:")
    for name in ("tn", "fp", "fn", "tp"):
        print(f"{name.upper()}: {metrics[name]}")
    print("\n95% bootstrap CI:")
    print(bootstrap[["metric", "ci_lower_95", "ci_upper_95", "valid_replicates"]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nDiagnosis descriptive breakdown:")
    print(breakdown.to_string(index=False))
    print("\nNo tuning or model modification was performed using TEST results.")
    print(f"Saved final evaluation outputs: {OUTPUT_DIRECTORY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
