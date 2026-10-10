"""Synthetic safeguards for one-time frozen-pipeline TEST evaluation helpers."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.asr.asr_linguistic_features import TASKS, build_asr_feature_tables_for_split
from ml.models.modality_comparison import PURE_ACOUSTIC_SUFFIXES
from scripts.evaluate_official_test import (
    BOOTSTRAP_RANDOM_STATE,
    FROZEN_THRESHOLD,
    assert_official_test_not_already_recorded,
    bootstrap_confidence_intervals,
    claim_official_evaluation_lock,
    diagnosis_prediction_breakdown,
    fixed_threshold_metrics,
)
from scripts.transcribe_train_base_en import build_split_worklist


def metadata_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_id": ["test-1", "test-2"],
            "diagnosis": ["HC", "MCI"],
            "screening_label": ["Healthy", "Impaired"],
            "Split": ["TEST", "TEST"],
        }
    )


def acoustic_frame(metadata: pd.DataFrame) -> pd.DataFrame:
    table = metadata.copy()
    for task in TASKS:
        for suffix in PURE_ACOUSTIC_SUFFIXES:
            table[f"{task.casefold()}_{suffix}"] = 60.0 if suffix == "audio_duration_seconds" else 30.0
    return table


def transcript_frame(metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for participant_id in metadata["participant_id"]:
        for task in TASKS:
            rows.append(
                {
                    "participant_id": participant_id,
                    "task": task,
                    "transcript_text": "piper pear",
                    "audio_duration_seconds": 60.0,
                    "transcription_runtime_seconds": 1.0,
                    "model": "base.en",
                }
            )
    return pd.DataFrame(rows)


def test_test_split_reuses_the_same_asr_feature_formulas_and_asr_rates() -> None:
    metadata = metadata_frame()
    worklist = build_split_worklist(metadata, "TEST")
    asr_linguistic, deployment, _ = build_asr_feature_tables_for_split(
        transcript_frame(metadata), metadata, acoustic_frame(metadata), "TEST"
    )
    assert len(worklist) == 6
    assert len(asr_linguistic) == len(deployment) == 2
    assert deployment["Split"].eq("TEST").all()
    assert deployment.loc[0, "sft_recording_word_rate_wpm"] == pytest.approx(2.0)
    assert deployment.loc[0, "sft_articulation_rate_wpm"] == pytest.approx(4.0)


def test_fixed_metrics_use_the_frozen_threshold_and_binary_confusion_order() -> None:
    metrics = fixed_threshold_metrics(
        np.array([0, 0, 1, 1]), np.array([0.10, 0.53, 0.70, 0.20]), FROZEN_THRESHOLD
    )
    assert metrics["tn"] == metrics["fp"] == metrics["fn"] == metrics["tp"] == 1
    assert metrics["specificity"] == pytest.approx(0.5)
    assert metrics["sensitivity"] == pytest.approx(0.5)


def test_bootstrap_is_deterministic_and_never_fits_a_model() -> None:
    target = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.6, 0.4, 0.8, 0.9])
    first = bootstrap_confidence_intervals(target, scores, replicates=50, random_state=BOOTSTRAP_RANDOM_STATE)
    second = bootstrap_confidence_intervals(target, scores, replicates=50, random_state=BOOTSTRAP_RANDOM_STATE)
    pd.testing.assert_frame_equal(first, second)
    assert set(first["metric"]) == {"accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "roc_auc"}
    assert "fit(" not in inspect.getsource(bootstrap_confidence_intervals)


def test_diagnosis_breakdown_is_descriptive_binary_prediction_counting() -> None:
    predictions = pd.DataFrame(
        {
            "original_diagnosis": ["HC", "HC", "MCI", "Dementia"],
            "predicted_binary_label": [0, 1, 1, 0],
        }
    )
    table = diagnosis_prediction_breakdown(predictions).set_index("original_diagnosis")
    assert table.loc["HC", "predicted_healthy_like"] == 1
    assert table.loc["HC", "predicted_possible_impairment_like"] == 1
    assert table.loc["MCI", "participant_count"] == 1
    assert table.loc["Dementia", "participant_count"] == 1


def test_official_evaluation_lock_is_atomic_and_preserves_audit_record(tmp_path: Path) -> None:
    lock_path = tmp_path / "official_test.lock"
    claim_official_evaluation_lock(lock_path)
    record = json.loads(lock_path.read_text(encoding="utf-8"))
    assert record["purpose"] == "one-time official held-out TEST evaluation"
    with pytest.raises(RuntimeError, match="lock already exists"):
        claim_official_evaluation_lock(lock_path)


def test_consumed_evaluation_guard_refuses_an_existing_metadata_record(tmp_path: Path) -> None:
    metadata_path = tmp_path / "evaluation_metadata.json"
    metadata_path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="already recorded"):
        assert_official_test_not_already_recorded(metadata_path)


def test_persisted_authoritative_bundle_reproduces_metrics_without_inference() -> None:
    result_directory = Path(__file__).resolve().parents[2] / "artifacts" / "results" / "official_test_evaluation"
    required = {
        "test_predictions.csv",
        "final_metrics.csv",
        "bootstrap_confidence_intervals.csv",
        "diagnosis_prediction_breakdown.csv",
        "evaluation_metadata.json",
        "confusion_matrix.png",
        "roc_curve.png",
        "test_score_distribution.png",
    }
    if not result_directory.is_dir():
        pytest.skip("Official TEST bundle is unavailable in this checkout.")
    assert all((result_directory / name).is_file() and (result_directory / name).stat().st_size > 0 for name in required)

    predictions = pd.read_csv(result_directory / "test_predictions.csv")
    assert len(predictions) == predictions["participant_id"].nunique() == 80
    assert ((predictions["raw_model_score"] >= FROZEN_THRESHOLD).astype(int) == predictions["predicted_binary_label"]).all()
    recalculated = fixed_threshold_metrics(
        predictions["true_binary_label"].to_numpy(), predictions["raw_model_score"].to_numpy(), FROZEN_THRESHOLD
    )
    saved = pd.read_csv(result_directory / "final_metrics.csv").iloc[0]
    for name, value in recalculated.items():
        assert float(saved[name]) == pytest.approx(value)

    metadata = json.loads((result_directory / "evaluation_metadata.json").read_text(encoding="utf-8"))
    assert metadata["test_status"] == "CONSUMED FOR FINAL EVALUATION"
    assert metadata["no_fitting_or_tuning_performed_on_test"] is True
    assert metadata["threshold_was_not_modified_from_test_results"] is True


def test_documentation_records_the_authoritative_persisted_result() -> None:
    project_root = Path(__file__).resolve().parents[2]
    for document in (project_root / "README.md", project_root / "docs" / "CODEX_HANDOFF.md", project_root / "docs" / "OFFICIAL_TEST_EVALUATION.md"):
        text = document.read_text(encoding="utf-8")
        assert "consumed" in text.casefold()
        assert "0.7949" in text
        assert "0.8250" in text
        assert "0.8000" in text
