"""Synthetic no-download tests for the local faster-whisper pilot helpers."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from ml.data.development_guard import DevelopmentDataGuardError
from ml.asr.whisper_transcriber import WhisperTranscriptionConfig
from scripts.pilot_whisper_asr import (
    TRANSCRIPT_OUTPUT_COLUMNS,
    feature_stability_rows,
    normalise_for_wer,
    output_frame,
    select_pilot_participants,
    summarise_feature_stability,
    word_error_rate,
)


def synthetic_train_table() -> pd.DataFrame:
    rows = []
    for label, diagnosis in (("Healthy", "HC"), ("Impaired", "MCI")):
        for index in range(5):
            rows.append(
                {
                    "participant_id": f"{label.lower()}-{index}",
                    "diagnosis": diagnosis,
                    "screening_label": label,
                    "Split": "TRAIN",
                }
            )
    return pd.DataFrame(rows)


def test_wer_normalization_removes_labels_pause_annotations_and_punctuation() -> None:
    assert normalise_for_wer("Pat: Hello, um (2 seconds) world!") == "hello um world"
    assert normalise_for_wer("hello um world") == "hello um world"
    assert word_error_rate("Pat: hello, world!", "hello world") == 0.0
    assert word_error_rate("one two", "one three") == 0.5


def test_wer_handles_empty_reference_without_inventing_a_score() -> None:
    assert word_error_rate("", "") == 0.0
    assert math.isnan(word_error_rate("", "word"))


def test_feature_stability_uses_existing_linguistic_features_and_excludes_pause_features() -> None:
    rows = feature_stability_rows("base.en", "um cat", "Pat: um cat cat (2 seconds)")
    features = {str(row["feature"]) for row in rows}
    by_feature = {str(row["feature"]): row for row in rows}
    assert "pause_annotation_count" not in features
    assert by_feature["word_count"]["absolute_difference"] == 1.0
    assert by_feature["filler_count"]["absolute_difference"] == 0.0


def test_feature_stability_summary_reports_mean_absolute_difference() -> None:
    rows = [
        {"model": "base.en", "feature": "word_count", "absolute_difference": 1.0},
        {"model": "base.en", "feature": "word_count", "absolute_difference": 3.0},
    ]
    summary = summarise_feature_stability(rows)
    assert summary.loc[0, "comparison_count"] == 2
    assert summary.loc[0, "mean_absolute_difference"] == 2.0


def test_pilot_selection_is_train_only_and_balanced() -> None:
    selected = select_pilot_participants(synthetic_train_table())
    assert len(selected) == 10
    assert selected["Split"].eq("TRAIN").all()
    assert selected["screening_label"].value_counts().to_dict() == {"Healthy": 5, "Impaired": 5}


def test_pilot_selection_rejects_test_rows() -> None:
    table = synthetic_train_table()
    table.loc[0, "Split"] = "TEST"
    with pytest.raises(DevelopmentDataGuardError, match="only Split == TRAIN"):
        select_pilot_participants(table)


def test_transcript_output_schema_excludes_diagnosis() -> None:
    frame = output_frame(
        [
            {
                "participant_id": "synthetic-1",
                "task": "SFT",
                "model": "base.en",
                "transcript_text": "synthetic text",
                "detected_language": "en",
                "audio_duration_seconds": 1.0,
                "transcription_runtime_seconds": 0.5,
                "success": True,
                "diagnosis": "HC",
            }
        ]
    )
    assert tuple(frame.columns) == TRANSCRIPT_OUTPUT_COLUMNS
    assert "diagnosis" not in frame.columns


def test_cpu_safe_config_defaults_do_not_construct_or_download_a_model() -> None:
    config = WhisperTranscriptionConfig(model_size="base.en")
    assert config.device == "cpu"
    assert config.compute_type == "int8"
    assert config.language == "en"
