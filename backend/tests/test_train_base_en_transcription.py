"""Synthetic no-download tests for frozen TRAIN base.en transcription and features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml.asr.asr_linguistic_features import (
    ASR_COMPATIBLE_FEATURE_NAMES,
    TASKS,
    build_asr_feature_tables,
    extract_asr_linguistic_features,
    validate_asr_transcripts,
)
from ml.asr.whisper_transcriber import WhisperTranscriptionConfig
from ml.data.development_guard import DevelopmentDataGuardError
from ml.models.modality_comparison import PURE_ACOUSTIC_SUFFIXES
from scripts.transcribe_train_base_en import (
    ACOUSTIC_TRAIN_PATH,
    LINGUISTIC_TRAIN_PATH,
    TRANSCRIPT_COLUMNS,
    _atomic_csv_write,
    build_train_worklist,
    load_checkpoint,
    remaining_worklist,
    write_or_validate_frozen_config,
)


def metadata(participant_count: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_id": [f"participant-{index}" for index in range(participant_count)],
            "diagnosis": ["HC", "MCI"][:participant_count],
            "screening_label": ["Healthy", "Impaired"][:participant_count],
            "Split": ["TRAIN"] * participant_count,
        }
    )


def acoustic_train(metadata_frame: pd.DataFrame) -> pd.DataFrame:
    table = metadata_frame.copy()
    for task in TASKS:
        for suffix in PURE_ACOUSTIC_SUFFIXES:
            value = 60.0 if suffix == "audio_duration_seconds" else 30.0 if suffix == "voiced_duration_seconds" else 1.0
            table[f"{task.casefold()}_{suffix}"] = value
    return table


def complete_transcripts(metadata_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for participant_id in metadata_frame["participant_id"]:
        for task in TASKS:
            rows.append(
                {
                    "participant_id": participant_id,
                    "task": task,
                    "transcript_text": "um piper piper",
                    "audio_duration_seconds": 60.0,
                    "transcription_runtime_seconds": 1.0,
                    "model": "base.en",
                }
            )
    return pd.DataFrame(rows, columns=TRANSCRIPT_COLUMNS)


def test_train_worklist_has_exactly_three_unique_tasks_per_participant() -> None:
    train = metadata()
    worklist = build_train_worklist(train)
    assert len(worklist) == len(train) * 3
    assert not worklist.duplicated(["participant_id", "task"]).any()
    assert set(worklist["task"]) == set(TASKS)


def test_train_worklist_rejects_test_rows() -> None:
    train = metadata()
    train.loc[0, "Split"] = "TEST"
    with pytest.raises(DevelopmentDataGuardError, match="only Split == TRAIN"):
        build_train_worklist(train)


def test_checkpoint_resume_keeps_completed_rows_and_requires_matching_config(tmp_path: Path) -> None:
    train = metadata()
    worklist = build_train_worklist(train)
    config_path = tmp_path / "asr_config.json"
    checkpoint_path = tmp_path / "checkpoint.csv"
    config = WhisperTranscriptionConfig(model_size="base.en")
    write_or_validate_frozen_config(config_path, config)
    write_or_validate_frozen_config(config_path, config)
    with pytest.raises(RuntimeError, match="does not match"):
        write_or_validate_frozen_config(
            config_path, WhisperTranscriptionConfig(model_size="small.en")
        )

    completed = complete_transcripts(train).iloc[:1]
    _atomic_csv_write(completed, checkpoint_path)
    loaded = load_checkpoint(checkpoint_path, worklist)
    remaining = remaining_worklist(worklist, loaded)
    assert len(loaded) == 1
    assert len(remaining) == len(worklist) - 1
    assert not set(map(tuple, loaded[["participant_id", "task"]].to_records(index=False))).intersection(
        set(map(tuple, remaining[["participant_id", "task"]].to_records(index=False)))
    )


def test_checkpoint_rejects_duplicate_participant_task_rows(tmp_path: Path) -> None:
    train = metadata()
    worklist = build_train_worklist(train)
    duplicate = pd.concat([complete_transcripts(train).iloc[:1]] * 2, ignore_index=True)
    checkpoint_path = tmp_path / "duplicate.csv"
    _atomic_csv_write(duplicate, checkpoint_path)
    with pytest.raises(ValueError, match="duplicate"):
        load_checkpoint(checkpoint_path, worklist)


def test_asr_features_are_pause_annotation_free_and_include_pft_p_initial_features() -> None:
    sft = extract_asr_linguistic_features("um apple apple", "SFT")
    pft = extract_asr_linguistic_features("piper pear apple", "PFT")
    assert set(sft) == set(ASR_COMPATIBLE_FEATURE_NAMES)
    assert not any("pause" in name for name in sft)
    assert pft["p_initial_word_count"] == 2
    assert pft["p_initial_ratio"] == pytest.approx(2 / 3)


def test_asr_feature_pivot_is_one_row_per_train_participant_and_uses_asr_word_count_for_rates() -> None:
    train = metadata()
    transcripts = complete_transcripts(train)
    asr_linguistic, deployment, per_recording = build_asr_feature_tables(
        transcripts, train, acoustic_train(train)
    )
    assert len(asr_linguistic) == len(deployment) == len(train)
    assert asr_linguistic["participant_id"].is_unique
    assert deployment["participant_id"].is_unique
    assert deployment["Split"].eq("TRAIN").all()
    # Three ASR tokens over a 60-second recording gives 3 WPM, not a manual count.
    assert deployment.loc[0, "sft_recording_word_rate_wpm"] == pytest.approx(3.0)
    assert deployment.loc[0, "sft_articulation_rate_wpm"] == pytest.approx(6.0)
    assert len(per_recording) == len(train) * 3


def test_asr_transcript_validation_requires_complete_unique_task_set() -> None:
    train = metadata()
    incomplete = complete_transcripts(train).iloc[:-1]
    with pytest.raises(ValueError, match="exactly one"):
        validate_asr_transcripts(incomplete, train["participant_id"])


def test_script_paths_are_train_only_and_never_locked_test_artifacts() -> None:
    assert LINGUISTIC_TRAIN_PATH.name == "linguistic_features_train.csv"
    assert ACOUSTIC_TRAIN_PATH.name == "acoustic_features_train.csv"
    assert "LOCKED" not in str(LINGUISTIC_TRAIN_PATH)
    assert "LOCKED" not in str(ACOUSTIC_TRAIN_PATH)
