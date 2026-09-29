"""Synthetic tests for full acoustic participant-level assembly and checkpoints."""

from __future__ import annotations

import pandas as pd
import pytest

from ml.audio.acoustic_features import ACOUSTIC_FEATURE_NAMES, empty_acoustic_features
from scripts import build_acoustic_features as builder


def synthetic_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_id": ["person-1", "person-2"],
            "diagnosis": ["HC", "MCI"],
            "screening_label": ["Healthy", "Impaired"],
            "Split": ["TRAIN", "TEST"],
        }
    )


def synthetic_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for participant in synthetic_metadata().itertuples(index=False):
        series = pd.Series(participant._asdict())
        for task in builder.TASKS:
            record = builder.recording_row(series, task)
            record.update(empty_acoustic_features())
            record.update(
                {
                    "audio_duration_seconds": 1.0,
                    "voiced_duration_seconds": 0.75,
                    "silence_duration_seconds": 0.25,
                    "silence_ratio": 0.25,
                    "pause_count": 0.0,
                    "pause_total_seconds": 0.0,
                    "recording_word_rate_wpm": 60.0,
                    "articulation_rate_wpm": 80.0,
                    "_status": "success",
                    "_sample_rate": 16000,
                    "_channel_count": 1,
                    "_pitch_failed": False,
                    "_pause_failed": False,
                }
            )
            records.append(record)
    return records


def test_pivot_creates_one_row_per_participant_with_task_prefixes_and_preserved_labels() -> None:
    metadata = synthetic_metadata()
    pivoted = builder.pivot_participant_features(metadata, synthetic_records())
    assert len(pivoted) == 2
    assert len([column for column in pivoted if column not in builder.BOOKKEEPING_COLUMNS]) == 3 * len(
        ACOUSTIC_FEATURE_NAMES
    )
    assert "sft_audio_duration_seconds" in pivoted
    assert "pft_mfcc_13_std" in pivoted
    assert "ctd_articulation_rate_wpm" in pivoted
    assert pivoted["Split"].tolist() == ["TRAIN", "TEST"]
    assert pivoted["screening_label"].tolist() == ["Healthy", "Impaired"]


def test_pivot_rejects_duplicate_participant_task_records() -> None:
    records = synthetic_records()
    with pytest.raises(ValueError, match="Duplicate participant-task"):
        builder.pivot_participant_features(synthetic_metadata(), records + [records[0]])


def test_sanity_constraints_detect_only_present_invalid_values() -> None:
    pivoted = builder.pivot_participant_features(synthetic_metadata(), synthetic_records())
    assert not any(builder.constraint_violations(pivoted).values())
    pivoted.loc[0, "sft_silence_ratio"] = 1.1
    assert builder.constraint_violations(pivoted)["silence_ratio_outside_0_1"] == 1


def test_checkpoint_resume_requires_matching_frozen_configuration(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(builder, "CHECKPOINT_PATH", tmp_path / "checkpoint.csv")
    monkeypatch.setattr(builder, "CHECKPOINT_CONFIG_PATH", tmp_path / "checkpoint_config.json")
    configuration = builder.frozen_configuration_payload()
    fingerprint = builder.configuration_fingerprint(configuration)
    records = synthetic_records()
    builder.write_checkpoint(records, fingerprint)
    assert len(builder.load_checkpoint_records(fingerprint)) == len(records)
    with pytest.raises(RuntimeError, match="configuration differs"):
        builder.load_checkpoint_records("different-fingerprint")
