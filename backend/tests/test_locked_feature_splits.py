"""Integrity tests for the derived TRAIN and locked TEST feature artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml.data.development_guard import DevelopmentDataGuardError
from ml.models.baseline_logistic import run_all_experiments
from scripts.split_and_lock_features import partition_feature_table


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIRECTORY = PROJECT_ROOT / "artifacts" / "features"


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(FEATURE_DIRECTORY / name)


def test_locked_feature_artifacts_have_expected_row_counts_and_no_id_overlap() -> None:
    linguistic_train = _read("linguistic_features_train.csv")
    linguistic_test = _read("linguistic_features_test_LOCKED.csv")
    acoustic_train = _read("acoustic_features_train.csv")
    acoustic_test = _read("acoustic_features_test_LOCKED.csv")

    assert len(linguistic_train) == 320
    assert len(linguistic_test) == 80
    assert len(acoustic_train) == 320
    assert len(acoustic_test) == 80
    assert linguistic_train["Split"].eq("TRAIN").all()
    assert acoustic_train["Split"].eq("TRAIN").all()
    assert linguistic_test["Split"].eq("TEST").all()
    assert acoustic_test["Split"].eq("TEST").all()
    assert not set(linguistic_train["participant_id"]).intersection(linguistic_test["participant_id"])
    assert not set(acoustic_train["participant_id"]).intersection(acoustic_test["participant_id"])
    assert set(linguistic_train["participant_id"]) == set(acoustic_train["participant_id"])
    assert set(linguistic_test["participant_id"]) == set(acoustic_test["participant_id"])


def test_locked_artifact_schemas_match_their_full_source_tables() -> None:
    for source_name, train_name, test_name in (
        (
            "manual_transcript_features.csv",
            "linguistic_features_train.csv",
            "linguistic_features_test_LOCKED.csv",
        ),
        (
            "acoustic_features.csv",
            "acoustic_features_train.csv",
            "acoustic_features_test_LOCKED.csv",
        ),
    ):
        source_columns = list(_read(source_name).columns)
        assert list(_read(train_name).columns) == source_columns
        assert list(_read(test_name).columns) == source_columns


def test_partition_preserves_source_schema() -> None:
    source = pd.DataFrame(
        {
            "participant_id": ["train-1", "test-1"],
            "screening_label": ["Healthy", "Impaired"],
            "Split": ["TRAIN", "TEST"],
            "sft_word_count": [10.0, 12.0],
        }
    )
    partition = partition_feature_table(source, "synthetic")
    assert list(partition.train.columns) == list(source.columns)
    assert list(partition.test_locked.columns) == list(source.columns)


def test_model_utility_rejects_locked_test_input_before_cross_validation() -> None:
    locked_test = _read("linguistic_features_test_LOCKED.csv")
    with pytest.raises(DevelopmentDataGuardError, match="only Split == TRAIN"):
        run_all_experiments(locked_test)
