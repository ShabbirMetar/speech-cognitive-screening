"""Tests for TRAIN-only pure-acoustic and multimodal-fusion comparison helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.models.modality_comparison import (
    PURE_ACOUSTIC_SUFFIXES,
    build_outer_splits,
    linguistic_feature_columns,
    merge_train_feature_tables,
    run_nested_logistic_representation,
    select_pure_acoustic_feature_columns,
)
from scripts.compare_modalities import ACOUSTIC_TRAIN_PATH, LINGUISTIC_TRAIN_PATH


def synthetic_train_tables(rows: int = 40) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = np.array(["Healthy", "Impaired"] * (rows // 2))
    metadata = {
        "participant_id": [f"participant-{index:03d}" for index in range(rows)],
        "diagnosis": np.where(labels == "Healthy", "HC", "MCI"),
        "screening_label": labels,
        "Split": ["TRAIN"] * rows,
    }
    linguistic = pd.DataFrame(
        {
            **metadata,
            "sft_word_count": np.arange(rows, dtype=float),
            "pft_word_count": np.arange(rows, dtype=float) + 1,
            "ctd_word_count": np.arange(rows, dtype=float) + 2,
        }
    )
    acoustic = pd.DataFrame(metadata)
    for task_index, task in enumerate(("sft", "pft", "ctd"), start=1):
        for suffix_index, suffix in enumerate(PURE_ACOUSTIC_SUFFIXES, start=1):
            acoustic[f"{task}_{suffix}"] = np.arange(rows, dtype=float) + task_index + suffix_index
        acoustic[f"{task}_recording_word_rate_wpm"] = np.arange(rows, dtype=float)
        acoustic[f"{task}_articulation_rate_wpm"] = np.arange(rows, dtype=float)
        acoustic[f"{task}_mfcc_01_mean"] = np.arange(rows, dtype=float)
    return linguistic, acoustic


def test_pure_acoustic_selector_has_13_per_task_and_39_overall_without_word_rates() -> None:
    linguistic, acoustic = synthetic_train_tables()
    merged = merge_train_feature_tables(linguistic, acoustic)
    assert len(select_pure_acoustic_feature_columns(merged, "SFT")) == 13
    assert len(select_pure_acoustic_feature_columns(merged, "PFT")) == 13
    assert len(select_pure_acoustic_feature_columns(merged, "CTD")) == 13
    all_features = select_pure_acoustic_feature_columns(merged, "ALL")
    assert len(all_features) == 39
    assert not any("word_rate" in feature or "mfcc" in feature for feature in all_features)


def test_linguistic_schema_is_selected_before_acoustic_columns_are_merged() -> None:
    linguistic, acoustic = synthetic_train_tables()
    merged = merge_train_feature_tables(linguistic, acoustic)
    assert linguistic_feature_columns(linguistic) == [
        "sft_word_count",
        "pft_word_count",
        "ctd_word_count",
    ]
    assert len(linguistic_feature_columns(linguistic)) < len(
        [column for column in merged.columns if column.startswith(("sft_", "pft_", "ctd_"))]
    )


def test_train_merge_is_one_to_one_and_preserves_training_split() -> None:
    linguistic, acoustic = synthetic_train_tables()
    merged = merge_train_feature_tables(linguistic, acoustic)
    assert len(merged) == 40
    assert merged["participant_id"].is_unique
    assert merged["Split"].eq("TRAIN").all()


def test_train_merge_rejects_metadata_label_disagreement() -> None:
    linguistic, acoustic = synthetic_train_tables()
    acoustic.loc[0, "screening_label"] = "Impaired"
    with pytest.raises(ValueError, match="screening_label"):
        merge_train_feature_tables(linguistic, acoustic)


def test_shared_outer_splits_are_fixed_and_complete() -> None:
    target = np.array([0, 1] * 20)
    first = build_outer_splits(target)
    second = build_outer_splits(target)
    assert len(first) == len(second) == 5
    for (first_fit, first_validation), (second_fit, second_validation) in zip(first, second, strict=True):
        assert np.array_equal(first_fit, second_fit)
        assert np.array_equal(first_validation, second_validation)
    assert sorted(np.concatenate([validation for _, validation in first]).tolist()) == list(range(40))


def test_representation_rejects_metadata_as_a_predictive_feature() -> None:
    linguistic, acoustic = synthetic_train_tables()
    merged = merge_train_feature_tables(linguistic, acoustic)
    target = np.array([0, 1] * 20)
    with pytest.raises(ValueError, match="Metadata/label columns"):
        run_nested_logistic_representation(
            merged,
            "invalid",
            ["participant_id"],
            build_outer_splits(target),
            search_n_jobs=1,
        )


def test_compare_modalities_script_is_pinned_to_train_artifacts() -> None:
    assert LINGUISTIC_TRAIN_PATH.name == "linguistic_features_train.csv"
    assert ACOUSTIC_TRAIN_PATH.name == "acoustic_features_train.csv"
    assert "LOCKED" not in str(LINGUISTIC_TRAIN_PATH)
    assert "LOCKED" not in str(ACOUSTIC_TRAIN_PATH)
