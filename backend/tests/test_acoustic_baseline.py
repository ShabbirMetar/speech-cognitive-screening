"""Tests for the TRAIN-only interpretable acoustic nested-CV baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.data.development_guard import DevelopmentDataGuardError
from ml.models.acoustic_baseline import (
    INTERPRETABLE_ACOUSTIC_SUFFIXES,
    build_acoustic_pipeline,
    build_inner_cv,
    build_outer_cv,
    run_nested_acoustic_experiment,
    select_acoustic_feature_columns,
)
from scripts.train_acoustic_baseline import FEATURE_TABLE_PATH


def synthetic_acoustic_table(rows: int = 40, split: str = "TRAIN") -> pd.DataFrame:
    """Create balanced synthetic rows with the frozen 45-feature schema."""

    labels = np.array(["Healthy", "Impaired"] * (rows // 2))
    table = pd.DataFrame(
        {
            "participant_id": [f"synthetic-{index:03d}" for index in range(rows)],
            "diagnosis": np.where(labels == "Healthy", "HC", "MCI"),
            "screening_label": labels,
            "Split": [split] * rows,
            "age": 70,
            "gender": "X",
            "MMSE": 25,
            "sft_mfcc_01_mean": np.arange(rows, dtype=float),
        }
    )
    for task_index, task in enumerate(("sft", "pft", "ctd"), start=1):
        for suffix_index, suffix in enumerate(INTERPRETABLE_ACOUSTIC_SUFFIXES, start=1):
            table[f"{task}_{suffix}"] = (
                np.arange(rows, dtype=float) + task_index + suffix_index
            )
    table.loc[0, "sft_f0_mean_hz"] = np.nan
    return table


def test_prespecified_feature_selection_uses_exactly_15_per_task_and_45_overall() -> None:
    table = synthetic_acoustic_table()
    assert len(select_acoustic_feature_columns(table, "SFT")) == 15
    assert len(select_acoustic_feature_columns(table, "PFT")) == 15
    assert len(select_acoustic_feature_columns(table, "CTD")) == 15
    all_features = select_acoustic_feature_columns(table, "ALL")
    assert len(all_features) == 45
    assert "sft_mfcc_01_mean" not in all_features
    assert not any("spectral" in column or "zero_crossing" in column for column in all_features)


def test_each_acoustic_pipeline_keeps_imputation_and_scaling_inside_pipeline() -> None:
    for model in ("logistic", "linear_svm", "rbf_svm"):
        pipeline = build_acoustic_pipeline(model)
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.named_steps["imputer"], SimpleImputer)
        assert isinstance(pipeline.named_steps["scaler"], StandardScaler)


def test_frozen_outer_and_inner_cv_counts_are_used() -> None:
    assert build_outer_cv().n_splits == 5
    assert build_inner_cv().n_splits == 3


def test_acoustic_model_rejects_test_rows_before_any_fit() -> None:
    with pytest.raises(DevelopmentDataGuardError, match="only Split == TRAIN"):
        run_nested_acoustic_experiment(synthetic_acoustic_table(split="TEST"), "SFT", "logistic")


def test_train_rows_receive_one_outer_oof_prediction() -> None:
    table = synthetic_acoustic_table()
    result = run_nested_acoustic_experiment(table, "SFT", "logistic", search_n_jobs=1)
    assert len(result.oof_predictions) == len(table)
    assert result.oof_predictions["participant_id"].is_unique
    assert set(result.oof_predictions["fold"]) == {1, 2, 3, 4, 5}


def test_script_is_pinned_to_the_train_acoustic_artifact() -> None:
    assert FEATURE_TABLE_PATH.name == "acoustic_features_train.csv"
    assert "LOCKED" not in str(FEATURE_TABLE_PATH)
