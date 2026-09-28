"""Synthetic tests for TRAIN-only nested classical-model comparison helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from ml.models.baseline_logistic import encode_screening_labels, select_feature_columns, training_rows
from ml.models.classical_comparison import (
    INNER_CV_SPLITS,
    OUTER_CV_SPLITS,
    build_inner_cv,
    build_model_pipeline,
    build_outer_cv,
    run_nested_cv_model,
)


def synthetic_feature_table(train_rows: int = 40, test_rows: int = 10) -> pd.DataFrame:
    """Create balanced synthetic rows with deliberately missing linguistic values."""

    total_rows = train_rows + test_rows
    labels = np.array(["Healthy", "Impaired"] * (total_rows // 2))
    table = pd.DataFrame(
        {
            "participant_id": [f"synthetic-{index:03d}" for index in range(total_rows)],
            "diagnosis": np.where(labels == "Healthy", "HC", "MCI"),
            "screening_label": labels,
            "Split": ["TRAIN"] * train_rows + ["TEST"] * test_rows,
            "age": 70,
            "gender": "X",
            "MMSE": 25,
            "sft_word_count": np.arange(total_rows, dtype=float),
            "pft_word_count": np.where(np.arange(total_rows) == 1, np.nan, np.arange(total_rows)),
            "ctd_word_count": np.arange(total_rows, dtype=float) + 1,
        }
    )
    return table


def test_training_rows_excludes_every_test_row() -> None:
    table = synthetic_feature_table()
    train = training_rows(table)
    assert len(train) == 40
    assert train["Split"].eq("TRAIN").all()
    assert not set(table.loc[table["Split"].eq("TEST"), "participant_id"]).intersection(
        set(train["participant_id"])
    )


def test_label_encoding_and_metadata_exclusion() -> None:
    table = synthetic_feature_table()
    assert encode_screening_labels(pd.Series(["Healthy", "Impaired"])).tolist() == [0, 1]
    selected = select_feature_columns(table, "ALL")
    assert selected == ["sft_word_count", "pft_word_count", "ctd_word_count"]
    assert not {"participant_id", "age", "gender", "MMSE", "diagnosis", "Split"}.intersection(selected)


def test_pipelines_keep_preprocessing_inside_model_pipelines() -> None:
    for model in ("logistic", "linear_svm", "rbf_svm"):
        pipeline = build_model_pipeline(model)
        assert isinstance(pipeline.named_steps["imputer"], SimpleImputer)
        assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    for model in ("random_forest", "xgboost"):
        pipeline = build_model_pipeline(model)
        assert isinstance(pipeline.named_steps["imputer"], SimpleImputer)
        assert "scaler" not in pipeline.named_steps


def test_outer_and_inner_split_counts() -> None:
    assert build_outer_cv().n_splits == OUTER_CV_SPLITS == 5
    assert build_inner_cv().n_splits == INNER_CV_SPLITS == 4


def test_every_train_row_gets_one_outer_oof_prediction_and_no_test_rows_do() -> None:
    table = synthetic_feature_table()
    result = run_nested_cv_model(table, "logistic", grid_n_jobs=1)
    train_identifiers = set(table.loc[table["Split"].eq("TRAIN"), "participant_id"])
    test_identifiers = set(table.loc[table["Split"].eq("TEST"), "participant_id"])
    assert len(result.oof_predictions) == len(train_identifiers)
    assert result.oof_predictions["participant_id"].is_unique
    assert set(result.oof_predictions["participant_id"]) == train_identifiers
    assert not set(result.oof_predictions["participant_id"]).intersection(test_identifiers)
    assert set(result.oof_predictions["fold"]) == {1, 2, 3, 4, 5}
