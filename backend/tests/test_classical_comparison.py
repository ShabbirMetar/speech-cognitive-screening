"""Synthetic tests for TRAIN-only nested classical-model comparison helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from ml.models.baseline_logistic import encode_screening_labels, select_feature_columns, training_rows
from ml.models.classical_comparison import (
    INNER_CV_SPLITS,
    OUTER_CV_SPLITS,
    build_inner_cv,
    build_inner_search,
    build_model_pipeline,
    build_outer_cv,
    parameter_grid,
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


def test_logistic_grid_preserves_c_and_solver_without_deprecated_penalty_setting() -> None:
    pipeline = build_model_pipeline("logistic")
    grid = parameter_grid("logistic")
    assert pipeline.named_steps["classifier"].max_iter == 2000
    assert pipeline.named_steps["classifier"].random_state == 42
    assert grid == {
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__solver": ["liblinear"],
    }
    assert "classifier__penalty" not in grid


def test_outer_and_inner_split_counts() -> None:
    assert build_outer_cv().n_splits == OUTER_CV_SPLITS == 5
    assert build_inner_cv().n_splits == INNER_CV_SPLITS == 3


def test_compact_search_strategy_uses_grid_for_linear_models_and_random_search_for_trees() -> None:
    for model in ("logistic", "linear_svm", "rbf_svm"):
        search = build_inner_search(model)
        assert isinstance(search, GridSearchCV)
        assert search.scoring == "f1"
        assert search.n_jobs == 2
    for model in ("random_forest", "xgboost"):
        search = build_inner_search(model)
        assert isinstance(search, RandomizedSearchCV)
        assert search.n_iter == 8
        assert search.random_state == 42
        assert search.n_jobs == 2


def test_optimized_search_spaces_remain_compact() -> None:
    assert parameter_grid("linear_svm") == {"classifier__C": [0.1, 1.0, 10.0]}
    assert parameter_grid("rbf_svm") == {
        "classifier__C": [0.5, 1.0, 5.0],
        "classifier__gamma": ["scale", 0.01],
    }
    assert parameter_grid("random_forest") == {
        "classifier__n_estimators": [200, 300, 500],
        "classifier__max_depth": [None, 4, 8],
        "classifier__min_samples_leaf": [1, 3, 5],
        "classifier__max_features": ["sqrt", 0.5],
    }
    assert parameter_grid("xgboost") == {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [2, 3, 4],
        "classifier__learning_rate": [0.03, 0.05, 0.1],
        "classifier__subsample": [0.8, 1.0],
        "classifier__colsample_bytree": [0.8, 1.0],
    }


def test_every_train_row_gets_one_outer_oof_prediction_and_no_test_rows_do() -> None:
    table = synthetic_feature_table()
    result = run_nested_cv_model(table, "logistic", search_n_jobs=1)
    train_identifiers = set(table.loc[table["Split"].eq("TRAIN"), "participant_id"])
    test_identifiers = set(table.loc[table["Split"].eq("TEST"), "participant_id"])
    assert len(result.oof_predictions) == len(train_identifiers)
    assert result.oof_predictions["participant_id"].is_unique
    assert set(result.oof_predictions["participant_id"]) == train_identifiers
    assert not set(result.oof_predictions["participant_id"]).intersection(test_identifiers)
    assert set(result.oof_predictions["fold"]) == {1, 2, 3, 4, 5}
