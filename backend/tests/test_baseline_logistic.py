from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.baseline_logistic import (
    build_logistic_pipeline,
    encode_screening_labels,
    run_cross_validated_experiment,
    select_feature_columns,
    training_rows,
)


def synthetic_feature_table() -> pd.DataFrame:
    labels = ["Healthy", "Impaired"] * 5
    train_rows = pd.DataFrame(
        {
            "participant_id": [f"id-{index}" for index in range(10)],
            "diagnosis": ["HC" if label == "Healthy" else "MCI" for label in labels],
            "screening_label": labels,
            "Split": ["TRAIN"] * 10,
            "sft_word_count": [10, 20] * 5,
            "pft_word_count": [5, 15] * 5,
            "ctd_word_count": [30, 60] * 5,
            "age": [70] * 10,
            "MMSE": [np.nan] * 10,
        }
    )
    test_rows = train_rows.iloc[:2].copy()
    test_rows["Split"] = "TEST"
    return pd.concat([train_rows, test_rows], ignore_index=True)


def test_training_rows_excludes_test_participants() -> None:
    train = training_rows(synthetic_feature_table())

    assert len(train) == 10
    assert set(train["Split"]) == {"TRAIN"}


def test_label_encoding_is_correct() -> None:
    encoded = encode_screening_labels(pd.Series(["Healthy", "Impaired"]))

    assert encoded.tolist() == [0, 1]


def test_task_feature_selection_excludes_metadata_columns() -> None:
    table = synthetic_feature_table()

    assert select_feature_columns(table, "SFT") == ["sft_word_count"]
    assert select_feature_columns(table, "PFT") == ["pft_word_count"]
    assert select_feature_columns(table, "CTD") == ["ctd_word_count"]
    assert select_feature_columns(table, "ALL") == [
        "sft_word_count",
        "pft_word_count",
        "ctd_word_count",
    ]


def test_pipeline_keeps_preprocessing_inside_sklearn_pipeline_and_handles_missing_values() -> None:
    pipeline = build_logistic_pipeline()

    assert isinstance(pipeline.named_steps["imputer"], SimpleImputer)
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    assert isinstance(pipeline.named_steps["classifier"], LogisticRegression)
    pipeline.fit(pd.DataFrame({"value": [1.0, np.nan, 3.0, 4.0]}), [0, 0, 1, 1])
    assert len(pipeline.predict(pd.DataFrame({"value": [np.nan, 2.0]}))) == 2


def test_five_train_only_cv_folds_are_produced() -> None:
    table = synthetic_feature_table()
    result = run_cross_validated_experiment(table, "ALL")

    assert len(result.cv_metrics) == 5
    assert len(result.oof_predictions) == 10
    assert set(result.oof_predictions["anonymized_participant_id"]) == {
        f"train_{index:03d}" for index in range(1, 11)
    }
    assert len(list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(
        table.iloc[:10][["sft_word_count"]], [0, 1] * 5
    ))) == 5
