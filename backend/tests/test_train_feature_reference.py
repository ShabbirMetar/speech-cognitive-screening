"""Checks for the aggregate-only TRAIN visualization reference.

These tests read only the existing TRAIN deployment feature table and never
load official TEST inputs, predictions, or evaluation outputs.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.models.explainability import model_direction
from ml.models.frozen_pipeline import DEFAULT_MODEL_DIRECTORY, load_frozen_artifacts, predict_screening
from ml.models.train_feature_reference import (
    DEFAULT_TRAIN_FEATURE_REFERENCE_PATH,
    load_train_feature_reference,
    validate_reference_model_alignment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT_TRAIN_PATH = PROJECT_ROOT / "artifacts" / "features" / "deployment_features_train.csv"


pytestmark = pytest.mark.skipif(
    not DEFAULT_TRAIN_FEATURE_REFERENCE_PATH.is_file(),
    reason="Run scripts/build_train_feature_reference.py before validating the local TRAIN reference.",
)


@pytest.fixture(scope="module")
def artifacts():
    return load_frozen_artifacts()


@pytest.fixture(scope="module")
def reference(artifacts):
    return load_train_feature_reference(artifacts.schema)


def test_reference_is_aggregate_only_and_uses_exactly_the_train_cohort(reference) -> None:
    source = reference["source"]
    assert source["split"] == "TRAIN"
    assert source["participant_count"] == 320
    assert source["healthy_participant_count"] == 160
    assert source["impaired_participant_count"] == 160
    assert source["official_test_participants_used"] == 0
    assert len(reference["features"]) == 51
    assert "participant_id" not in json.dumps(reference).lower()


def test_reference_source_table_is_exactly_the_authorized_train_cohort(artifacts) -> None:
    train = pd.read_csv(DEPLOYMENT_TRAIN_PATH)
    assert len(train) == 320
    assert train["participant_id"].nunique() == 320
    assert train["Split"].eq("TRAIN").all()
    assert train["screening_label"].value_counts().to_dict() == {"Healthy": 160, "Impaired": 160}
    assert list(train.loc[:, artifacts.schema["ordered_feature_names"]].columns) == artifacts.schema[
        "ordered_feature_names"
    ]


def test_reference_features_and_directions_match_the_frozen_schema_and_model(artifacts, reference) -> None:
    features = reference["features"]
    ordered = artifacts.schema["ordered_feature_names"]
    assert [entry["feature_key"] for entry in features] == ordered
    assert len(set(entry["feature_key"] for entry in features)) == 51
    coefficients = artifacts.pipeline.named_steps["classifier"].coef_.ravel()
    for entry, coefficient in zip(features, coefficients, strict=True):
        assert np.isclose(entry["coefficient"], coefficient)
        assert entry["model_direction"] == model_direction(coefficient)
        assert entry["healthy_reference"]["valid_n"] <= 160
        assert entry["impaired_reference"]["valid_n"] <= 160
    validate_reference_model_alignment(reference, artifacts.pipeline, artifacts.schema)
    altered = deepcopy(reference)
    altered["features"][0]["coefficient"] = 999.0
    with pytest.raises(ValueError, match="coefficient mismatch"):
        validate_reference_model_alignment(altered, artifacts.pipeline, artifacts.schema)


def test_reference_standardized_means_use_the_frozen_preprocessing(artifacts, reference) -> None:
    train = pd.read_csv(DEPLOYMENT_TRAIN_PATH)
    ordered = artifacts.schema["ordered_feature_names"]
    raw = train.loc[:, ordered].apply(pd.to_numeric, errors="coerce")
    standardized = artifacts.pipeline.named_steps["scaler"].transform(
        artifacts.pipeline.named_steps["imputer"].transform(raw)
    )
    healthy = train["screening_label"].eq("Healthy").to_numpy()
    impaired = train["screening_label"].eq("Impaired").to_numpy()
    for index, entry in enumerate(reference["features"]):
        assert np.isclose(entry["healthy_reference_standardized_mean"], standardized[healthy, index].mean())
        assert np.isclose(entry["impaired_reference_standardized_mean"], standardized[impaired, index].mean())


def test_reference_loading_cannot_change_frozen_inference(artifacts, reference) -> None:
    train = pd.read_csv(DEPLOYMENT_TRAIN_PATH)
    row = train.loc[[0], artifacts.schema["ordered_feature_names"]]
    before = predict_screening(row, artifacts)
    # Loading/inspecting the optional JSON must not mutate the pipeline or its score.
    assert reference["artifact_role"].endswith("does not affect model prediction")
    after = predict_screening(row, artifacts)
    assert before == after
