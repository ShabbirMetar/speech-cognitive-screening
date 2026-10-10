"""Integration checks for locally generated frozen model artifacts.

The artifacts are intentionally generated locally and ignored by Git, so this
module skips only when the explicit freeze script has not been run yet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ml.models.explainability import RESEARCH_OPERATING_THRESHOLD, screening_classification
from ml.models.frozen_pipeline import (
    DEFAULT_MODEL_DIRECTORY,
    MANIFEST_FILENAME,
    linear_model_attributions,
    load_frozen_artifacts,
    predict_screening,
    validate_and_order_features,
    validate_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT_TRAIN_PATH = PROJECT_ROOT / "artifacts" / "features" / "deployment_features_train.csv"


pytestmark = pytest.mark.skipif(
    not (DEFAULT_MODEL_DIRECTORY / MANIFEST_FILENAME).is_file(),
    reason="Run scripts/freeze_final_model.py before validating local frozen artifacts.",
)


@pytest.fixture(scope="module")
def artifacts():
    return load_frozen_artifacts()


@pytest.fixture(scope="module")
def one_feature_row(artifacts):
    train = pd.read_csv(DEPLOYMENT_TRAIN_PATH)
    ordered = artifacts.schema["ordered_feature_names"]
    return train.loc[[0], ordered]


def test_serialized_pipeline_and_metadata_match_the_frozen_contract(artifacts) -> None:
    pipeline = artifacts.pipeline
    assert len(artifacts.schema["ordered_feature_names"]) == 51
    assert len(set(artifacts.schema["ordered_feature_names"])) == 51
    assert isinstance(pipeline.named_steps["imputer"], SimpleImputer)
    assert pipeline.named_steps["imputer"].strategy == "median"
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    assert isinstance(pipeline.named_steps["classifier"], LogisticRegression)
    assert pipeline.named_steps["classifier"].C == 0.1
    assert artifacts.metadata["training_participants"] == 320
    assert artifacts.metadata["official_test_used_during_development"] is False
    assert artifacts.metadata["research_threshold"] == RESEARCH_OPERATING_THRESHOLD


def test_prediction_schema_reorders_but_rejects_altered_inputs(artifacts, one_feature_row) -> None:
    shuffled = one_feature_row.loc[:, list(reversed(one_feature_row.columns))]
    ordered = validate_and_order_features(shuffled, artifacts.schema)
    assert list(ordered.columns) == artifacts.schema["ordered_feature_names"]
    with pytest.raises(ValueError, match="missing required"):
        validate_and_order_features(one_feature_row.drop(columns=[one_feature_row.columns[0]]), artifacts.schema)
    duplicated = pd.concat([one_feature_row, one_feature_row.iloc[:, [0]]], axis=1)
    with pytest.raises(ValueError, match="duplicate"):
        validate_and_order_features(duplicated, artifacts.schema)


def test_scores_boundary_and_manifest_integrity(artifacts, one_feature_row) -> None:
    result = predict_screening(one_feature_row, artifacts)[0]
    assert 0.0 <= result["raw_model_score"] <= 1.0
    assert 0.0 <= result["cognitive_speech_screening_score"] <= 100.0
    assert screening_classification(0.53) == "Possible impairment-like speech pattern"
    assert screening_classification(np.nextafter(0.53, 0.0)) == "Healthy-like speech pattern"
    validate_manifest(artifacts.model_directory, artifacts.manifest)
    invalid_manifest = {**artifacts.manifest, "artifact_sha256": {**artifacts.manifest["artifact_sha256"]}}
    invalid_manifest["artifact_sha256"]["feature_schema.json"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_manifest(artifacts.model_directory, invalid_manifest)


def test_linear_runtime_attributions_reconstruct_the_frozen_log_odds(artifacts, one_feature_row) -> None:
    attribution = linear_model_attributions(one_feature_row, artifacts)
    reconstructed = attribution.base_log_odds + attribution.log_odds_contributions.sum(axis=1)
    assert attribution.log_odds_contributions.shape == (1, 51)
    assert np.allclose(reconstructed, attribution.decision_log_odds, rtol=1e-10, atol=1e-10)
