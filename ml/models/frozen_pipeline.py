"""Loading, schema validation, and inference helpers for the frozen research pipeline.

The helpers deliberately do not fit or alter preprocessing.  They load the
serialized all-TRAIN pipeline and apply its frozen imputer, scaler, and
Logistic Regression classifier to a schema-validated feature table.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.models.explainability import (
    RESEARCH_OPERATING_THRESHOLD,
    screening_classification,
    screening_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIRECTORY = PROJECT_ROOT / "artifacts" / "models"
PIPELINE_FILENAME = "cognitive_screening_pipeline.joblib"
SCHEMA_FILENAME = "feature_schema.json"
METADATA_FILENAME = "model_metadata.json"
MANIFEST_FILENAME = "model_manifest.json"


@dataclass(frozen=True)
class FrozenScreeningArtifacts:
    """Loaded, integrity-checked assets required for frozen-model inference."""

    pipeline: Pipeline
    schema: dict[str, object]
    metadata: dict[str, object]
    manifest: dict[str, object]
    model_directory: Path


@dataclass(frozen=True)
class LinearModelAttributions:
    """Exact additive linear-model contributions for schema-validated rows.

    Contributions are on the classifier decision-function (log-odds) scale
    relative to the zero-centred, TRAIN-fitted standardized representation.
    They are suitable for a runtime explanation without storing participant
    training rows or recomputing global SHAP values.
    """

    feature_names: tuple[str, ...]
    observed_values: np.ndarray
    imputed_values: np.ndarray
    train_standardization_means: np.ndarray
    log_odds_contributions: np.ndarray
    base_log_odds: float
    decision_log_odds: np.ndarray


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for one artifact file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_feature_schema(schema: Mapping[str, object]) -> list[str]:
    """Validate the frozen 51-feature schema and return its ordering."""

    ordered = schema.get("ordered_feature_names")
    feature_count = schema.get("feature_count")
    if not isinstance(ordered, list) or not all(isinstance(feature, str) for feature in ordered):
        raise ValueError("Frozen feature schema must contain an ordered string feature list.")
    if feature_count != 51 or len(ordered) != 51 or len(set(ordered)) != 51:
        raise ValueError("Frozen feature schema must contain exactly 51 unique features.")
    groups = schema.get("feature_groups")
    if not isinstance(groups, dict) or set(groups) != set(ordered):
        raise ValueError("Frozen feature-group metadata must cover exactly the ordered features.")
    allowed_groups = {"ASR_LINGUISTIC", "ASR_RATE", "CTD_ACOUSTIC"}
    if not set(groups.values()).issubset(allowed_groups):
        raise ValueError("Frozen feature schema contains an unsupported feature group.")
    if Counter(groups.values()) != Counter({"ASR_LINGUISTIC": 32, "ASR_RATE": 6, "CTD_ACOUSTIC": 13}):
        raise ValueError("Frozen feature schema must retain the 32/6/13 modality split.")
    return ordered


def validate_and_order_features(feature_table: pd.DataFrame, schema: Mapping[str, object]) -> pd.DataFrame:
    """Fail safely on altered inputs, then explicitly restore frozen feature order."""

    if not isinstance(feature_table, pd.DataFrame):
        raise TypeError("Inference features must be provided as a pandas DataFrame.")
    if feature_table.columns.duplicated().any():
        duplicates = feature_table.columns[feature_table.columns.duplicated()].tolist()
        raise ValueError(f"Inference features contain duplicate columns: {duplicates}")
    ordered = validate_feature_schema(schema)
    provided = set(feature_table.columns)
    required = set(ordered)
    missing = sorted(required.difference(provided))
    unexpected = sorted(provided.difference(required))
    if missing:
        raise ValueError(f"Inference features are missing required frozen fields: {missing}")
    if unexpected:
        raise ValueError(f"Inference features contain unexpected fields: {unexpected}")
    return feature_table.loc[:, ordered].apply(pd.to_numeric, errors="coerce")


def _validate_pipeline(pipeline: object, schema: Mapping[str, object], metadata: Mapping[str, object]) -> Pipeline:
    """Require the exact preprocessing and classifier family frozen for deployment."""

    if not isinstance(pipeline, Pipeline):
        raise TypeError("Frozen model artifact must be an sklearn Pipeline.")
    required_steps = {"imputer", "scaler", "classifier"}
    if set(pipeline.named_steps) != required_steps or list(pipeline.named_steps) != [
        "imputer",
        "scaler",
        "classifier",
    ]:
        raise ValueError("Frozen pipeline must contain only imputer, scaler, and classifier steps.")
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["classifier"]
    if not isinstance(imputer, SimpleImputer) or imputer.strategy != "median":
        raise ValueError("Frozen pipeline imputer must use median strategy.")
    if not isinstance(scaler, StandardScaler):
        raise ValueError("Frozen pipeline scaler must be StandardScaler.")
    if not isinstance(classifier, LogisticRegression) or not np.isclose(classifier.C, 0.1):
        raise ValueError("Frozen pipeline classifier must be LogisticRegression with C=0.1.")
    ordered = validate_feature_schema(schema)
    if getattr(classifier, "n_features_in_", None) != len(ordered):
        raise ValueError("Frozen classifier feature count does not match the frozen schema.")
    if metadata.get("training_participants") != 320:
        raise ValueError("Frozen metadata must record 320 TRAIN participants.")
    if metadata.get("feature_count") != 51 or not np.isclose(float(metadata.get("logistic_C", float("nan"))), 0.1):
        raise ValueError("Frozen metadata must retain the 51-feature C=0.1 configuration.")
    if metadata.get("official_test_used_during_development") is not False:
        raise ValueError("Frozen metadata must record that official TEST was not used during development.")
    if not np.isclose(float(metadata.get("research_threshold", float("nan"))), RESEARCH_OPERATING_THRESHOLD):
        raise ValueError("Frozen metadata must retain the 0.53 research threshold.")
    return pipeline


def validate_manifest(model_directory: Path, manifest: Mapping[str, object]) -> None:
    """Verify recorded SHA-256 checksums before using frozen artifacts."""

    artifacts = manifest.get("artifact_sha256")
    if not isinstance(artifacts, dict):
        raise ValueError("Model manifest is missing artifact SHA-256 entries.")
    for relative_path, expected_hash in artifacts.items():
        path = Path(relative_path)
        resolved = path if path.is_absolute() else model_directory / path
        if not resolved.is_file():
            raise FileNotFoundError(f"Manifest artifact is missing: {resolved}")
        actual_hash = sha256_file(resolved)
        if actual_hash != expected_hash:
            raise ValueError(f"SHA-256 mismatch for frozen artifact: {resolved.name}")


def load_frozen_artifacts(model_directory: Path = DEFAULT_MODEL_DIRECTORY) -> FrozenScreeningArtifacts:
    """Load and integrity-check the serialized pipeline, schema, and metadata."""

    directory = Path(model_directory)
    paths = {
        "pipeline": directory / PIPELINE_FILENAME,
        "schema": directory / SCHEMA_FILENAME,
        "metadata": directory / METADATA_FILENAME,
        "manifest": directory / MANIFEST_FILENAME,
    }
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Frozen model artifacts are missing: {missing}")
    schema = json.loads(paths["schema"].read_text(encoding="utf-8"))
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    validate_manifest(directory, manifest)
    pipeline = _validate_pipeline(joblib.load(paths["pipeline"]), schema, metadata)
    return FrozenScreeningArtifacts(
        pipeline=pipeline,
        schema=schema,
        metadata=metadata,
        manifest=manifest,
        model_directory=directory,
    )


def transform_frozen_features(
    feature_table: pd.DataFrame, artifacts: FrozenScreeningArtifacts
) -> np.ndarray:
    """Return the exact imputed and standardized representation used by the classifier.

    This is the appropriate input for a future participant-level linear SHAP
    explainer; global SHAP is deliberately not recomputed during inference.
    """

    ordered = validate_and_order_features(feature_table, artifacts.schema)
    imputer = artifacts.pipeline.named_steps["imputer"]
    scaler = artifacts.pipeline.named_steps["scaler"]
    return scaler.transform(imputer.transform(ordered))


def linear_model_attributions(
    feature_table: pd.DataFrame, artifacts: FrozenScreeningArtifacts
) -> LinearModelAttributions:
    """Return exact local linear contributions without fitting or TRAIN-row access.

    For the frozen Logistic Regression model, the standardized TRAIN reference
    has zero-valued coordinates.  Therefore ``coefficient * standardized
    value`` is the exact additive contribution for each feature relative to
    that reference, and the intercept is the base log-odds value.  This is a
    linear, interventional SHAP-equivalent decomposition, but is intentionally
    named as a log-odds attribution so API consumers do not mistake it for a
    probability change or correlation-conditional SHAP analysis.
    """

    ordered = validate_and_order_features(feature_table, artifacts.schema)
    imputer = artifacts.pipeline.named_steps["imputer"]
    scaler = artifacts.pipeline.named_steps["scaler"]
    classifier = artifacts.pipeline.named_steps["classifier"]
    imputed = imputer.transform(ordered)
    transformed = scaler.transform(imputed)
    coefficients = np.asarray(classifier.coef_, dtype=float).reshape(-1)
    if coefficients.size != transformed.shape[1]:
        raise ValueError("Frozen classifier coefficients do not match transformed features.")
    contributions = transformed * coefficients
    base = float(np.asarray(classifier.intercept_, dtype=float).reshape(-1)[0])
    decision = np.asarray(classifier.decision_function(transformed), dtype=float).reshape(-1)
    reconstructed = base + contributions.sum(axis=1)
    if not np.allclose(decision, reconstructed, rtol=1e-10, atol=1e-10):
        raise ValueError("Linear attribution reconstruction does not match the frozen model output.")
    return LinearModelAttributions(
        feature_names=tuple(ordered.columns),
        observed_values=ordered.to_numpy(dtype=float),
        imputed_values=np.asarray(imputed, dtype=float),
        train_standardization_means=np.asarray(scaler.mean_, dtype=float),
        log_odds_contributions=np.asarray(contributions, dtype=float),
        base_log_odds=base,
        decision_log_odds=decision,
    )


def predict_screening(
    feature_table: pd.DataFrame, artifacts: FrozenScreeningArtifacts
) -> list[dict[str, float | str]]:
    """Return raw scores and non-clinical screening labels for validated features."""

    ordered = validate_and_order_features(feature_table, artifacts.schema)
    scores = artifacts.pipeline.predict_proba(ordered)[:, 1]
    threshold = float(artifacts.metadata["research_threshold"])
    results: list[dict[str, float | str]] = []
    for score in scores:
        raw_score = float(score)
        results.append(
            {
                "raw_model_score": raw_score,
                "cognitive_speech_screening_score": screening_score(raw_score),
                "classification": screening_classification(raw_score, threshold),
                "research_threshold": threshold,
                "schema_version": str(artifacts.schema.get("schema_version")),
                "model_version": str(artifacts.metadata.get("model_version")),
            }
        )
    return results
