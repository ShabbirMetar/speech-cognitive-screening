"""Aggregate-only TRAIN reference metadata for result-page visualization.

This module deliberately stores and validates descriptive group aggregates only.
It is not read by the frozen classifier and cannot affect prediction, thresholding,
or preprocessing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from ml.models.explainability import (
    feature_group,
    feature_task,
    feature_unit,
    human_feature_label,
    model_direction,
)
from ml.models.frozen_pipeline import PROJECT_ROOT, validate_feature_schema


DEFAULT_TRAIN_FEATURE_REFERENCE_PATH = PROJECT_ROOT / "artifacts" / "reference" / "train_feature_reference.json"
REFERENCE_ARTIFACT_VERSION = "1.0.0"


def validate_train_feature_reference(
    payload: Mapping[str, object], schema: Mapping[str, object]
) -> dict[str, object]:
    """Require an aggregate-only reference artifact matching the frozen schema."""

    if not isinstance(payload, dict):
        raise ValueError("TRAIN reference artifact must be a JSON object.")
    if payload.get("artifact_version") != REFERENCE_ARTIFACT_VERSION:
        raise ValueError("TRAIN reference artifact version is unsupported.")
    if payload.get("artifact_role") != "TRAIN-only aggregated visualization reference; does not affect model prediction":
        raise ValueError("TRAIN reference artifact role is invalid.")
    source = payload.get("source")
    if not isinstance(source, dict):
        raise ValueError("TRAIN reference artifact is missing source metadata.")
    expected_source = {
        "split": "TRAIN",
        "participant_count": 320,
        "healthy_participant_count": 160,
        "impaired_participant_count": 160,
        "official_test_participants_used": 0,
    }
    if {key: source.get(key) for key in expected_source} != expected_source:
        raise ValueError("TRAIN reference artifact does not record the required TRAIN-only cohort.")
    if source.get("feature_count") != 51 or source.get("representation") != "ASR_RATE_PLUS_CTD":
        raise ValueError("TRAIN reference artifact does not match the frozen representation.")

    ordered = validate_feature_schema(schema)
    features = payload.get("features")
    if not isinstance(features, list) or len(features) != 51:
        raise ValueError("TRAIN reference artifact must contain exactly 51 feature aggregates.")
    if "participant_id" in json.dumps(payload).lower():
        raise ValueError("TRAIN reference artifact must not contain participant identifiers.")

    by_key: dict[str, dict[str, object]] = {}
    for item in features:
        if not isinstance(item, dict) or not isinstance(item.get("feature_key"), str):
            raise ValueError("TRAIN reference artifact contains an invalid feature entry.")
        key = item["feature_key"]
        if key in by_key:
            raise ValueError("TRAIN reference artifact contains duplicate features.")
        expected_group = feature_group(key)
        if item.get("feature_group") != expected_group:
            raise ValueError(f"TRAIN reference group mismatch for {key}.")
        if item.get("human_readable_label") != human_feature_label(key):
            raise ValueError(f"TRAIN reference label mismatch for {key}.")
        if item.get("task") != feature_task(key) or item.get("unit") != feature_unit(key):
            raise ValueError(f"TRAIN reference display metadata mismatch for {key}.")
        for reference_name in ("healthy_reference", "impaired_reference"):
            statistics = item.get(reference_name)
            if not isinstance(statistics, dict):
                raise ValueError(f"TRAIN reference statistics are missing for {key}.")
            valid_n = statistics.get("valid_n")
            if not isinstance(valid_n, int) or not 0 <= valid_n <= 160:
                raise ValueError(f"TRAIN reference valid count is invalid for {key}.")
            for statistic in ("mean", "median", "q25", "q75"):
                value = statistics.get(statistic)
                if value is not None and not np.isfinite(float(value)):
                    raise ValueError(f"TRAIN reference {statistic} is invalid for {key}.")
        for numeric_name in (
            "healthy_reference_standardized_mean",
            "impaired_reference_standardized_mean",
            "coefficient",
        ):
            if not np.isfinite(float(item.get(numeric_name, float("nan")))):
                raise ValueError(f"TRAIN reference {numeric_name} is invalid for {key}.")
        if item.get("model_direction") != model_direction(float(item["coefficient"])):
            raise ValueError(f"TRAIN reference model direction is invalid for {key}.")
        by_key[key] = item
    if list(by_key) != ordered:
        raise ValueError("TRAIN reference feature ordering does not match the frozen schema.")
    return dict(payload)


def load_train_feature_reference(
    schema: Mapping[str, object], path: Path = DEFAULT_TRAIN_FEATURE_REFERENCE_PATH
) -> dict[str, object]:
    """Load one validated aggregate-only artifact; callers may handle absence gracefully."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"TRAIN reference artifact is unavailable: {source.name}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    return validate_train_feature_reference(payload, schema)


def validate_reference_model_alignment(
    payload: Mapping[str, object], pipeline: object, schema: Mapping[str, object]
) -> None:
    """Ensure visual coefficients/directions still match the frozen classifier.

    A mismatch only makes the optional reference unavailable; it must never be
    used to alter classifier prediction or preprocessing.
    """

    reference = validate_train_feature_reference(payload, schema)
    named_steps = getattr(pipeline, "named_steps", None)
    classifier = named_steps.get("classifier") if isinstance(named_steps, dict) else None
    coefficients = np.asarray(getattr(classifier, "coef_", []), dtype=float).reshape(-1)
    ordered = validate_feature_schema(schema)
    if coefficients.size != len(ordered):
        raise ValueError("Frozen classifier coefficients are unavailable for reference validation.")
    for entry, coefficient in zip(reference["features"], coefficients, strict=True):
        if not isinstance(entry, dict):
            raise ValueError("TRAIN reference contains an invalid feature entry.")
        if not np.isclose(float(entry.get("coefficient", float("nan"))), coefficient, rtol=1e-12, atol=1e-12):
            raise ValueError(f"TRAIN reference coefficient mismatch for {entry.get('feature_key')}.")
        if entry.get("model_direction") != model_direction(float(coefficient)):
            raise ValueError(f"TRAIN reference model direction mismatch for {entry.get('feature_key')}.")
