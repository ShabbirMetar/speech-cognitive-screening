"""Build a compact, aggregate-only TRAIN reference for result visualization.

This development utility reads only existing 320-participant TRAIN feature
artifacts. It does not fit, modify, or serialize the frozen classifier, and it
does not load any PROCESS-2 TEST row or result.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from ml.data.development_guard import assert_train_only
from ml.models.explainability import (
    feature_group,
    feature_task,
    feature_unit,
    human_feature_label,
    model_direction,
)
from ml.models.frozen_pipeline import DEFAULT_MODEL_DIRECTORY, load_frozen_artifacts
from ml.models.train_feature_reference import (
    DEFAULT_TRAIN_FEATURE_REFERENCE_PATH,
    REFERENCE_ARTIFACT_VERSION,
    validate_reference_model_alignment,
    validate_train_feature_reference,
)
from scripts.analyze_model_explainability import (
    ASR_LINGUISTIC_TRAIN_PATH,
    DEPLOYMENT_TRAIN_PATH,
    EXPECTED_DEVELOPMENT_PARTICIPANTS,
    validate_frozen_schema,
)


def _finite_or_none(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _raw_statistics(values: pd.Series) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = numeric.dropna()
    return {
        "valid_n": int(valid.size),
        "mean": _finite_or_none(valid.mean()),
        "median": _finite_or_none(valid.median()),
        "q25": _finite_or_none(valid.quantile(0.25)),
        "q75": _finite_or_none(valid.quantile(0.75)),
    }


def _atomic_json_write(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, destination)


def build_reference_payload() -> dict[str, object]:
    """Produce one strictly TRAIN-only aggregate artifact from frozen inputs."""

    for path in (ASR_LINGUISTIC_TRAIN_PATH, DEPLOYMENT_TRAIN_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required TRAIN feature artifact is missing: {path.name}")
    asr = assert_train_only(pd.read_csv(ASR_LINGUISTIC_TRAIN_PATH))
    deployment, constructed_features, _ = validate_frozen_schema(asr, pd.read_csv(DEPLOYMENT_TRAIN_PATH))
    train = assert_train_only(deployment)
    if len(train) != EXPECTED_DEVELOPMENT_PARTICIPANTS or train["participant_id"].nunique() != 320:
        raise AssertionError("TRAIN reference requires exactly 320 unique development participants.")
    if not train["Split"].eq("TRAIN").all():
        raise AssertionError("TRAIN reference must not include any TEST row.")
    counts = train["screening_label"].value_counts().to_dict()
    if counts != {"Healthy": 160, "Impaired": 160}:
        raise AssertionError("TRAIN reference requires 160 Healthy and 160 Impaired participants.")

    artifacts = load_frozen_artifacts(DEFAULT_MODEL_DIRECTORY)
    ordered = list(artifacts.schema["ordered_feature_names"])
    if constructed_features != ordered or len(ordered) != 51:
        raise AssertionError("TRAIN reference features do not match the frozen 51-feature schema.")
    raw = train.loc[:, ordered].apply(pd.to_numeric, errors="coerce")
    imputed = artifacts.pipeline.named_steps["imputer"].transform(raw)
    standardized = artifacts.pipeline.named_steps["scaler"].transform(imputed)
    coefficients = np.asarray(artifacts.pipeline.named_steps["classifier"].coef_, dtype=float).reshape(-1)
    if coefficients.size != 51:
        raise AssertionError("Frozen Logistic Regression coefficient count must be 51.")

    healthy_mask = train["screening_label"].eq("Healthy").to_numpy()
    impaired_mask = train["screening_label"].eq("Impaired").to_numpy()
    entries: list[dict[str, object]] = []
    for index, feature in enumerate(ordered):
        healthy = _raw_statistics(raw.loc[healthy_mask, feature])
        impaired = _raw_statistics(raw.loc[impaired_mask, feature])
        overall = _raw_statistics(raw[feature])
        coefficient = float(coefficients[index])
        entries.append(
            {
                "feature_key": feature,
                "human_readable_label": human_feature_label(feature),
                "task": feature_task(feature),
                "feature_group": feature_group(feature),
                "unit": feature_unit(feature),
                "healthy_reference": healthy,
                "impaired_reference": impaired,
                "overall_train_mean": overall["mean"],
                "overall_train_std": _finite_or_none(raw[feature].std(ddof=0)),
                "healthy_reference_standardized_mean": float(np.mean(standardized[healthy_mask, index])),
                "impaired_reference_standardized_mean": float(np.mean(standardized[impaired_mask, index])),
                "coefficient": coefficient,
                "model_direction": model_direction(coefficient),
            }
        )
    payload: dict[str, object] = {
        "artifact_version": REFERENCE_ARTIFACT_VERSION,
        "artifact_role": "TRAIN-only aggregated visualization reference; does not affect model prediction",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": {
            "split": "TRAIN",
            "participant_count": 320,
            "healthy_participant_count": 160,
            "impaired_participant_count": 160,
            "official_test_participants_used": 0,
            "representation": "ASR_RATE_PLUS_CTD",
            "feature_count": 51,
            "raw_statistics": "NaN values are excluded per feature; no global raw-value imputation is applied.",
            "standardized_statistics": "Frozen TRAIN-fitted median imputer followed by frozen StandardScaler; no scaler is fitted here.",
        },
        "features": entries,
    }
    validate_train_feature_reference(payload, artifacts.schema)
    validate_reference_model_alignment(payload, artifacts.pipeline, artifacts.schema)
    return payload


def main() -> int:
    payload = build_reference_payload()
    _atomic_json_write(payload, DEFAULT_TRAIN_FEATURE_REFERENCE_PATH)
    print("TRAIN Feature Reference Build")
    print("=============================")
    print("Development participants: 320")
    print("Healthy TRAIN reference participants: 160")
    print("Impaired TRAIN reference participants: 160")
    print("Official TEST participants used: 0")
    print("Frozen representation: ASR_RATE_PLUS_CTD")
    print("Features: 51")
    print("Raw statistics: per-feature finite values only (undefined values remain excluded).")
    print("Standardized statistics: frozen median imputer + frozen StandardScaler; no fitting performed.")
    print(f"Saved aggregate-only reference: {DEFAULT_TRAIN_FEATURE_REFERENCE_PATH}")
    print("This reference artifact does not affect model prediction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
