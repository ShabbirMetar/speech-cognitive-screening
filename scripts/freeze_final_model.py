"""Fit, serialize, and integrity-verify the frozen all-TRAIN screening pipeline.

This script never reads a TEST artifact.  It is a reproducibility/freeze step,
not an estimate of held-out performance.
"""

from __future__ import annotations

from datetime import UTC, datetime
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from ml.data.development_guard import assert_train_only
from ml.models.baseline_logistic import build_logistic_pipeline, encode_screening_labels
from ml.models.explainability import FROZEN_REPRESENTATION, RESEARCH_OPERATING_THRESHOLD
from ml.models.frozen_pipeline import (
    METADATA_FILENAME,
    PIPELINE_FILENAME,
    SCHEMA_FILENAME,
    DEFAULT_MODEL_DIRECTORY,
    load_frozen_artifacts,
    predict_screening,
    sha256_file,
)
from scripts.analyze_model_explainability import (
    ASR_LINGUISTIC_TRAIN_PATH,
    DEPLOYMENT_TRAIN_PATH,
    EXPECTED_DEVELOPMENT_PARTICIPANTS,
    validate_frozen_schema,
)


ASR_CONFIG_PATH = PROJECT_ROOT / "artifacts" / "transcripts" / "asr_train" / "asr_config.json"
MANIFEST_FILENAME = "model_manifest.json"
MODEL_VERSION = "1.0.0"
FINAL_C = 0.1


def _atomic_json_write(payload: dict[str, Any], destination: Path) -> None:
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, destination)


def _atomic_joblib_dump(value: object, destination: Path) -> None:
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    joblib.dump(value, temporary)
    os.replace(temporary, destination)


def _frozen_asr_config() -> dict[str, object]:
    """Load and validate the established base.en config without invoking ASR."""

    if not ASR_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Frozen ASR configuration is missing: {ASR_CONFIG_PATH}")
    config = json.loads(ASR_CONFIG_PATH.read_text(encoding="utf-8"))
    expected = {
        "extractor": "faster-whisper",
        "model_size": "base.en",
        "device": "cpu",
        "compute_type": "int8",
        "language": "en",
        "cpu_threads": 4,
        "num_workers": 1,
    }
    mismatches = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if mismatches:
        raise ValueError(f"Stored ASR configuration does not match the frozen deployment settings: {mismatches}")
    return config


def _metadata(
    feature_names: list[str], groups: dict[str, str], asr_config: dict[str, object]
) -> dict[str, object]:
    """Build non-clinical deployment metadata without recording training accuracy."""

    return {
        "model_version": MODEL_VERSION,
        "project_name": "Speech-Based Cognitive Decline Screening",
        "model_type": "Logistic Regression",
        "model_role": "speech-based cognitive screening research prototype",
        "target": {"0": "Healthy", "1": "Impaired (MCI + Dementia)"},
        "training_participants": EXPECTED_DEVELOPMENT_PARTICIPANTS,
        "official_test_used_during_development": False,
        "asr_model": "base.en",
        "asr_deployment_configuration": asr_config,
        "feature_representation": FROZEN_REPRESENTATION,
        "feature_count": len(feature_names),
        "feature_groups": groups,
        "feature_provenance": {
            "ASR_LINGUISTIC": "Derived from faster-whisper base.en transcripts.",
            "ASR_RATE": "Derived from ASR word counts combined with frozen waveform timing.",
            "CTD_ACOUSTIC": "Derived directly from CTD waveform processing.",
            "excluded_predictors": [
                "manual transcript pause annotations",
                "age",
                "sex/gender",
                "MMSE",
                "participant_id",
                "diagnosis",
                "labels",
            ],
        },
        "logistic_C": FINAL_C,
        "logistic_max_iter": 2000,
        "logistic_random_state": 42,
        "imputer": "SimpleImputer(strategy=median)",
        "scaler": "StandardScaler",
        "score_type": "raw Logistic Regression impaired-class classifier score",
        "score_terminology": "Cognitive Speech Screening Score; a preliminary speech-based screening signal",
        "display_score": "raw model score * 100; display transformation only, not a medical probability or risk percentage",
        "research_threshold": RESEARCH_OPERATING_THRESHOLD,
        "threshold_source": "TRAIN-only OOF threshold analysis; maximum balanced accuracy and maximum Youden J",
        "threshold_claim": "TRAIN-derived research operating threshold; not clinically validated",
        "classification_rule": {
            "raw_score_greater_than_or_equal_to_0.53": "Possible impairment-like speech pattern",
            "raw_score_less_than_0.53": "Healthy-like speech pattern",
        },
        "calibration": "raw Logistic Regression score; no additional sigmoid calibration",
        "shap_scale": "log-odds",
        "development_results": {
            "frozen_representation_outer_cv": {
                "f1_mean": 0.650,
                "f1_std": 0.085,
                "roc_auc_mean": 0.717,
                "roc_auc_std": 0.079,
                "train_validation_f1_gap": 0.073,
                "label": "TRAIN-only nested-CV development result; not final TEST performance",
            },
            "final_C_resolution": {
                "C": FINAL_C,
                "five_fold_train_cv_f1_mean": 0.6501836512184415,
                "five_fold_train_cv_f1_std": 0.07618159968899109,
                "label": "TRAIN-only C resolution for final all-TRAIN fit; not held-out performance",
            },
            "raw_score_oof_diagnostics": {
                "brier_score": 0.21674911799243257,
                "log_loss": 0.6558658429852636,
                "roc_auc": 0.7137109375,
                "label": "TRAIN-only OOF diagnostics; not final TEST performance",
            },
        },
    }


def _schema(feature_names: list[str], groups: dict[str, str]) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "feature_representation": FROZEN_REPRESENTATION,
        "feature_count": len(feature_names),
        "ordered_feature_names": feature_names,
        "feature_groups": groups,
        "inference_contract": {
            "missing_required_features": "fail",
            "duplicate_features": "fail",
            "unexpected_features": "fail",
            "column_order": "explicitly reordered to ordered_feature_names after validation",
        },
    }


def main() -> int:
    for path in (ASR_LINGUISTIC_TRAIN_PATH, DEPLOYMENT_TRAIN_PATH, ASR_CONFIG_PATH):
        if not path.is_file():
            print(f"Required frozen TRAIN/config artifact not found: {path}")
            return 1
    deployment, feature_names, groups = validate_frozen_schema(
        pd.read_csv(ASR_LINGUISTIC_TRAIN_PATH), pd.read_csv(DEPLOYMENT_TRAIN_PATH)
    )
    train = assert_train_only(deployment)
    if len(train) != EXPECTED_DEVELOPMENT_PARTICIPANTS:
        raise AssertionError("Final freeze requires exactly 320 TRAIN participants.")
    raw_features = train.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    target = encode_screening_labels(train["screening_label"]).to_numpy()
    pipeline = build_logistic_pipeline().set_params(classifier__C=FINAL_C)
    pipeline.fit(raw_features, target)
    if pipeline.named_steps["classifier"].C != FINAL_C:
        raise AssertionError("Final pipeline C must remain 0.1.")
    before_serialization = pipeline.predict_proba(raw_features)[:, 1]

    asr_config = _frozen_asr_config()
    schema = _schema(feature_names, groups)
    metadata = _metadata(feature_names, groups, asr_config)
    DEFAULT_MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    pipeline_path = DEFAULT_MODEL_DIRECTORY / PIPELINE_FILENAME
    schema_path = DEFAULT_MODEL_DIRECTORY / SCHEMA_FILENAME
    metadata_path = DEFAULT_MODEL_DIRECTORY / METADATA_FILENAME
    manifest_path = DEFAULT_MODEL_DIRECTORY / MANIFEST_FILENAME
    _atomic_joblib_dump(pipeline, pipeline_path)
    _atomic_json_write(schema, schema_path)
    _atomic_json_write(metadata, metadata_path)

    manifest = {
        "manifest_version": "1.0.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_version": MODEL_VERSION,
        "creation_scope": "all-TRAIN final artifact freeze; official TEST participants used: 0",
        "runtime": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "scikit_learn_version": sklearn.__version__,
            "shap_version": shap.__version__,
        },
        "asr": {
            "faster_whisper_version": package_version("faster-whisper"),
            "source_config_path": str(ASR_CONFIG_PATH),
        },
        "artifact_sha256": {
            PIPELINE_FILENAME: sha256_file(pipeline_path),
            SCHEMA_FILENAME: sha256_file(schema_path),
            METADATA_FILENAME: sha256_file(metadata_path),
            str(ASR_CONFIG_PATH): sha256_file(ASR_CONFIG_PATH),
        },
    }
    _atomic_json_write(manifest, manifest_path)
    loaded = load_frozen_artifacts(DEFAULT_MODEL_DIRECTORY)
    after_serialization = loaded.pipeline.predict_proba(raw_features)[:, 1]
    maximum_difference = float(np.max(np.abs(before_serialization - after_serialization)))
    if not np.allclose(before_serialization, after_serialization, rtol=1e-15, atol=1e-15):
        raise AssertionError("Serialized pipeline scores changed after roundtrip reload.")
    inference_results = predict_screening(raw_features.iloc[:3], loaded)
    if not all(0.0 <= result["raw_model_score"] <= 1.0 for result in inference_results):
        raise AssertionError("Frozen inference helper returned an invalid raw score.")
    if not all(0.0 <= result["cognitive_speech_screening_score"] <= 100.0 for result in inference_results):
        raise AssertionError("Frozen inference helper returned an invalid display score.")
    manifest["roundtrip_validation"] = {
        "train_rows_checked": int(len(train)),
        "maximum_pre_save_vs_post_load_prediction_difference": maximum_difference,
        "tolerance": {"rtol": 1e-15, "atol": 1e-15},
    }
    _atomic_json_write(manifest, manifest_path)

    print("Final All-TRAIN Model Freeze")
    print("============================")
    print(f"Development participants: {len(train)}")
    print("Official TEST participants used: 0")
    print(f"Frozen representation: {FROZEN_REPRESENTATION}")
    print(f"Features: {len(feature_names)}")
    print("Classifier: Logistic Regression")
    print(f"Selected C: {FINAL_C}")
    print(f"Research operating threshold: {RESEARCH_OPERATING_THRESHOLD:.2f}")
    print(f"Pipeline: {pipeline_path}")
    print(f"Feature schema: {schema_path}")
    print(f"Model metadata: {metadata_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Maximum pre-save vs post-load prediction difference: {maximum_difference:.3e}")
    print("SHA-256:")
    for name, digest in manifest["artifact_sha256"].items():
        print(f"  {name}: {digest}")
    print("No training performance is reported; this run verifies artifact integrity only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
