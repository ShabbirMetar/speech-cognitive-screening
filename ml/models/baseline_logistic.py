"""TRAIN-only Logistic Regression baseline for linguistic features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ExperimentName = Literal["SFT", "PFT", "CTD", "ALL"]
TARGET_COLUMN = "screening_label"
SPLIT_COLUMN = "Split"
EXCLUDED_COLUMNS = {
    "participant_id",
    "IDs",
    "diagnosis",
    TARGET_COLUMN,
    SPLIT_COLUMN,
    "age",
    "gender",
    "MMSE",
}
EXPERIMENT_PREFIXES: dict[ExperimentName, tuple[str, ...]] = {
    "SFT": ("sft_",),
    "PFT": ("pft_",),
    "CTD": ("ctd_",),
    "ALL": ("sft_", "pft_", "ctd_"),
}
METRIC_COLUMNS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "roc_auc",
)


@dataclass(frozen=True)
class ExperimentResult:
    """Fold-level and aggregate outputs for one TRAIN-only experiment."""

    name: ExperimentName
    cv_metrics: pd.DataFrame
    comparison_row: dict[str, float | str]
    oof_predictions: pd.DataFrame
    confusion_matrix: np.ndarray


def encode_screening_labels(labels: pd.Series) -> pd.Series:
    """Encode Healthy as 0 and Impaired as 1, rejecting unexpected labels."""

    encoded = labels.astype("string").str.strip().map({"Healthy": 0, "Impaired": 1})
    if encoded.isna().any():
        raise ValueError("screening_label must contain only Healthy or Impaired values.")
    return encoded.astype(int)


def training_rows(feature_table: pd.DataFrame) -> pd.DataFrame:
    """Return only official TRAIN rows; TEST is not passed to cross-validation."""

    if SPLIT_COLUMN not in feature_table:
        raise ValueError(f"Feature table is missing {SPLIT_COLUMN!r}.")
    return feature_table[
        feature_table[SPLIT_COLUMN].astype("string").str.strip().str.upper().eq("TRAIN")
    ].reset_index(drop=True)


def select_feature_columns(feature_table: pd.DataFrame, experiment: ExperimentName) -> list[str]:
    """Select schema-defined linguistic features without using labels or outcomes."""

    prefixes = EXPERIMENT_PREFIXES[experiment]
    selected = [
        column
        for column in feature_table.columns
        if column not in EXCLUDED_COLUMNS and column.casefold().startswith(prefixes)
    ]
    if not selected:
        raise ValueError(f"No {experiment} linguistic features were found.")
    return selected


def build_logistic_pipeline() -> Pipeline:
    """Create fold-safe numeric preprocessing and Logistic Regression pipeline."""

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )


def _specificity(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    true_negative, false_positive, _, _ = confusion_matrix(
        y_true, y_predicted, labels=[0, 1]
    ).ravel()
    return float(true_negative / (true_negative + false_positive)) if true_negative + false_positive else float("nan")


def _validation_metrics(
    y_true: np.ndarray, y_predicted: np.ndarray, impaired_probability: np.ndarray
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_predicted)),
        "precision": float(precision_score(y_true, y_predicted, zero_division=0)),
        "recall": float(recall_score(y_true, y_predicted, zero_division=0)),
        "specificity": _specificity(y_true, y_predicted),
        "f1": float(f1_score(y_true, y_predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, impaired_probability)),
    }


def run_cross_validated_experiment(
    feature_table: pd.DataFrame, experiment: ExperimentName
) -> ExperimentResult:
    """Run deterministic 5-fold CV using only official TRAIN participants."""

    train_table = training_rows(feature_table)
    feature_names = select_feature_columns(train_table, experiment)
    features = train_table[feature_names].apply(pd.to_numeric, errors="coerce")
    target = encode_screening_labels(train_table[TARGET_COLUMN]).to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = build_logistic_pipeline()
    fold_rows: list[dict[str, float | int | str]] = []
    oof_rows: list[dict[str, float | int | str]] = []

    # TEST rows are deliberately absent from both fitting and validation folds.
    for fold_number, (fit_index, validation_index) in enumerate(cv.split(features, target), start=1):
        fitted_pipeline = clone(pipeline)
        fitted_pipeline.fit(features.iloc[fit_index], target[fit_index])
        training_prediction = fitted_pipeline.predict(features.iloc[fit_index])
        validation_prediction = fitted_pipeline.predict(features.iloc[validation_index])
        validation_probability = fitted_pipeline.predict_proba(features.iloc[validation_index])[:, 1]
        validation_metrics = _validation_metrics(
            target[validation_index], validation_prediction, validation_probability
        )
        train_f1 = float(f1_score(target[fit_index], training_prediction, zero_division=0))
        fold_rows.append(
            {
                "experiment": experiment,
                "fold": fold_number,
                **validation_metrics,
                "training_f1": train_f1,
                "validation_f1": validation_metrics["f1"],
                "train_validation_f1_gap": train_f1 - validation_metrics["f1"],
            }
        )
        for sample_index, true_label, predicted_label, probability in zip(
            validation_index,
            target[validation_index],
            validation_prediction,
            validation_probability,
            strict=True,
        ):
            oof_rows.append(
                {
                    "anonymized_participant_id": f"train_{sample_index + 1:03d}",
                    "fold": fold_number,
                    "true_binary_label": int(true_label),
                    "predicted_binary_label": int(predicted_label),
                    "impaired_probability": float(probability),
                }
            )

    fold_metrics = pd.DataFrame(fold_rows)
    comparison_row: dict[str, float | str] = {
        "experiment": experiment,
        "feature_count": len(feature_names),
    }
    for metric in METRIC_COLUMNS:
        comparison_row[f"{metric}_mean"] = float(fold_metrics[metric].mean())
        comparison_row[f"{metric}_std"] = float(fold_metrics[metric].std(ddof=1))
    comparison_row["training_f1_mean"] = float(fold_metrics["training_f1"].mean())
    comparison_row["validation_f1_mean"] = float(fold_metrics["validation_f1"].mean())
    comparison_row["train_validation_f1_gap_mean"] = float(
        fold_metrics["train_validation_f1_gap"].mean()
    )
    oof_predictions = pd.DataFrame(oof_rows).sort_values("anonymized_participant_id")
    oof_confusion = confusion_matrix(
        oof_predictions["true_binary_label"],
        oof_predictions["predicted_binary_label"],
        labels=[0, 1],
    )
    return ExperimentResult(
        name=experiment,
        cv_metrics=fold_metrics,
        comparison_row=comparison_row,
        oof_predictions=oof_predictions,
        confusion_matrix=oof_confusion,
    )


def run_all_experiments(feature_table: pd.DataFrame) -> dict[ExperimentName, ExperimentResult]:
    """Run SFT, PFT, CTD, and ALL TRAIN-only CV experiments."""

    return {
        experiment: run_cross_validated_experiment(feature_table, experiment)
        for experiment in ("SFT", "PFT", "CTD", "ALL")
    }
