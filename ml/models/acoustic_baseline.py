"""Nested-CV acoustic-only baselines restricted to the official TRAIN split."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Literal

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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from ml.data.development_guard import assert_train_only
from ml.models.baseline_logistic import METRIC_COLUMNS, TARGET_COLUMN, encode_screening_labels


AcousticExperimentName = Literal["SFT", "PFT", "CTD", "ALL"]
AcousticModelName = Literal["logistic", "linear_svm", "rbf_svm"]
ACOUSTIC_EXPERIMENTS: tuple[AcousticExperimentName, ...] = ("SFT", "PFT", "CTD", "ALL")
ACOUSTIC_MODEL_NAMES: tuple[AcousticModelName, ...] = ("logistic", "linear_svm", "rbf_svm")
OUTER_CV_SPLITS = 5
INNER_CV_SPLITS = 3
SEARCH_N_JOBS = 2

# This intentionally excludes MFCC, spectral, and zero-crossing measurements.
INTERPRETABLE_ACOUSTIC_SUFFIXES: tuple[str, ...] = (
    "audio_duration_seconds",
    "voiced_duration_seconds",
    "silence_duration_seconds",
    "silence_ratio",
    "pause_count",
    "pause_total_seconds",
    "pause_mean_seconds",
    "pause_max_seconds",
    "recording_word_rate_wpm",
    "articulation_rate_wpm",
    "f0_mean_hz",
    "f0_std_hz",
    "f0_range_hz",
    "rms_mean",
    "rms_std",
)
TASK_PREFIXES: dict[AcousticExperimentName, tuple[str, ...]] = {
    "SFT": ("sft_",),
    "PFT": ("pft_",),
    "CTD": ("ctd_",),
    "ALL": ("sft_", "pft_", "ctd_"),
}


@dataclass(frozen=True)
class AcousticExperimentResult:
    """Outer-fold metrics and out-of-fold outputs for one task/model pair."""

    experiment: AcousticExperimentName
    model: AcousticModelName
    feature_columns: list[str]
    fold_metrics: pd.DataFrame
    oof_predictions: pd.DataFrame
    selected_hyperparameters: pd.DataFrame
    comparison_row: dict[str, float | int | str]
    oof_confusion_matrix: np.ndarray


def build_outer_cv() -> StratifiedKFold:
    """Return the frozen five-fold outer development split."""

    return StratifiedKFold(n_splits=OUTER_CV_SPLITS, shuffle=True, random_state=42)


def build_inner_cv() -> StratifiedKFold:
    """Return the frozen three-fold inner hyperparameter-selection split."""

    return StratifiedKFold(n_splits=INNER_CV_SPLITS, shuffle=True, random_state=42)


def select_acoustic_feature_columns(
    feature_table: pd.DataFrame, experiment: AcousticExperimentName
) -> list[str]:
    """Return exactly the prespecified interpretable acoustic columns.

    The selection is schema-defined and does not inspect values or labels, so
    MFCC, spectral, and zero-crossing columns cannot enter this baseline.
    """

    expected = [
        f"{prefix}{suffix}"
        for prefix in TASK_PREFIXES[experiment]
        for suffix in INTERPRETABLE_ACOUSTIC_SUFFIXES
    ]
    missing = [column for column in expected if column not in feature_table.columns]
    if missing:
        raise ValueError(
            f"Acoustic feature table is missing expected {experiment} columns: {missing}"
        )
    return expected


def build_acoustic_pipeline(model: AcousticModelName) -> Pipeline:
    """Create fold-local median imputation, scaling, and a regularized model."""

    classifier: LogisticRegression | LinearSVC | SVC
    if model == "logistic":
        classifier = LogisticRegression(max_iter=2000, random_state=42)
    elif model == "linear_svm":
        classifier = LinearSVC(max_iter=10000, random_state=42)
    elif model == "rbf_svm":
        classifier = SVC(kernel="rbf", random_state=42)
    else:
        raise ValueError(f"Unknown acoustic model family: {model}")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def parameter_grid(model: AcousticModelName) -> dict[str, list[object]]:
    """Return the deliberately compact F1-optimised search grid."""

    if model in ("logistic", "linear_svm"):
        return {"classifier__C": [0.1, 1.0, 10.0]}
    if model == "rbf_svm":
        return {
            "classifier__C": [0.5, 1.0, 5.0],
            "classifier__gamma": ["scale", 0.01],
        }
    raise ValueError(f"Unknown acoustic model family: {model}")


def build_inner_search(
    model: AcousticModelName, *, search_n_jobs: int = SEARCH_N_JOBS
) -> GridSearchCV:
    """Build a fold-local compact grid search using F1 as its score."""

    return GridSearchCV(
        estimator=clone(build_acoustic_pipeline(model)),
        param_grid=parameter_grid(model),
        scoring="f1",
        cv=build_inner_cv(),
        n_jobs=search_n_jobs,
        refit=True,
        return_train_score=False,
        error_score="raise",
        verbose=0,
    )


def _specificity(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    true_negative, false_positive, _, _ = confusion_matrix(y_true, y_predicted, labels=[0, 1]).ravel()
    denominator = true_negative + false_positive
    return float(true_negative / denominator) if denominator else float("nan")


def _validation_metrics(
    y_true: np.ndarray, y_predicted: np.ndarray, score: np.ndarray
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_predicted)),
        "precision": float(precision_score(y_true, y_predicted, zero_division=0)),
        "recall": float(recall_score(y_true, y_predicted, zero_division=0)),
        "specificity": _specificity(y_true, y_predicted),
        "f1": float(f1_score(y_true, y_predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, score)),
    }


def _model_score(fitted_pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Use probabilities where available, otherwise a decision-function score."""

    if hasattr(fitted_pipeline, "predict_proba"):
        return np.asarray(fitted_pipeline.predict_proba(features))[:, 1]
    return np.asarray(fitted_pipeline.decision_function(features))


def run_nested_acoustic_experiment(
    feature_table: pd.DataFrame,
    experiment: AcousticExperimentName,
    model: AcousticModelName,
    *,
    search_n_jobs: int = SEARCH_N_JOBS,
    progress_callback: Callable[[str], None] | None = None,
) -> AcousticExperimentResult:
    """Evaluate one predeclared acoustic task/model pair on TRAIN participants only."""

    train_table = assert_train_only(feature_table)
    feature_columns = select_acoustic_feature_columns(train_table, experiment)
    features = train_table[feature_columns].apply(pd.to_numeric, errors="coerce")
    target = encode_screening_labels(train_table[TARGET_COLUMN]).to_numpy()

    fold_rows: list[dict[str, float | int | str]] = []
    oof_rows: list[dict[str, float | int | str]] = []
    parameter_rows: list[dict[str, object]] = []
    for fold, (fit_index, validation_index) in enumerate(
        build_outer_cv().split(features, target), start=1
    ):
        if progress_callback:
            progress_callback(f"    Outer fold {fold}/{OUTER_CV_SPLITS}...",)
        started_at = perf_counter()
        search = build_inner_search(model, search_n_jobs=search_n_jobs)
        fit_features = features.iloc[fit_index]
        validation_features = features.iloc[validation_index]
        search.fit(fit_features, target[fit_index])
        fitted_pipeline = search.best_estimator_
        training_prediction = fitted_pipeline.predict(fit_features)
        validation_prediction = fitted_pipeline.predict(validation_features)
        validation_score = _model_score(fitted_pipeline, validation_features)
        metrics = _validation_metrics(target[validation_index], validation_prediction, validation_score)
        train_f1 = float(f1_score(target[fit_index], training_prediction, zero_division=0))
        if progress_callback:
            progress_callback(
                f"      best inner F1 {search.best_score_:.3f}; "
                f"outer F1 {metrics['f1']:.3f}; "
                f"{perf_counter() - started_at:.1f}s"
            )
        fold_rows.append(
            {
                "experiment": experiment,
                "model": model,
                "fold": fold,
                **metrics,
                "training_f1": train_f1,
                "validation_f1": metrics["f1"],
                "train_validation_f1_gap": train_f1 - metrics["f1"],
            }
        )
        parameter_rows.append(
            {
                "experiment": experiment,
                "model": model,
                "fold": fold,
                **{key.removeprefix("classifier__"): value for key, value in search.best_params_.items()},
            }
        )
        for index, true_label, prediction, score in zip(
            validation_index,
            target[validation_index],
            validation_prediction,
            validation_score,
            strict=True,
        ):
            oof_rows.append(
                {
                    "participant_id": train_table.iloc[index]["participant_id"],
                    "true_label": int(true_label),
                    "fold": fold,
                    "prediction": int(prediction),
                    "score": float(score),
                }
            )

    fold_metrics = pd.DataFrame(fold_rows)
    oof_predictions = pd.DataFrame(oof_rows).sort_values("participant_id").reset_index(drop=True)
    comparison_row: dict[str, float | int | str] = {
        "experiment": experiment,
        "model": model,
        "feature_count": len(feature_columns),
    }
    for metric in METRIC_COLUMNS:
        comparison_row[f"{metric}_mean"] = float(fold_metrics[metric].mean())
        comparison_row[f"{metric}_std"] = float(fold_metrics[metric].std(ddof=1))
    comparison_row["train_f1_mean"] = float(fold_metrics["training_f1"].mean())
    comparison_row["validation_f1_mean"] = float(fold_metrics["validation_f1"].mean())
    comparison_row["f1_gap_mean"] = float(fold_metrics["train_validation_f1_gap"].mean())
    return AcousticExperimentResult(
        experiment=experiment,
        model=model,
        feature_columns=feature_columns,
        fold_metrics=fold_metrics,
        oof_predictions=oof_predictions,
        selected_hyperparameters=pd.DataFrame(parameter_rows),
        comparison_row=comparison_row,
        oof_confusion_matrix=confusion_matrix(
            oof_predictions["true_label"], oof_predictions["prediction"], labels=[0, 1]
        ),
    )
