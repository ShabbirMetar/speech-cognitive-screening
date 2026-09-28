"""Nested-CV comparison of classical linguistic screening models.

This module is deliberately restricted to the official TRAIN rows.  All
imputation, scaling, and hyperparameter selection happens inside the relevant
outer-fold training partition.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier

from ml.models.baseline_logistic import (
    METRIC_COLUMNS,
    TARGET_COLUMN,
    encode_screening_labels,
    select_feature_columns,
    training_rows,
)


ModelName = Literal["logistic", "linear_svm", "rbf_svm", "random_forest", "xgboost"]
MODEL_NAMES: tuple[ModelName, ...] = (
    "logistic",
    "linear_svm",
    "rbf_svm",
    "random_forest",
    "xgboost",
)
OUTER_CV_SPLITS = 5
INNER_CV_SPLITS = 3
SEARCH_N_JOBS = 2


@dataclass(frozen=True)
class ModelResult:
    """Nested-CV outputs for one model family."""

    model: ModelName
    fold_metrics: pd.DataFrame
    oof_predictions: pd.DataFrame
    selected_hyperparameters: pd.DataFrame
    comparison_row: dict[str, float | str]
    oof_confusion_matrix: np.ndarray


def build_outer_cv() -> StratifiedKFold:
    """Return the fixed five-fold outer estimator-comparison split."""

    return StratifiedKFold(n_splits=OUTER_CV_SPLITS, shuffle=True, random_state=42)


def build_inner_cv() -> StratifiedKFold:
    """Return the fixed three-fold fold-local hyperparameter split."""

    return StratifiedKFold(n_splits=INNER_CV_SPLITS, shuffle=True, random_state=42)


def build_model_pipeline(model: ModelName) -> Pipeline:
    """Build a preprocessing-safe pipeline for the requested estimator family."""

    if model == "logistic":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        )
    if model == "linear_svm":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", LinearSVC(random_state=42, max_iter=10000)),
            ]
        )
    if model == "rbf_svm":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", SVC(kernel="rbf", random_state=42)),
            ]
        )
    if model == "random_forest":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", RandomForestClassifier(random_state=42, n_jobs=1)),
            ]
        )
    if model == "xgboost":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=42,
                        n_jobs=1,
                        verbosity=0,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown model family: {model}")


def parameter_grid(model: ModelName) -> dict[str, list[object]]:
    """Return the intentionally compact model-specific search grid."""

    if model == "logistic":
        return {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__solver": ["liblinear"],
        }
    if model == "linear_svm":
        return {"classifier__C": [0.1, 1.0, 10.0]}
    if model == "rbf_svm":
        return {"classifier__C": [0.5, 1.0, 5.0], "classifier__gamma": ["scale", 0.01]}
    if model == "random_forest":
        return {
            "classifier__n_estimators": [200, 300, 500],
            "classifier__max_depth": [None, 4, 8],
            "classifier__min_samples_leaf": [1, 3, 5],
            "classifier__max_features": ["sqrt", 0.5],
        }
    if model == "xgboost":
        return {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [2, 3, 4],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__subsample": [0.8, 1.0],
            "classifier__colsample_bytree": [0.8, 1.0],
        }
    raise ValueError(f"Unknown model family: {model}")


def build_inner_search(
    model: ModelName, *, search_n_jobs: int = SEARCH_N_JOBS
) -> GridSearchCV | RandomizedSearchCV:
    """Build a compact fold-local F1 search without nested estimator parallelism."""

    common_arguments = {
        "estimator": clone(build_model_pipeline(model)),
        "scoring": "f1",
        "cv": build_inner_cv(),
        "n_jobs": search_n_jobs,
        "refit": True,
        "return_train_score": False,
        "error_score": "raise",
        "verbose": 0,
    }
    if model in ("logistic", "linear_svm", "rbf_svm"):
        return GridSearchCV(param_grid=parameter_grid(model), **common_arguments)
    return RandomizedSearchCV(
        param_distributions=parameter_grid(model),
        n_iter=8,
        random_state=42,
        **common_arguments,
    )


def _specificity(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    true_negative, false_positive, _, _ = confusion_matrix(y_true, y_predicted, labels=[0, 1]).ravel()
    denominator = true_negative + false_positive
    return float(true_negative / denominator) if denominator else float("nan")


def validation_metrics(y_true: np.ndarray, y_predicted: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """Compute the project metrics from one outer validation fold."""

    return {
        "accuracy": float(accuracy_score(y_true, y_predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_predicted)),
        "precision": float(precision_score(y_true, y_predicted, zero_division=0)),
        "recall": float(recall_score(y_true, y_predicted, zero_division=0)),
        "specificity": _specificity(y_true, y_predicted),
        "f1": float(f1_score(y_true, y_predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, score)),
    }


def model_score(fitted_pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return impaired-class probability when available, otherwise decision score."""

    if hasattr(fitted_pipeline, "predict_proba"):
        return np.asarray(fitted_pipeline.predict_proba(features))[:, 1]
    return np.asarray(fitted_pipeline.decision_function(features))


def _compact_parameters(parameters: dict[str, object]) -> dict[str, object]:
    return {key.removeprefix("classifier__"): value for key, value in parameters.items()}


def run_nested_cv_model(
    feature_table: pd.DataFrame,
    model: ModelName,
    *,
    search_n_jobs: int = SEARCH_N_JOBS,
    progress_callback: Callable[[str], None] | None = None,
) -> ModelResult:
    """Tune one model inside each outer TRAIN fold and save complete OOF data."""

    train_table = training_rows(feature_table)
    feature_names = select_feature_columns(train_table, "ALL")
    features = train_table[feature_names].apply(pd.to_numeric, errors="coerce")
    target = encode_screening_labels(train_table[TARGET_COLUMN]).to_numpy()

    outer_cv = build_outer_cv()
    fold_rows: list[dict[str, float | int | str]] = []
    parameter_rows: list[dict[str, object]] = []
    oof_rows: list[dict[str, float | int | str]] = []

    # The outer splitter receives TRAIN data only. TEST never reaches GridSearchCV or an estimator.
    for fold, (fit_index, validation_index) in enumerate(outer_cv.split(features, target), start=1):
        if progress_callback:
            progress_callback(f"  Outer fold {fold}/{OUTER_CV_SPLITS}...")
            progress_callback("    Inner search started...")
        started_at = perf_counter()
        search = build_inner_search(model, search_n_jobs=search_n_jobs)
        fit_features, validation_features = features.iloc[fit_index], features.iloc[validation_index]
        search.fit(fit_features, target[fit_index])
        fitted = search.best_estimator_
        training_prediction = fitted.predict(fit_features)
        validation_prediction = fitted.predict(validation_features)
        validation_score = model_score(fitted, validation_features)
        metrics = validation_metrics(target[validation_index], validation_prediction, validation_score)
        train_f1 = float(f1_score(target[fit_index], training_prediction, zero_division=0))
        if progress_callback:
            progress_callback(f"    Best inner F1: {search.best_score_:.3f}")
            progress_callback(f"    Outer validation F1: {metrics['f1']:.3f}")
            progress_callback(f"    Fold finished in {perf_counter() - started_at:.1f} seconds")
        fold_rows.append(
            {
                "model": model,
                "fold": fold,
                **metrics,
                "training_f1": train_f1,
                "validation_f1": metrics["f1"],
                "train_validation_f1_gap": train_f1 - metrics["f1"],
            }
        )
        parameter_rows.append(
            {"model": model, "fold": fold, **_compact_parameters(search.best_params_)}
        )
        for sample_index, true_label, prediction, score in zip(
            validation_index,
            target[validation_index],
            validation_prediction,
            validation_score,
            strict=True,
        ):
            oof_rows.append(
                {
                    "participant_id": train_table.iloc[sample_index]["participant_id"],
                    "true_label": int(true_label),
                    "fold": fold,
                    "prediction": int(prediction),
                    "score": float(score),
                }
            )

    fold_metrics = pd.DataFrame(fold_rows)
    oof_predictions = pd.DataFrame(oof_rows).sort_values("participant_id").reset_index(drop=True)
    summary: dict[str, float | str] = {"model": model}
    for metric in METRIC_COLUMNS:
        summary[f"{metric}_mean"] = float(fold_metrics[metric].mean())
        summary[f"{metric}_std"] = float(fold_metrics[metric].std(ddof=1))
    summary["train_f1_mean"] = float(fold_metrics["training_f1"].mean())
    summary["validation_f1_mean"] = float(fold_metrics["validation_f1"].mean())
    summary["f1_gap_mean"] = float(fold_metrics["train_validation_f1_gap"].mean())
    return ModelResult(
        model=model,
        fold_metrics=fold_metrics,
        oof_predictions=oof_predictions,
        selected_hyperparameters=pd.DataFrame(parameter_rows),
        comparison_row=summary,
        oof_confusion_matrix=confusion_matrix(
            oof_predictions["true_label"], oof_predictions["prediction"], labels=[0, 1]
        ),
    )


def run_classical_comparison(
    feature_table: pd.DataFrame,
    *,
    search_n_jobs: int = SEARCH_N_JOBS,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[ModelName, ModelResult]:
    """Run each requested classical model using the same TRAIN-only outer folds."""

    return {
        model: run_nested_cv_model(
            feature_table,
            model,
            search_n_jobs=search_n_jobs,
            progress_callback=progress_callback,
        )
        for model in MODEL_NAMES
    }
