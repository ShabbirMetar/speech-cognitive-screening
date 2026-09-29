"""TRAIN-only pure-acoustic and linguistic/acoustic fusion comparison helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
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

from ml.data.development_guard import assert_train_only
from ml.models.baseline_logistic import (
    METRIC_COLUMNS,
    TARGET_COLUMN,
    build_logistic_pipeline,
    encode_screening_labels,
    select_feature_columns,
)


OUTER_CV_SPLITS = 5
INNER_CV_SPLITS = 3
SEARCH_N_JOBS = 2
TASK_PREFIXES = {
    "SFT": ("sft_",),
    "PFT": ("pft_",),
    "CTD": ("ctd_",),
    "ALL": ("sft_", "pft_", "ctd_"),
}
PURE_ACOUSTIC_SUFFIXES: tuple[str, ...] = (
    "audio_duration_seconds",
    "voiced_duration_seconds",
    "silence_duration_seconds",
    "silence_ratio",
    "pause_count",
    "pause_total_seconds",
    "pause_mean_seconds",
    "pause_max_seconds",
    "f0_mean_hz",
    "f0_std_hz",
    "f0_range_hz",
    "rms_mean",
    "rms_std",
)
FORBIDDEN_PREDICTIVE_COLUMNS = {
    "participant_id",
    "diagnosis",
    "screening_label",
    "Split",
    "age",
    "gender",
    "MMSE",
}
IDENTITY_COLUMNS = ("diagnosis", "screening_label", "Split")


@dataclass(frozen=True)
class ModalityResult:
    """Outer-fold evaluation outputs for one fixed Logistic Regression representation."""

    representation: str
    feature_columns: list[str]
    fold_metrics: pd.DataFrame
    oof_predictions: pd.DataFrame
    selected_hyperparameters: pd.DataFrame
    comparison_row: dict[str, float | int | str]
    oof_confusion_matrix: np.ndarray


def select_pure_acoustic_feature_columns(feature_table: pd.DataFrame, task: str) -> list[str]:
    """Select exactly 13 waveform-only features per task, without word-rate fields."""

    try:
        prefixes = TASK_PREFIXES[task]
    except KeyError as error:
        raise ValueError(f"Unsupported pure acoustic task: {task}") from error
    expected = [
        f"{prefix}{suffix}" for prefix in prefixes for suffix in PURE_ACOUSTIC_SUFFIXES
    ]
    missing = [column for column in expected if column not in feature_table.columns]
    if missing:
        raise ValueError(f"Missing expected pure acoustic columns for {task}: {missing}")
    return expected


def merge_train_feature_tables(
    linguistic_table: pd.DataFrame, acoustic_table: pd.DataFrame
) -> pd.DataFrame:
    """Strictly one-to-one merge aligned TRAIN tables after metadata agreement checks."""

    linguistic = assert_train_only(linguistic_table)
    acoustic = assert_train_only(acoustic_table)
    linguistic_ids = set(linguistic["participant_id"])
    acoustic_ids = set(acoustic["participant_id"])
    if linguistic_ids != acoustic_ids:
        raise ValueError("Linguistic and acoustic TRAIN participant IDs do not match.")

    acoustic_index = acoustic.set_index("participant_id")
    for column in IDENTITY_COLUMNS:
        if column not in linguistic.columns or column not in acoustic_index.columns:
            raise ValueError(f"Both TRAIN feature tables must contain {column!r}.")
        expected = linguistic.set_index("participant_id")[column]
        observed = acoustic_index.loc[expected.index, column]
        if not expected.equals(observed):
            raise ValueError(f"Linguistic and acoustic TRAIN values disagree for {column!r}.")

    acoustic_feature_columns = [
        column for column in acoustic.columns if column not in {"participant_id", *IDENTITY_COLUMNS}
    ]
    duplicate_feature_names = set(linguistic.columns).intersection(acoustic_feature_columns)
    if duplicate_feature_names:
        raise ValueError(
            "Linguistic and acoustic feature schemas overlap unexpectedly: "
            f"{sorted(duplicate_feature_names)}"
        )
    merged = linguistic.merge(
        acoustic[["participant_id", *acoustic_feature_columns]],
        on="participant_id",
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if len(merged) != len(linguistic):
        raise AssertionError("Strict TRAIN feature merge changed the participant-row count.")
    return assert_train_only(merged)


def build_outer_splits(target: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Precompute one shared five-fold outer split for every representation."""

    placeholder = np.zeros((len(target), 1))
    splitter = StratifiedKFold(n_splits=OUTER_CV_SPLITS, shuffle=True, random_state=42)
    return tuple((fit.copy(), validation.copy()) for fit, validation in splitter.split(placeholder, target))


def _build_inner_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=INNER_CV_SPLITS, shuffle=True, random_state=42)


def _validate_predictive_columns(feature_table: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    selected = list(columns)
    if not selected:
        raise ValueError("A representation must contain at least one predictive feature.")
    if len(selected) != len(set(selected)):
        raise ValueError("Predictive feature list contains duplicate columns.")
    missing = set(selected).difference(feature_table.columns)
    if missing:
        raise ValueError(f"Predictive feature list contains unknown columns: {sorted(missing)}")
    forbidden = set(selected).intersection(FORBIDDEN_PREDICTIVE_COLUMNS)
    if forbidden:
        raise ValueError(f"Metadata/label columns cannot be predictive features: {sorted(forbidden)}")
    return selected


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


def run_nested_logistic_representation(
    feature_table: pd.DataFrame,
    representation: str,
    feature_columns: Sequence[str],
    outer_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    search_n_jobs: int = SEARCH_N_JOBS,
) -> ModalityResult:
    """Evaluate a fixed representation with fold-local preprocessing and inner F1 tuning."""

    train_table = assert_train_only(feature_table)
    selected = _validate_predictive_columns(train_table, feature_columns)
    if len(outer_splits) != OUTER_CV_SPLITS:
        raise ValueError(f"Exactly {OUTER_CV_SPLITS} outer splits are required.")
    features = train_table[selected].apply(pd.to_numeric, errors="coerce")
    target = encode_screening_labels(train_table[TARGET_COLUMN]).to_numpy()
    fold_rows: list[dict[str, float | int | str]] = []
    oof_rows: list[dict[str, float | int | str]] = []
    parameter_rows: list[dict[str, object]] = []

    for fold, (fit_index, validation_index) in enumerate(outer_splits, start=1):
        search = GridSearchCV(
            estimator=clone(build_logistic_pipeline()),
            param_grid={"classifier__C": [0.1, 1.0, 10.0]},
            scoring="f1",
            cv=_build_inner_cv(),
            n_jobs=search_n_jobs,
            refit=True,
            return_train_score=False,
            error_score="raise",
            verbose=0,
        )
        fit_features = features.iloc[fit_index]
        validation_features = features.iloc[validation_index]
        search.fit(fit_features, target[fit_index])
        fitted = search.best_estimator_
        training_prediction = fitted.predict(fit_features)
        validation_prediction = fitted.predict(validation_features)
        validation_score = fitted.predict_proba(validation_features)[:, 1]
        metrics = _validation_metrics(target[validation_index], validation_prediction, validation_score)
        training_f1 = float(f1_score(target[fit_index], training_prediction, zero_division=0))
        fold_rows.append(
            {
                "representation": representation,
                "fold": fold,
                **metrics,
                "training_f1": training_f1,
                "validation_f1": metrics["f1"],
                "train_validation_f1_gap": training_f1 - metrics["f1"],
            }
        )
        parameter_rows.append(
            {
                "representation": representation,
                "fold": fold,
                "C": search.best_params_["classifier__C"],
                "inner_f1": float(search.best_score_),
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
        "representation": representation,
        "feature_count": len(selected),
    }
    for metric in METRIC_COLUMNS:
        comparison_row[f"{metric}_mean"] = float(fold_metrics[metric].mean())
        comparison_row[f"{metric}_std"] = float(fold_metrics[metric].std(ddof=1))
    comparison_row["train_f1_mean"] = float(fold_metrics["training_f1"].mean())
    comparison_row["validation_f1_mean"] = float(fold_metrics["validation_f1"].mean())
    comparison_row["f1_gap_mean"] = float(fold_metrics["train_validation_f1_gap"].mean())
    return ModalityResult(
        representation=representation,
        feature_columns=selected,
        fold_metrics=fold_metrics,
        oof_predictions=oof_predictions,
        selected_hyperparameters=pd.DataFrame(parameter_rows),
        comparison_row=comparison_row,
        oof_confusion_matrix=confusion_matrix(
            oof_predictions["true_label"], oof_predictions["prediction"], labels=[0, 1]
        ),
    )


def linguistic_feature_columns(feature_table: pd.DataFrame) -> list[str]:
    """Return the existing complete linguistic schema without metadata fields."""

    return select_feature_columns(feature_table, "ALL")
