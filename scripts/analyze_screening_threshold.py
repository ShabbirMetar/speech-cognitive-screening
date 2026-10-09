"""TRAIN-only OOF calibration diagnostics and operating-threshold analysis.

The frozen model representation is ASR_RATE_PLUS_CTD: 38 ASR linguistic/rate
features and 13 CTD pure acoustic features.  TEST rows are never read.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from ml.asr.asr_linguistic_features import asr_feature_columns
from ml.data.development_guard import assert_train_only
from ml.models.baseline_logistic import build_logistic_pipeline, encode_screening_labels
from ml.models.modality_comparison import build_outer_splits, run_nested_logistic_representation, select_pure_acoustic_feature_columns
from scripts.compare_deployment_representations import asr_rate_feature_columns, validate_asr_rate_provenance


DEPLOYMENT_TRAIN_PATH = PROJECT_ROOT / "artifacts" / "features" / "deployment_features_train.csv"
ASR_LINGUISTIC_TRAIN_PATH = PROJECT_ROOT / "artifacts" / "features" / "asr_linguistic_features_train.csv"
OUTPUT_DIRECTORY = PROJECT_ROOT / "artifacts" / "results" / "screening_threshold"
EXPECTED_DEVELOPMENT_PARTICIPANTS = 320
FROZEN_REPRESENTATION = "ASR_RATE_PLUS_CTD"
THRESHOLDS = np.round(np.arange(0.20, 0.801, 0.01), 2)
SENSITIVITY_TARGETS = (0.70, 0.75, 0.80)


def _align_to_asr(asr_table: pd.DataFrame, deployment_table: pd.DataFrame) -> pd.DataFrame:
    """Validate and align unique TRAIN deployment rows to the ASR table order."""

    asr = assert_train_only(asr_table)
    deployment = assert_train_only(deployment_table)
    if len(asr) != EXPECTED_DEVELOPMENT_PARTICIPANTS or len(deployment) != EXPECTED_DEVELOPMENT_PARTICIPANTS:
        raise AssertionError("Threshold analysis requires exactly 320 TRAIN participants per table.")
    if set(asr["participant_id"]) != set(deployment["participant_id"]):
        raise ValueError("ASR linguistic and deployment TRAIN participant IDs do not match.")
    aligned = deployment.set_index("participant_id").loc[asr["participant_id"]].copy().reset_index()
    for column in ("diagnosis", "screening_label", "Split"):
        if not asr[column].equals(aligned[column]):
            raise ValueError(f"ASR linguistic and deployment values disagree for {column!r}.")
    return assert_train_only(aligned)


def frozen_feature_columns(asr_table: pd.DataFrame, deployment_table: pd.DataFrame) -> list[str]:
    """Return exactly the frozen 38 ASR-rate plus 13 CTD acoustic features."""

    asr_linguistic = asr_feature_columns(asr_table)
    asr_rates = asr_rate_feature_columns(deployment_table)
    ctd_acoustic = select_pure_acoustic_feature_columns(deployment_table, "CTD")
    columns = [*asr_linguistic, *asr_rates, *ctd_acoustic]
    if (len(asr_linguistic), len(asr_rates), len(ctd_acoustic), len(columns)) != (32, 6, 13, 51):
        raise AssertionError("Frozen representation must contain 32 ASR linguistic, 6 rate, and 13 CTD acoustic features.")
    if len(set(columns)) != len(columns):
        raise AssertionError("Frozen representation contains duplicate features.")
    if any("pause_annotation" in column for column in columns):
        raise AssertionError("Frozen deployment representation must not include manual pause annotations.")
    return columns


def _specificity(target: np.ndarray, prediction: np.ndarray) -> float:
    negatives = target == 0
    denominator = int(negatives.sum())
    return float(((prediction == 0) & negatives).sum() / denominator) if denominator else float("nan")


def threshold_metrics(target: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    """Calculate fixed-threshold screening metrics from OOF classifier scores."""

    prediction = (np.asarray(probabilities) >= threshold).astype(int)
    sensitivity = float(recall_score(target, prediction, zero_division=0))
    specificity = _specificity(target, prediction)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "sensitivity": sensitivity,
        "recall": sensitivity,
        "specificity": specificity,
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "youden_j": float(sensitivity + specificity - 1.0),
    }


def build_threshold_sweep(target: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    """Evaluate every prespecified threshold from 0.20 through 0.80 inclusive."""

    return pd.DataFrame([threshold_metrics(target, probabilities, threshold) for threshold in THRESHOLDS])


def _deterministic_best(metrics: pd.DataFrame, metric: str) -> pd.Series:
    """Maximize one metric, resolving ties by distance to 0.5 and then lower threshold."""

    ordered = metrics.assign(
        _distance_from_half=(metrics["threshold"] - 0.5).abs()
    ).sort_values(
        [metric, "_distance_from_half", "threshold"], ascending=[False, True, True]
    )
    return ordered.iloc[0]


def select_threshold_candidates(metrics: pd.DataFrame) -> pd.DataFrame:
    """Apply pre-specified objective rules and descriptive sensitivity constraints."""

    candidates: list[dict[str, object]] = []
    rules = (
        ("max_f1", "f1"),
        ("max_balanced_accuracy", "balanced_accuracy"),
        ("max_youden_j", "youden_j"),
    )
    for rule, metric in rules:
        row = _deterministic_best(metrics, metric).to_dict()
        candidates.append({"rule": rule, "selection_metric": metric, **row})
    for target in SENSITIVITY_TARGETS:
        eligible = metrics.loc[metrics["sensitivity"] >= target]
        if eligible.empty:
            continue
        # Descriptive view: highest specificity among thresholds that attain the target.
        row = _deterministic_best(eligible, "specificity").to_dict()
        candidates.append(
            {
                "rule": f"sensitivity_at_least_{target:.2f}",
                "selection_metric": "max_specificity_subject_to_sensitivity",
                **row,
            }
        )
    return pd.DataFrame(candidates)


def probability_metrics(target: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Calculate calibration and discrimination diagnostics for OOF classifier scores."""

    scores = np.asarray(probabilities, dtype=float)
    return {
        "brier_score": float(brier_score_loss(target, scores)),
        "log_loss": float(log_loss(target, scores, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(target, scores)),
    }


def sigmoid_calibrated_oof(
    feature_table: pd.DataFrame,
    feature_columns: Sequence[str],
    outer_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    selected_hyperparameters: pd.DataFrame,
) -> pd.DataFrame:
    """Cross-fit sigmoid calibration inside each outer TRAIN partition only."""

    train = assert_train_only(feature_table)
    features = train[list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    target = encode_screening_labels(train["screening_label"]).to_numpy()
    selected_c = selected_hyperparameters.set_index("fold")["C"]
    rows: list[dict[str, object]] = []
    for fold, (fit_index, validation_index) in enumerate(outer_splits, start=1):
        c_value = float(selected_c.loc[fold])
        base_estimator = clone(build_logistic_pipeline()).set_params(classifier__C=c_value)
        calibration_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        calibrated = CalibratedClassifierCV(
            estimator=base_estimator,
            method="sigmoid",
            cv=calibration_cv,
        )
        calibrated.fit(features.iloc[fit_index], target[fit_index])
        probabilities = calibrated.predict_proba(features.iloc[validation_index])[:, 1]
        for index, score in zip(validation_index, probabilities, strict=True):
            rows.append(
                {
                    "participant_id": train.iloc[index]["participant_id"],
                    "fold": fold,
                    "sigmoid_probability": float(score),
                }
            )
    calibrated_oof = pd.DataFrame(rows).sort_values("participant_id").reset_index(drop=True)
    if len(calibrated_oof) != len(train) or calibrated_oof["participant_id"].duplicated().any():
        raise AssertionError("Sigmoid calibration must produce exactly one OOF score per TRAIN participant.")
    return calibrated_oof


def _plot_calibration(target: np.ndarray, raw: np.ndarray, sigmoid: np.ndarray, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="#666666", label="Perfect calibration")
    for scores, label, color, marker in (
        (raw, "Raw Logistic Regression", "#4C78A8", "o"),
        (sigmoid, "Sigmoid-calibrated", "#DD8452", "s"),
    ):
        observed, predicted = calibration_curve(target, scores, n_bins=10, strategy="quantile")
        axis.plot(predicted, observed, marker=marker, linewidth=2, color=color, label=label)
    axis.set_title("TRAIN OOF reliability diagram")
    axis.set_xlabel("Mean classifier probability-like score")
    axis.set_ylabel("Observed impaired-label frequency")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend(frameon=True)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_threshold_tradeoff(metrics: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for column, label, color, linestyle in (
        ("sensitivity", "Sensitivity", "#4C78A8", "-"),
        ("specificity", "Specificity", "#DD8452", "--"),
        ("f1", "F1", "#55A868", "-"),
    ):
        axis.plot(metrics["threshold"], metrics[column], label=label, color=color, linestyle=linestyle, linewidth=2)
    axis.set_title("TRAIN OOF threshold trade-offs")
    axis.set_xlabel("Classifier probability-like score threshold")
    axis.set_ylabel("Metric value")
    axis.set_xlim(0.20, 0.80)
    axis.set_ylim(0, 1)
    axis.legend(frameon=True)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_score_distribution(oof: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for label, color in ((0, "#4C78A8"), (1, "#DD8452")):
        values = oof.loc[oof["true_label"].eq(label), "raw_probability"]
        name = "Healthy-like label" if label == 0 else "Possible impairment-like label"
        axis.hist(values, bins=np.linspace(0, 1, 21), alpha=0.55, color=color, label=name, edgecolor="white")
        axis.axvline(values.median(), color=color, linestyle="--", linewidth=1.5)
    axis.set_title("TRAIN OOF classifier-score distributions")
    axis.set_xlabel("Raw classifier probability-like score")
    axis.set_ylabel("Participants")
    axis.set_xlim(0, 1)
    axis.legend(frameon=True)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _score_distribution_summary(oof: pd.DataFrame) -> pd.DataFrame:
    labels = {0: "Healthy-like label", 1: "Possible impairment-like label"}
    return (
        oof.groupby("true_label")["raw_probability"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
        .assign(label=lambda frame: frame["true_label"].map(labels))
        [["true_label", "label", "count", "mean", "std", "min", "median", "max"]]
    )


def main() -> int:
    for path in (ASR_LINGUISTIC_TRAIN_PATH, DEPLOYMENT_TRAIN_PATH):
        if not path.is_file():
            print(f"Required TRAIN feature table not found: {path}")
            return 1
    asr = assert_train_only(pd.read_csv(ASR_LINGUISTIC_TRAIN_PATH))
    deployment = _align_to_asr(asr, pd.read_csv(DEPLOYMENT_TRAIN_PATH))
    features = frozen_feature_columns(asr, deployment)
    validate_asr_rate_provenance(asr, deployment)
    for column in asr_feature_columns(asr):
        if not asr[column].equals(deployment[column]):
            raise ValueError(f"Deployment ASR linguistic value mismatch for {column}.")

    target = encode_screening_labels(deployment["screening_label"]).to_numpy()
    outer_splits = build_outer_splits(target)
    print("Screening Threshold Analysis")
    print("============================")
    print(f"Development participants: {len(deployment)}")
    print("Official TEST participants used: 0")
    print(f"Frozen representation: {FROZEN_REPRESENTATION} ({len(features)} features)")

    raw_result = run_nested_logistic_representation(
        deployment, FROZEN_REPRESENTATION, features, outer_splits
    )
    raw_oof = raw_result.oof_predictions.rename(columns={"score": "raw_probability", "prediction": "raw_prediction"})
    sigmoid_oof = sigmoid_calibrated_oof(
        deployment, features, outer_splits, raw_result.selected_hyperparameters
    )
    oof = raw_oof.merge(sigmoid_oof, on=["participant_id", "fold"], how="inner", validate="one_to_one")
    if len(oof) != EXPECTED_DEVELOPMENT_PARTICIPANTS or not oof["true_label"].isin([0, 1]).all():
        raise AssertionError("Expected one valid raw and sigmoid OOF score per TRAIN participant.")
    if oof[["raw_probability", "sigmoid_probability"]].isna().any().any():
        raise AssertionError("OOF probability outputs must be complete.")
    oof["sigmoid_prediction"] = (oof["sigmoid_probability"] >= 0.5).astype(int)
    oof_target = oof["true_label"].to_numpy()

    raw_metrics = probability_metrics(oof_target, oof["raw_probability"].to_numpy())
    sigmoid_metrics = probability_metrics(oof_target, oof["sigmoid_probability"].to_numpy())
    calibration_metrics = pd.DataFrame(
        [
            {"score_source": "raw_logistic", "calibration_method": "none", **raw_metrics},
            {"score_source": "sigmoid_logistic", "calibration_method": "sigmoid", **sigmoid_metrics},
        ]
    )
    # Raw scores remain the frozen-model operating-score source; sigmoid is a
    # TRAIN-only diagnostic comparator rather than an automatic model change.
    sweep = build_threshold_sweep(oof_target, oof["raw_probability"].to_numpy())
    candidates = select_threshold_candidates(sweep)
    candidates.insert(0, "score_source", "raw_logistic")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    oof.to_csv(OUTPUT_DIRECTORY / "oof_probabilities.csv", index=False)
    sweep.to_csv(OUTPUT_DIRECTORY / "threshold_metrics.csv", index=False)
    candidates.to_csv(OUTPUT_DIRECTORY / "threshold_candidates.csv", index=False)
    calibration_metrics.to_csv(OUTPUT_DIRECTORY / "calibration_metrics.csv", index=False)
    _score_distribution_summary(oof).to_csv(OUTPUT_DIRECTORY / "oof_score_distribution_summary.csv", index=False)
    raw_result.selected_hyperparameters.to_csv(OUTPUT_DIRECTORY / "selected_hyperparameters.csv", index=False)
    _plot_calibration(oof_target, oof["raw_probability"].to_numpy(), oof["sigmoid_probability"].to_numpy(), OUTPUT_DIRECTORY / "calibration_curve.png")
    _plot_threshold_tradeoff(sweep, OUTPUT_DIRECTORY / "threshold_tradeoff.png")
    _plot_score_distribution(oof, OUTPUT_DIRECTORY / "oof_score_distribution.png")

    print("\nRaw Logistic")
    print(f"Brier: {raw_metrics['brier_score']:.4f}")
    print(f"Log loss: {raw_metrics['log_loss']:.4f}")
    print(f"ROC-AUC: {raw_metrics['roc_auc']:.4f}")
    print("\nCalibration")
    print(
        f"raw: Brier {raw_metrics['brier_score']:.4f}; log loss {raw_metrics['log_loss']:.4f}; "
        f"ROC-AUC {raw_metrics['roc_auc']:.4f}"
    )
    print(
        f"sigmoid: Brier {sigmoid_metrics['brier_score']:.4f}; log loss {sigmoid_metrics['log_loss']:.4f}; "
        f"ROC-AUC {sigmoid_metrics['roc_auc']:.4f}"
    )
    print("\nCandidate operating thresholds")
    print("Rule                          Threshold  Sensitivity  Specificity  F1     BalAcc")
    for row in candidates.itertuples(index=False):
        print(
            f"{row.rule:<29} {row.threshold:>8.2f}  {row.sensitivity:>11.3f}  "
            f"{row.specificity:>11.3f}  {row.f1:>5.3f}  {row.balanced_accuracy:>6.3f}"
        )
    print("\nRaw score source retained for threshold analysis; sigmoid is reported as a TRAIN-only calibration diagnostic.")
    print(f"Saved TRAIN-only threshold artifacts to: {OUTPUT_DIRECTORY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
