"""Compare compact early fusion with nested probability-level late fusion on TRAIN only."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import sys
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
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
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from ml.asr.asr_linguistic_features import asr_feature_columns
from ml.data.development_guard import assert_train_only
from ml.models.baseline_logistic import METRIC_COLUMNS, build_logistic_pipeline, encode_screening_labels
from ml.models.modality_comparison import (
    INNER_CV_SPLITS,
    ModalityResult,
    build_outer_splits,
    run_nested_logistic_representation,
    select_pure_acoustic_feature_columns,
)
from scripts.compare_deployment_representations import (
    asr_rate_feature_columns,
    validate_asr_rate_provenance,
)


ASR_LINGUISTIC_TRAIN_PATH = PROJECT_ROOT / "artifacts" / "features" / "asr_linguistic_features_train.csv"
DEPLOYMENT_TRAIN_PATH = PROJECT_ROOT / "artifacts" / "features" / "deployment_features_train.csv"
DEPLOYMENT_REFERENCE_PATH = (
    PROJECT_ROOT / "artifacts" / "results" / "deployment_representation_comparison" / "representation_summary.csv"
)
OUTPUT_DIRECTORY = PROJECT_ROOT / "artifacts" / "results" / "fusion_strategy_comparison"
EXPECTED_DEVELOPMENT_PARTICIPANTS = 320
C_VALUES = (0.1, 1.0, 10.0)
ALPHA_VALUES = (0.6, 0.7, 0.8, 0.9)
LATE_FUSION_NAME = "LATE_FUSION_ASR_CTD"


@dataclass(frozen=True)
class LateFusionResult:
    """Nested-CV outputs for separate ASR-rate and CTD-acoustic probability fusion."""

    representation: str
    feature_columns: list[str]
    fold_metrics: pd.DataFrame
    oof_predictions: pd.DataFrame
    selected_hyperparameters: pd.DataFrame
    comparison_row: dict[str, float | int | str]
    oof_confusion_matrix: np.ndarray


def _align_to_asr(asr_table: pd.DataFrame, deployment_table: pd.DataFrame) -> pd.DataFrame:
    """Require matching unique TRAIN identities and align deployment rows to ASR order."""

    asr = assert_train_only(asr_table)
    deployment = assert_train_only(deployment_table)
    if len(asr) != EXPECTED_DEVELOPMENT_PARTICIPANTS or len(deployment) != EXPECTED_DEVELOPMENT_PARTICIPANTS:
        raise AssertionError("Fusion comparison requires exactly 320 TRAIN participants per table.")
    if set(asr["participant_id"]) != set(deployment["participant_id"]):
        raise ValueError("ASR linguistic and deployment TRAIN participant IDs do not match.")
    aligned = deployment.set_index("participant_id").loc[asr["participant_id"]].copy().reset_index()
    for column in ("diagnosis", "screening_label", "Split"):
        if not asr[column].equals(aligned[column]):
            raise ValueError(f"ASR linguistic and deployment TRAIN values disagree for {column!r}.")
    return assert_train_only(aligned)


def late_fusion_probabilities(
    asr_probabilities: np.ndarray, ctd_probabilities: np.ndarray, alpha: float
) -> np.ndarray:
    """Blend impaired-class probabilities while retaining ASR-rate as the larger contribution."""

    if alpha not in ALPHA_VALUES:
        raise ValueError(f"alpha must be one of {ALPHA_VALUES}; found {alpha}.")
    asr_values = np.asarray(asr_probabilities, dtype=float)
    ctd_values = np.asarray(ctd_probabilities, dtype=float)
    if asr_values.shape != ctd_values.shape:
        raise ValueError("ASR-rate and CTD probability arrays must have the same shape.")
    return alpha * asr_values + (1.0 - alpha) * ctd_values


def _inner_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=INNER_CV_SPLITS, shuffle=True, random_state=42)


def _pipeline_with_c(value: float):
    return clone(build_logistic_pipeline()).set_params(classifier__C=value)


def _specificity(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    true_negative, false_positive, _, _ = confusion_matrix(y_true, y_predicted, labels=[0, 1]).ravel()
    return float(true_negative / (true_negative + false_positive)) if true_negative + false_positive else float("nan")


def _metrics(y_true: np.ndarray, prediction: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": _specificity(y_true, prediction),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, score)),
    }


def select_inner_late_fusion_parameters(
    asr_features: pd.DataFrame, ctd_features: pd.DataFrame, target: np.ndarray
) -> tuple[float, float, float, float]:
    """Select ASR C, CTD C, and alpha using only inner validation predictions."""

    candidate_scores = {
        (asr_c, ctd_c, alpha): []
        for asr_c, ctd_c, alpha in product(C_VALUES, C_VALUES, ALPHA_VALUES)
    }
    for fit_index, validation_index in _inner_cv().split(asr_features, target):
        asr_scores: dict[float, np.ndarray] = {}
        ctd_scores: dict[float, np.ndarray] = {}
        for value in C_VALUES:
            asr_model = _pipeline_with_c(value).fit(asr_features.iloc[fit_index], target[fit_index])
            ctd_model = _pipeline_with_c(value).fit(ctd_features.iloc[fit_index], target[fit_index])
            asr_scores[value] = asr_model.predict_proba(asr_features.iloc[validation_index])[:, 1]
            ctd_scores[value] = ctd_model.predict_proba(ctd_features.iloc[validation_index])[:, 1]
        for asr_c, ctd_c, alpha in candidate_scores:
            probability = late_fusion_probabilities(asr_scores[asr_c], ctd_scores[ctd_c], alpha)
            prediction = (probability >= 0.5).astype(int)
            candidate_scores[(asr_c, ctd_c, alpha)].append(
                float(f1_score(target[validation_index], prediction, zero_division=0))
            )
    means = {parameters: float(np.mean(scores)) for parameters, scores in candidate_scores.items()}
    best_parameters = max(means, key=means.get)
    return (*best_parameters, means[best_parameters])


def run_nested_late_fusion(
    feature_table: pd.DataFrame,
    asr_rate_columns: Sequence[str],
    ctd_acoustic_columns: Sequence[str],
    outer_splits: Sequence[tuple[np.ndarray, np.ndarray]],
) -> LateFusionResult:
    """Evaluate probability fusion with all parameter decisions isolated inside outer TRAIN folds."""

    train = assert_train_only(feature_table)
    asr_features = train[list(asr_rate_columns)].apply(pd.to_numeric, errors="coerce")
    ctd_features = train[list(ctd_acoustic_columns)].apply(pd.to_numeric, errors="coerce")
    target = encode_screening_labels(train["screening_label"]).to_numpy()
    fold_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    oof_rows: list[dict[str, object]] = []

    for fold, (fit_index, validation_index) in enumerate(outer_splits, start=1):
        asr_c, ctd_c, alpha, inner_f1 = select_inner_late_fusion_parameters(
            asr_features.iloc[fit_index], ctd_features.iloc[fit_index], target[fit_index]
        )
        asr_model = _pipeline_with_c(asr_c).fit(asr_features.iloc[fit_index], target[fit_index])
        ctd_model = _pipeline_with_c(ctd_c).fit(ctd_features.iloc[fit_index], target[fit_index])
        train_asr_probability = asr_model.predict_proba(asr_features.iloc[fit_index])[:, 1]
        train_ctd_probability = ctd_model.predict_proba(ctd_features.iloc[fit_index])[:, 1]
        validation_asr_probability = asr_model.predict_proba(asr_features.iloc[validation_index])[:, 1]
        validation_ctd_probability = ctd_model.predict_proba(ctd_features.iloc[validation_index])[:, 1]
        train_probability = late_fusion_probabilities(train_asr_probability, train_ctd_probability, alpha)
        validation_probability = late_fusion_probabilities(
            validation_asr_probability, validation_ctd_probability, alpha
        )
        train_prediction = (train_probability >= 0.5).astype(int)
        validation_prediction = (validation_probability >= 0.5).astype(int)
        metrics = _metrics(target[validation_index], validation_prediction, validation_probability)
        training_f1 = float(f1_score(target[fit_index], train_prediction, zero_division=0))
        fold_rows.append(
            {
                "representation": LATE_FUSION_NAME,
                "fold": fold,
                **metrics,
                "training_f1": training_f1,
                "validation_f1": metrics["f1"],
                "train_validation_f1_gap": training_f1 - metrics["f1"],
            }
        )
        parameter_rows.append(
            {
                "representation": LATE_FUSION_NAME,
                "fold": fold,
                "asr_rate_C": asr_c,
                "ctd_acoustic_C": ctd_c,
                "alpha": alpha,
                "inner_f1": inner_f1,
            }
        )
        for index, true_label, prediction, final_probability, asr_probability, ctd_probability in zip(
            validation_index,
            target[validation_index],
            validation_prediction,
            validation_probability,
            validation_asr_probability,
            validation_ctd_probability,
            strict=True,
        ):
            oof_rows.append(
                {
                    "participant_id": train.iloc[index]["participant_id"],
                    "true_label": int(true_label),
                    "fold": fold,
                    "prediction": int(prediction),
                    "score": float(final_probability),
                    "asr_rate_probability": float(asr_probability),
                    "ctd_acoustic_probability": float(ctd_probability),
                }
            )

    fold_metrics = pd.DataFrame(fold_rows)
    oof_predictions = pd.DataFrame(oof_rows).sort_values("participant_id").reset_index(drop=True)
    summary: dict[str, float | int | str] = {
        "representation": LATE_FUSION_NAME,
        "feature_count": len(asr_rate_columns) + len(ctd_acoustic_columns),
        "feature_mode": "late probability fusion (38 + 13 separate)",
    }
    for metric in METRIC_COLUMNS:
        summary[f"{metric}_mean"] = float(fold_metrics[metric].mean())
        summary[f"{metric}_std"] = float(fold_metrics[metric].std(ddof=1))
    summary["train_f1_mean"] = float(fold_metrics["training_f1"].mean())
    summary["validation_f1_mean"] = float(fold_metrics["validation_f1"].mean())
    summary["f1_gap_mean"] = float(fold_metrics["train_validation_f1_gap"].mean())
    return LateFusionResult(
        representation=LATE_FUSION_NAME,
        feature_columns=[*asr_rate_columns, *ctd_acoustic_columns],
        fold_metrics=fold_metrics,
        oof_predictions=oof_predictions,
        selected_hyperparameters=pd.DataFrame(parameter_rows),
        comparison_row=summary,
        oof_confusion_matrix=confusion_matrix(
            oof_predictions["true_label"], oof_predictions["prediction"], labels=[0, 1]
        ),
    )


def _paired_deltas(results: Iterable[ModalityResult | LateFusionResult]) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_name = {result.representation: result for result in results}
    comparisons = (
        ("compact early fusion - ASR_RATE", "ASR_RATE_PLUS_CTD", "ASR_RATE"),
        ("late fusion - ASR_RATE", LATE_FUSION_NAME, "ASR_RATE"),
        ("late fusion - compact early fusion", LATE_FUSION_NAME, "ASR_RATE_PLUS_CTD"),
    )
    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for name, minuend_name, subtrahend_name in comparisons:
        minuend = by_name[minuend_name].fold_metrics.set_index("fold")["f1"]
        subtrahend = by_name[subtrahend_name].fold_metrics.set_index("fold")["f1"]
        if not minuend.index.equals(subtrahend.index):
            raise AssertionError("Paired fusion folds do not align.")
        delta = minuend - subtrahend
        rows.extend(
            {
                "comparison": name,
                "minuend_representation": minuend_name,
                "subtrahend_representation": subtrahend_name,
                "fold": int(fold),
                "f1_delta": float(value),
            }
            for fold, value in delta.items()
        )
        summary_rows.append(
            {
                "comparison": name,
                "minuend_representation": minuend_name,
                "subtrahend_representation": subtrahend_name,
                "f1_delta_mean": float(delta.mean()),
                "f1_delta_std": float(delta.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def _build_oof_table(
    feature_table: pd.DataFrame, results: Iterable[ModalityResult | LateFusionResult]
) -> pd.DataFrame:
    oof = feature_table[["participant_id", "screening_label"]].copy()
    oof["true_label"] = oof["screening_label"].map({"Healthy": 0, "Impaired": 1}).astype(int)
    oof = oof.drop(columns="screening_label")
    for result in results:
        prefix = result.representation.casefold()
        values = result.oof_predictions.set_index("participant_id")
        selected = ["fold", "prediction", "score"]
        if result.representation == LATE_FUSION_NAME:
            selected.extend(["asr_rate_probability", "ctd_acoustic_probability"])
        oof = oof.join(values[selected].rename(columns={column: f"{prefix}_{column}" for column in selected}), on="participant_id")
    return oof


def _plot_metric(summary: pd.DataFrame, metric: str, output_path: Path) -> None:
    labels = summary["representation"].str.replace("_", "\n", regex=False).tolist()
    means = summary[f"{metric}_mean"].to_numpy(dtype=float)
    stds = summary[f"{metric}_std"].to_numpy(dtype=float)
    colors = ["#4C78A8"] * (len(summary) - 1) + ["#9E9E9E"]
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(range(len(summary)), means, yerr=stds, capsize=4, color=colors)
    axis.set_xticks(range(len(summary)), labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel(metric.replace("_", " ").upper())
    axis.set_title(f"TRAIN-only fusion strategy comparison: {metric.replace('_', ' ').upper()}")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _reference_row() -> dict[str, object]:
    reference = pd.read_csv(DEPLOYMENT_REFERENCE_PATH)
    row = reference.loc[reference["representation"].eq("DEPLOYMENT_MULTIMODAL")]
    if len(row) != 1:
        raise ValueError("Expected exactly one previous DEPLOYMENT_MULTIMODAL reference row.")
    payload = row.iloc[0].to_dict()
    payload["representation"] = "DEPLOYMENT_MULTIMODAL_REFERENCE"
    payload["feature_mode"] = "77-feature early fusion reference (not rerun)"
    payload["evaluation_mode"] = "reference_only"
    return payload


def main() -> int:
    for path in (ASR_LINGUISTIC_TRAIN_PATH, DEPLOYMENT_TRAIN_PATH, DEPLOYMENT_REFERENCE_PATH):
        if not path.is_file():
            print(f"Required TRAIN artifact not found: {path}")
            return 1
    asr = assert_train_only(pd.read_csv(ASR_LINGUISTIC_TRAIN_PATH))
    deployment = _align_to_asr(asr, pd.read_csv(DEPLOYMENT_TRAIN_PATH))
    asr_linguistic = asr_feature_columns(asr)
    asr_rates = asr_rate_feature_columns(deployment)
    asr_rate = [*asr_linguistic, *asr_rates]
    ctd_acoustic = select_pure_acoustic_feature_columns(deployment, "CTD")
    compact_early = [*asr_rate, *ctd_acoustic]
    if (len(asr_rate), len(ctd_acoustic), len(compact_early)) != (38, 13, 51):
        raise AssertionError("Expected 38 ASR-rate, 13 CTD acoustic, and 51 compact early-fusion features.")
    if len(set(compact_early)) != len(compact_early):
        raise AssertionError("Compact early fusion contains duplicate features.")
    if any("pause_annotation" in column for column in compact_early):
        raise AssertionError("Fusion comparison must not use manual pause annotations.")
    validate_asr_rate_provenance(asr, deployment)
    for column in asr_linguistic:
        if not asr[column].equals(deployment[column]):
            raise ValueError(f"Deployment ASR linguistic value mismatch for {column}.")

    target = encode_screening_labels(deployment["screening_label"]).to_numpy()
    outer_splits = build_outer_splits(target)
    representations = {
        "ASR_RATE": asr_rate,
        "CTD_ACOUSTIC": ctd_acoustic,
        "ASR_RATE_PLUS_CTD": compact_early,
    }
    print("Fusion Strategy Comparison")
    print("==========================")
    print(f"Development participants: {len(deployment)}")
    print("Official TEST participants used: 0")
    print("Threshold selection: not performed; fixed 0.5 threshold used for class metrics.")

    results: list[ModalityResult | LateFusionResult] = []
    for name, columns in representations.items():
        print(f"  {name} ({len(columns)} features)...", flush=True)
        result = run_nested_logistic_representation(deployment, name, columns, outer_splits)
        result.comparison_row["feature_mode"] = f"{len(columns)} early-fusion features"
        results.append(result)
    print(f"  {LATE_FUSION_NAME} (separate 38 + 13 feature pipelines)...", flush=True)
    results.append(run_nested_late_fusion(deployment, asr_rate, ctd_acoustic, outer_splits))

    deltas, delta_summary = _paired_deltas(results)
    evaluated_summary = pd.DataFrame([result.comparison_row for result in results])
    evaluated_summary["evaluation_mode"] = "nested_cv"
    summary = pd.concat([evaluated_summary, pd.DataFrame([_reference_row()])], ignore_index=True)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIRECTORY / "fusion_summary.csv", index=False)
    pd.concat([result.fold_metrics for result in results], ignore_index=True).to_csv(
        OUTPUT_DIRECTORY / "fold_metrics.csv", index=False
    )
    deltas.to_csv(OUTPUT_DIRECTORY / "paired_f1_deltas.csv", index=False)
    delta_summary.to_csv(OUTPUT_DIRECTORY / "paired_f1_delta_summary.csv", index=False)
    _build_oof_table(deployment, results).to_csv(OUTPUT_DIRECTORY / "oof_predictions.csv", index=False)
    late_result = next(result for result in results if result.representation == LATE_FUSION_NAME)
    late_result.selected_hyperparameters.to_csv(OUTPUT_DIRECTORY / "late_fusion_weights.csv", index=False)
    _plot_metric(summary, "f1", OUTPUT_DIRECTORY / "fusion_f1_comparison.png")
    _plot_metric(summary, "roc_auc", OUTPUT_DIRECTORY / "fusion_roc_auc_comparison.png")

    print("\nRepresentation                Features/Mode                     F1             ROC-AUC       Gap")
    for result in results:
        row = result.comparison_row
        mode = row["feature_mode"]
        print(
            f"{result.representation:<29} {mode:<32} {row['f1_mean']:.3f} ± {row['f1_std']:.3f}  "
            f"{row['roc_auc_mean']:.3f} ± {row['roc_auc_std']:.3f}  {row['f1_gap_mean']:.3f}"
        )
    print("\nPaired F1 deltas")
    for row in delta_summary.itertuples(index=False):
        print(f"{row.comparison}: {row.f1_delta_mean:.3f} ± {row.f1_delta_std:.3f}")
    print("\nSelected alpha values by outer fold")
    print(late_result.selected_hyperparameters[["fold", "alpha", "asr_rate_C", "ctd_acoustic_C", "inner_f1"]].to_string(index=False))
    print(f"\nSaved TRAIN-only outputs to: {OUTPUT_DIRECTORY}")
    print("Fold-wise deltas are descriptive only; no significance claim is made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
