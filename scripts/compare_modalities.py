"""Compare pure acoustic and linguistic/acoustic fusion representations on TRAIN only."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from ml.data.development_guard import assert_train_only
from ml.models.baseline_logistic import encode_screening_labels
from ml.models.modality_comparison import (
    ModalityResult,
    build_outer_splits,
    linguistic_feature_columns,
    merge_train_feature_tables,
    run_nested_logistic_representation,
    select_pure_acoustic_feature_columns,
)


LINGUISTIC_TRAIN_PATH = PROJECT_ROOT / "artifacts" / "features" / "linguistic_features_train.csv"
ACOUSTIC_TRAIN_PATH = PROJECT_ROOT / "artifacts" / "features" / "acoustic_features_train.csv"
OUTPUT_DIRECTORY = PROJECT_ROOT / "artifacts" / "results" / "modality_comparison"
EXPECTED_DEVELOPMENT_PARTICIPANTS = 320


def _plot_metric(results: list[ModalityResult], metric: str, title: str, output_path: Path) -> None:
    labels = [result.representation for result in results]
    means = [float(result.comparison_row[f"{metric}_mean"]) for result in results]
    stds = [float(result.comparison_row[f"{metric}_std"]) for result in results]
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(range(len(labels)), means, yerr=stds, capsize=4, color="#4C78A8")
    axis.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel(metric.replace("_", " ").upper())
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _save_fusion_confusion_matrix(result: ModalityResult) -> None:
    figure, axis = plt.subplots(figsize=(5, 4))
    display = ConfusionMatrixDisplay(
        confusion_matrix=result.oof_confusion_matrix,
        display_labels=["Healthy (0)", "Impaired (1)"],
    )
    display.plot(ax=axis, colorbar=False)
    axis.set_title("Linguistic + CTD Pure Acoustic: TRAIN OOF")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIRECTORY / "fusion_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def _build_oof_table(merged: pd.DataFrame, results: list[ModalityResult]) -> pd.DataFrame:
    oof = merged[["participant_id", "screening_label"]].copy()
    oof["true_label"] = oof["screening_label"].map({"Healthy": 0, "Impaired": 1}).astype(int)
    oof = oof.drop(columns="screening_label")
    for result in results:
        name = result.representation.casefold().replace(" ", "_").replace("+", "plus")
        values = result.oof_predictions.set_index("participant_id")
        oof = oof.join(
            values[["fold", "prediction", "score"]].rename(
                columns={
                    "fold": f"{name}_fold",
                    "prediction": f"{name}_prediction",
                    "score": f"{name}_score",
                }
            ),
            on="participant_id",
        )
    return oof


def _fusion_fold_comparison(results: list[ModalityResult]) -> pd.DataFrame:
    by_name = {result.representation: result.fold_metrics.set_index("fold") for result in results}
    linguistic = by_name["Linguistic"]
    ctd = by_name["CTD Pure Acoustic"]
    fusion_ctd = by_name["Linguistic + CTD Pure Acoustic"]
    fusion_all = by_name["Linguistic + ALL Pure Acoustic"]
    summary = pd.DataFrame(
        {
            "fold": linguistic.index,
            "linguistic_f1": linguistic["f1"].to_numpy(),
            "ctd_pure_acoustic_f1": ctd["f1"].to_numpy(),
            "linguistic_plus_ctd_f1": fusion_ctd["f1"].to_numpy(),
            "linguistic_plus_all_f1": fusion_all["f1"].to_numpy(),
        }
    )
    summary["fusion_ctd_minus_linguistic_f1"] = (
        summary["linguistic_plus_ctd_f1"] - summary["linguistic_f1"]
    )
    summary["fusion_all_minus_linguistic_f1"] = (
        summary["linguistic_plus_all_f1"] - summary["linguistic_f1"]
    )
    return summary.reset_index(drop=True)


def main() -> int:
    """Run the frozen pure-acoustic correction and targeted fusion comparison."""

    for path in (LINGUISTIC_TRAIN_PATH, ACOUSTIC_TRAIN_PATH):
        if not path.is_file():
            print(f"Required TRAIN feature table not found: {path}")
            return 1
    linguistic = assert_train_only(pd.read_csv(LINGUISTIC_TRAIN_PATH))
    acoustic = assert_train_only(pd.read_csv(ACOUSTIC_TRAIN_PATH))
    # Select the established linguistic schema before joining acoustic columns.
    # Selecting after the merge would also match acoustic task prefixes.
    linguistic_features = linguistic_feature_columns(linguistic)
    merged = merge_train_feature_tables(linguistic, acoustic)
    if len(merged) != EXPECTED_DEVELOPMENT_PARTICIPANTS:
        raise AssertionError(f"Expected 320 development rows; found {len(merged)}.")

    pure_sft = select_pure_acoustic_feature_columns(merged, "SFT")
    pure_pft = select_pure_acoustic_feature_columns(merged, "PFT")
    pure_ctd = select_pure_acoustic_feature_columns(merged, "CTD")
    pure_all = select_pure_acoustic_feature_columns(merged, "ALL")
    if (len(pure_sft), len(pure_pft), len(pure_ctd), len(pure_all)) != (13, 13, 13, 39):
        raise AssertionError("Pure acoustic feature schema must be 13 per task and 39 overall.")

    target = encode_screening_labels(merged["screening_label"]).to_numpy()
    outer_splits = build_outer_splits(target)
    representations = {
        "SFT Pure Acoustic": pure_sft,
        "PFT Pure Acoustic": pure_pft,
        "CTD Pure Acoustic": pure_ctd,
        "ALL Pure Acoustic": pure_all,
        "Linguistic": linguistic_features,
        "Linguistic + CTD Pure Acoustic": [*linguistic_features, *pure_ctd],
        "Linguistic + ALL Pure Acoustic": [*linguistic_features, *pure_all],
    }

    print("Pure Acoustic and Multimodal Fusion Comparison")
    print("================================================")
    print(f"Development participants: {len(merged)}")
    print("Official TEST participants used: 0")
    print("Pure acoustic features: 13 per task; 39 ALL", flush=True)

    results: list[ModalityResult] = []
    for name, columns in representations.items():
        print(f"  {name} ({len(columns)} features)...", flush=True)
        results.append(
            run_nested_logistic_representation(merged, name, columns, outer_splits)
        )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result.comparison_row for result in results]).to_csv(
        OUTPUT_DIRECTORY / "modality_comparison.csv", index=False
    )
    pd.concat([result.fold_metrics for result in results], ignore_index=True).to_csv(
        OUTPUT_DIRECTORY / "fold_metrics.csv", index=False
    )
    _build_oof_table(merged, results).to_csv(OUTPUT_DIRECTORY / "oof_predictions.csv", index=False)
    fusion_folds = _fusion_fold_comparison(results)
    fusion_folds.to_csv(OUTPUT_DIRECTORY / "fusion_fold_comparison.csv", index=False)
    _save_fusion_confusion_matrix(
        next(result for result in results if result.representation == "Linguistic + CTD Pure Acoustic")
    )
    _plot_metric(
        results,
        "f1",
        "TRAIN-only nested-CV F1 by representation",
        OUTPUT_DIRECTORY / "modality_f1_comparison.png",
    )
    _plot_metric(
        results,
        "roc_auc",
        "TRAIN-only nested-CV ROC-AUC by representation",
        OUTPUT_DIRECTORY / "modality_roc_auc_comparison.png",
    )

    comparison = pd.DataFrame([result.comparison_row for result in results]).set_index("representation")
    display_names = (
        "Linguistic",
        "CTD Pure Acoustic",
        "Linguistic + CTD Pure Acoustic",
        "Linguistic + ALL Pure Acoustic",
    )
    print("\nRepresentation                         F1            ROC-AUC       F1 gap")
    for name in display_names:
        row = comparison.loc[name]
        print(
            f"{name:<38} {row['f1_mean']:.3f} ± {row['f1_std']:.3f}  "
            f"{row['roc_auc_mean']:.3f} ± {row['roc_auc_std']:.3f}  {row['f1_gap_mean']:.3f}"
        )
    print("\nFold-level fusion deltas relative to Linguistic")
    print(fusion_folds.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(
        "Fusion CTD minus linguistic F1: "
        f"{fusion_folds['fusion_ctd_minus_linguistic_f1'].mean():.3f} ± "
        f"{fusion_folds['fusion_ctd_minus_linguistic_f1'].std(ddof=1):.3f}"
    )
    print(
        "Fusion ALL minus linguistic F1: "
        f"{fusion_folds['fusion_all_minus_linguistic_f1'].mean():.3f} ± "
        f"{fusion_folds['fusion_all_minus_linguistic_f1'].std(ddof=1):.3f}"
    )
    print(f"\nSaved TRAIN-only modality-comparison outputs to: {OUTPUT_DIRECTORY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
