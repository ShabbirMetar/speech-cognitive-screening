"""Run TRAIN-only nested-CV baselines with prespecified acoustic features."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from ml.data.development_guard import assert_train_only
from ml.models.acoustic_baseline import (
    ACOUSTIC_EXPERIMENTS,
    ACOUSTIC_MODEL_NAMES,
    AcousticExperimentResult,
    run_nested_acoustic_experiment,
    select_acoustic_feature_columns,
)


FEATURE_TABLE_PATH = PROJECT_ROOT / "artifacts" / "features" / "acoustic_features_train.csv"
OUTPUT_DIRECTORY = PROJECT_ROOT / "artifacts" / "results" / "acoustic_baseline"
EXPECTED_DEVELOPMENT_PARTICIPANTS = 320


def save_confusion_matrix(result: AcousticExperimentResult) -> None:
    """Save an outer-fold OOF confusion matrix for one task/model pair."""

    figure, axis = plt.subplots(figsize=(5, 4))
    display = ConfusionMatrixDisplay(
        confusion_matrix=result.oof_confusion_matrix,
        display_labels=["Healthy (0)", "Impaired (1)"],
    )
    display.plot(ax=axis, colorbar=False)
    axis.set_title(
        f"{result.experiment} acoustic / {result.model.replace('_', ' ').title()}: TRAIN OOF"
    )
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIRECTORY
        / f"{result.experiment.lower()}_{result.model}_confusion_matrix.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def _task_comparison(results: list[AcousticExperimentResult]) -> pd.DataFrame:
    """Create a compact task-by-model metrics table without choosing a winner."""

    rows: list[dict[str, float | int | str]] = []
    for experiment in ACOUSTIC_EXPERIMENTS:
        matching_results = [result for result in results if result.experiment == experiment]
        row: dict[str, float | int | str] = {
            "experiment": experiment,
            "feature_count": matching_results[0].comparison_row["feature_count"],
        }
        for result in matching_results:
            for metric in (
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "specificity",
                "f1",
                "roc_auc",
            ):
                row[f"{result.model}_{metric}_mean"] = result.comparison_row[f"{metric}_mean"]
                row[f"{result.model}_{metric}_std"] = result.comparison_row[f"{metric}_std"]
            row[f"{result.model}_f1_gap_mean"] = result.comparison_row["f1_gap_mean"]
        rows.append(row)
    return pd.DataFrame(rows)


def save_outputs(
    feature_table: pd.DataFrame, results: list[AcousticExperimentResult]
) -> None:
    """Persist summaries, TRAIN OOF predictions, selected grids, and matrices."""

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result.comparison_row for result in results]).to_csv(
        OUTPUT_DIRECTORY / "model_comparison.csv", index=False
    )
    _task_comparison(results).to_csv(OUTPUT_DIRECTORY / "task_comparison.csv", index=False)
    pd.concat(
        [result.fold_metrics for result in results], ignore_index=True
    ).to_csv(OUTPUT_DIRECTORY / "fold_metrics.csv", index=False)
    pd.concat(
        [result.selected_hyperparameters for result in results], ignore_index=True
    ).to_csv(OUTPUT_DIRECTORY / "selected_hyperparameters.csv", index=False)

    oof = feature_table[["participant_id", "screening_label"]].copy()
    oof["true_label"] = oof["screening_label"].map({"Healthy": 0, "Impaired": 1}).astype(int)
    oof = oof.drop(columns="screening_label")
    for result in results:
        name = f"{result.experiment.lower()}_{result.model}"
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
        save_confusion_matrix(result)
    oof.to_csv(OUTPUT_DIRECTORY / "oof_predictions.csv", index=False)


def main() -> int:
    """Run all predeclared acoustic model/task comparisons on development data."""

    if not FEATURE_TABLE_PATH.is_file():
        print(f"TRAIN acoustic feature table not found: {FEATURE_TABLE_PATH}")
        return 1
    feature_table = assert_train_only(pd.read_csv(FEATURE_TABLE_PATH))
    if len(feature_table) != EXPECTED_DEVELOPMENT_PARTICIPANTS:
        raise AssertionError(
            f"Expected {EXPECTED_DEVELOPMENT_PARTICIPANTS} development participants; found {len(feature_table)}."
        )
    all_feature_count = len(select_acoustic_feature_columns(feature_table, "ALL"))
    if all_feature_count != 45:
        raise AssertionError(f"Expected 45 selected acoustic features; found {all_feature_count}.")

    print("Acoustic Nested-CV Baseline")
    print("==========================")
    print(f"Development participants: {len(feature_table)}")
    print("Official TEST participants used: 0")
    print(f"Interpretable acoustic feature count: {all_feature_count}", flush=True)

    results: list[AcousticExperimentResult] = []
    for experiment in ACOUSTIC_EXPERIMENTS:
        feature_count = len(select_acoustic_feature_columns(feature_table, experiment))
        print(f"\n{experiment} acoustic only ({feature_count} features)", flush=True)
        for model in ACOUSTIC_MODEL_NAMES:
            print(f"  {model.replace('_', ' ').title()}", flush=True)
            result = run_nested_acoustic_experiment(
                feature_table,
                experiment,
                model,
                progress_callback=lambda message: print(message, flush=True),
            )
            results.append(result)

    save_outputs(feature_table, results)
    print("\nOuter-fold summary")
    for experiment in ACOUSTIC_EXPERIMENTS:
        print(f"{experiment}")
        for result in (item for item in results if item.experiment == experiment):
            row = result.comparison_row
            print(
                f"  {result.model}: F1 {row['f1_mean']:.3f} ± {row['f1_std']:.3f}; "
                f"ROC-AUC {row['roc_auc_mean']:.3f} ± {row['roc_auc_std']:.3f}; "
                f"F1 gap {row['f1_gap_mean']:.3f}"
            )
    print("\nLinguistic reference (nested CV; not a modality-superiority claim)")
    print("  Logistic Regression ALL: F1 approximately 0.676; ROC-AUC approximately 0.762")
    print(f"\nSaved TRAIN-only acoustic baseline outputs to: {OUTPUT_DIRECTORY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
