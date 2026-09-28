"""Run TRAIN-only cross-validated Logistic Regression linguistic baselines."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.baseline_logistic import run_all_experiments, training_rows


FEATURE_TABLE_PATH = PROJECT_ROOT / "artifacts" / "features" / "manual_transcript_features.csv"
OUTPUT_DIRECTORY = PROJECT_ROOT / "artifacts" / "results" / "baseline"


def save_confusion_matrix(confusion, output_path: Path) -> None:
    """Save the ALL-experiment TRAIN out-of-fold confusion matrix."""

    figure, axis = plt.subplots(figsize=(5, 4))
    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion,
        display_labels=["Healthy (0)", "Impaired (1)"],
    )
    display.plot(ax=axis, colorbar=False)
    axis.set_title("ALL linguistic features: TRAIN out-of-fold confusion matrix")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    if not FEATURE_TABLE_PATH.is_file():
        print(f"Feature table not found: {FEATURE_TABLE_PATH}")
        return 1

    feature_table = pd.read_csv(FEATURE_TABLE_PATH)
    train_table = training_rows(feature_table)
    results = run_all_experiments(feature_table)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    cv_metrics = pd.concat([result.cv_metrics for result in results.values()], ignore_index=True)
    task_comparison = pd.DataFrame([result.comparison_row for result in results.values()])
    cv_metrics.to_csv(OUTPUT_DIRECTORY / "cv_metrics.csv", index=False)
    task_comparison.to_csv(OUTPUT_DIRECTORY / "task_comparison.csv", index=False)
    results["ALL"].oof_predictions.to_csv(
        OUTPUT_DIRECTORY / "oof_predictions_all.csv", index=False
    )
    save_confusion_matrix(
        results["ALL"].confusion_matrix,
        OUTPUT_DIRECTORY / "oof_confusion_matrix_all.png",
    )

    print("Linguistic Logistic Regression Baseline")
    print("=======================================")
    print(f"TRAIN participants: {len(train_table)}")
    print("TEST participants used: 0")
    for experiment_name, result in results.items():
        summary = result.comparison_row
        print(f"\n{experiment_name}")
        print(f"Feature columns: {summary['feature_count']}")
        for metric_name, display_name in (
            ("accuracy", "Accuracy"),
            ("balanced_accuracy", "Balanced accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall / sensitivity"),
            ("specificity", "Specificity"),
            ("f1", "F1"),
            ("roc_auc", "ROC-AUC"),
        ):
            print(
                f"{display_name}: {summary[f'{metric_name}_mean']:.3f} "
                f"± {summary[f'{metric_name}_std']:.3f}"
            )
        print(
            "Train-vs-validation F1 gap: "
            f"{summary['train_validation_f1_gap_mean']:.3f}"
        )
    print(f"\nSaved CV outputs to: {OUTPUT_DIRECTORY}")
    print("This is a baseline comparison only; no final model winner is declared.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
