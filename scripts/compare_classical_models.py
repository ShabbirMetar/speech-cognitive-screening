"""Run the TRAIN-only nested-CV comparison for classical linguistic models."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.baseline_logistic import training_rows
from ml.models.classical_comparison import MODEL_NAMES, run_classical_comparison


FEATURE_TABLE_PATH = PROJECT_ROOT / "artifacts" / "features" / "manual_transcript_features.csv"
OUTPUT_DIRECTORY = PROJECT_ROOT / "artifacts" / "results" / "model_comparison"


def save_confusion_matrix(model: str, confusion: object, output_path: Path) -> None:
    """Save one labelled outer-fold OOF confusion matrix for a model family."""

    figure, axis = plt.subplots(figsize=(5, 4))
    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion,
        display_labels=["Healthy (0)", "Impaired (1)"],
    )
    display.plot(ax=axis, colorbar=False)
    axis.set_title(f"{model.replace('_', ' ').title()}: TRAIN outer-fold OOF")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    if not FEATURE_TABLE_PATH.is_file():
        print(f"Feature table not found: {FEATURE_TABLE_PATH}")
        return 1
    feature_table = pd.read_csv(FEATURE_TABLE_PATH)
    train_table = training_rows(feature_table)
    # A single GridSearchCV worker keeps this reproducible on Windows and avoids
    # process-pool contention while estimator internals use their configured worker.
    results = run_classical_comparison(feature_table, grid_n_jobs=1)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    model_comparison = pd.DataFrame([result.comparison_row for result in results.values()])
    selected_parameters = pd.concat(
        [result.selected_hyperparameters for result in results.values()], ignore_index=True
    )
    oof_predictions = train_table[["participant_id"]].copy()
    oof_predictions["true_label"] = (
        train_table["screening_label"].map({"Healthy": 0, "Impaired": 1}).astype(int)
    )
    for model, result in results.items():
        model_oof = result.oof_predictions.set_index("participant_id")
        oof_predictions = oof_predictions.join(
            model_oof[["fold", "prediction", "score"]].rename(
                columns={
                    "fold": f"{model}_fold",
                    "prediction": f"{model}_prediction",
                    "score": f"{model}_score",
                }
            ),
            on="participant_id",
        )
        save_confusion_matrix(
            model,
            result.oof_confusion_matrix,
            OUTPUT_DIRECTORY / f"{model}_confusion_matrix.png",
        )

    model_comparison.to_csv(OUTPUT_DIRECTORY / "model_comparison.csv", index=False)
    selected_parameters.to_csv(OUTPUT_DIRECTORY / "selected_hyperparameters.csv", index=False)
    oof_predictions.to_csv(OUTPUT_DIRECTORY / "oof_predictions.csv", index=False)

    print("Nested-CV Classical Linguistic Model Comparison")
    print("===============================================")
    print(f"TRAIN participants: {len(train_table)}")
    print("TEST participants used: 0")
    for model in MODEL_NAMES:
        row = results[model].comparison_row
        print(
            f"{model}: F1 {row['f1_mean']:.3f} ± {row['f1_std']:.3f}; "
            f"ROC-AUC {row['roc_auc_mean']:.3f} ± {row['roc_auc_std']:.3f}; "
            f"F1 gap {row['f1_gap_mean']:.3f}"
        )
    print(f"Saved nested-CV outputs to: {OUTPUT_DIRECTORY}")
    print("This reports the best observed classical linguistic model under nested CV, not a final deployed model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
