"""Compare original and redundancy-reduced linguistic features with nested CV.

The official TEST split is intentionally excluded.  The reduced set removes
only repeated-word count and repetition ratio from SFT, PFT, and CTD.
"""

from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter
from typing import Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.baseline_logistic import select_feature_columns, training_rows
from ml.models.classical_comparison import ModelName, run_nested_cv_model


FEATURE_TABLE_PATH = PROJECT_ROOT / "artifacts" / "features" / "manual_transcript_features.csv"
OUTPUT_DIRECTORY = PROJECT_ROOT / "artifacts" / "results" / "feature_ablation"
ABLATED_MODELS: tuple[ModelName, ...] = ("logistic", "linear_svm", "rbf_svm")
REDUNDANT_FEATURES = {
    "sft_repeated_word_count",
    "sft_repetition_ratio",
    "pft_repeated_word_count",
    "pft_repetition_ratio",
    "ctd_repeated_word_count",
    "ctd_repetition_ratio",
}


def reduced_feature_columns(original_columns: Sequence[str]) -> list[str]:
    """Drop only the six prespecified derived repetition measurements."""

    original = list(original_columns)
    missing_expected_columns = REDUNDANT_FEATURES.difference(original)
    if missing_expected_columns:
        raise ValueError(
            "The feature table is missing expected redundant columns: "
            f"{sorted(missing_expected_columns)}"
        )
    return [column for column in original if column not in REDUNDANT_FEATURES]


def save_outputs(
    comparison_rows: list[dict[str, float | int | str]],
    fold_frames: list[pd.DataFrame],
    feature_sets: dict[str, list[str]],
    parameter_frames: list[pd.DataFrame],
) -> None:
    """Save compact, reproducible ablation summaries without transcript content."""

    pd.DataFrame(comparison_rows).to_csv(
        OUTPUT_DIRECTORY / "feature_ablation_comparison.csv", index=False
    )
    pd.concat(fold_frames, ignore_index=True).to_csv(
        OUTPUT_DIRECTORY / "feature_ablation_fold_metrics.csv", index=False
    )
    pd.concat(parameter_frames, ignore_index=True).to_csv(
        OUTPUT_DIRECTORY / "feature_ablation_selected_hyperparameters.csv", index=False
    )
    pd.DataFrame(
        [
            {"feature_set": name, "feature": feature}
            for name, features in feature_sets.items()
            for feature in features
        ]
    ).to_csv(OUTPUT_DIRECTORY / "feature_ablation_feature_sets.csv", index=False)


def main() -> int:
    if not FEATURE_TABLE_PATH.is_file():
        print(f"Feature table not found: {FEATURE_TABLE_PATH}")
        return 1

    feature_table = pd.read_csv(FEATURE_TABLE_PATH)
    train_table = training_rows(feature_table)
    if not train_table["Split"].astype("string").str.strip().str.upper().eq("TRAIN").all():
        raise AssertionError("Only official TRAIN rows may enter this ablation.")

    original_features = select_feature_columns(train_table, "ALL")
    feature_sets = {
        "original_44": original_features,
        "reduced_38": reduced_feature_columns(original_features),
    }
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    print("Linguistic Redundancy Ablation")
    print("===============================")
    print(f"TRAIN participants: {len(train_table)}")
    print("Official TEST participants used: 0", flush=True)
    print(
        f"Original features: {len(feature_sets['original_44'])}; "
        f"reduced features: {len(feature_sets['reduced_38'])}",
        flush=True,
    )

    started_at = perf_counter()
    comparison_rows: list[dict[str, float | int | str]] = []
    fold_frames: list[pd.DataFrame] = []
    parameter_frames: list[pd.DataFrame] = []
    for feature_set_name, feature_columns in feature_sets.items():
        print(f"\n{feature_set_name} ({len(feature_columns)} features)", flush=True)
        for model in ABLATED_MODELS:
            print(f"  {model.replace('_', ' ').title()}...", flush=True)
            result = run_nested_cv_model(
                feature_table,
                model,
                feature_columns=feature_columns,
                progress_callback=lambda message: print(message, flush=True),
            )
            row = {
                "feature_set": feature_set_name,
                "feature_count": len(feature_columns),
                **result.comparison_row,
            }
            comparison_rows.append(row)
            fold_frames.append(result.fold_metrics.assign(feature_set=feature_set_name))
            parameter_frames.append(
                result.selected_hyperparameters.assign(feature_set=feature_set_name)
            )

    save_outputs(comparison_rows, fold_frames, feature_sets, parameter_frames)
    comparison = pd.DataFrame(comparison_rows)
    display_columns = [
        "feature_set",
        "feature_count",
        "model",
        "accuracy_mean",
        "recall_mean",
        "specificity_mean",
        "f1_mean",
        "roc_auc_mean",
        "f1_gap_mean",
    ]
    print("\nCompact outer-fold comparison")
    print(comparison[display_columns].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Elapsed time: {perf_counter() - started_at:.1f} seconds")
    print(f"Saved ablation outputs to: {OUTPUT_DIRECTORY}")
    print("This ablation does not select a final model and does not evaluate TEST.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
