"""Compare five fixed TRAIN-only representations for deployment alignment.

This script keeps manual-transcript results as a research reference while
quantifying the effect of ASR text, ASR-derived rates, and the frozen pure
acoustic subset.  It neither loads nor evaluates the official locked TEST set.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from ml.asr.asr_linguistic_features import asr_feature_columns
from ml.audio.acoustic_features import speech_rate_features
from ml.data.development_guard import assert_train_only
from ml.models.baseline_logistic import encode_screening_labels, select_feature_columns
from ml.models.modality_comparison import (
    PURE_ACOUSTIC_SUFFIXES,
    ModalityResult,
    build_outer_splits,
    run_nested_logistic_representation,
)


MANUAL_TRAIN_PATH = PROJECT_ROOT / "artifacts" / "features" / "linguistic_features_train.csv"
ASR_LINGUISTIC_TRAIN_PATH = PROJECT_ROOT / "artifacts" / "features" / "asr_linguistic_features_train.csv"
ACOUSTIC_TRAIN_PATH = PROJECT_ROOT / "artifacts" / "features" / "acoustic_features_train.csv"
DEPLOYMENT_TRAIN_PATH = PROJECT_ROOT / "artifacts" / "features" / "deployment_features_train.csv"
OUTPUT_DIRECTORY = PROJECT_ROOT / "artifacts" / "results" / "deployment_representation_comparison"
EXPECTED_DEVELOPMENT_PARTICIPANTS = 320
TASK_PREFIXES = ("sft_", "pft_", "ctd_")
RATE_SUFFIXES = ("recording_word_rate_wpm", "articulation_rate_wpm")
IDENTITY_COLUMNS = ("diagnosis", "screening_label", "Split")


def pure_acoustic_feature_columns(feature_table: pd.DataFrame) -> list[str]:
    """Select exactly the frozen 39 waveform-derived interpretable features."""

    selected = [
        f"{prefix}{suffix}"
        for prefix in TASK_PREFIXES
        for suffix in PURE_ACOUSTIC_SUFFIXES
    ]
    missing = [column for column in selected if column not in feature_table.columns]
    if missing:
        raise ValueError(f"Missing frozen pure acoustic features: {missing}")
    return selected


def asr_rate_feature_columns(feature_table: pd.DataFrame) -> list[str]:
    """Select only the six rates derived from base.en word counts."""

    selected = [
        f"{prefix}{suffix}"
        for prefix in TASK_PREFIXES
        for suffix in RATE_SUFFIXES
    ]
    missing = [column for column in selected if column not in feature_table.columns]
    if missing:
        raise ValueError(f"Missing ASR-derived speech-rate features: {missing}")
    return selected


def _align_train_table(reference: pd.DataFrame, candidate: pd.DataFrame, name: str) -> pd.DataFrame:
    """Require matched TRAIN identities, then return candidate in reference order."""

    reference = assert_train_only(reference)
    candidate = assert_train_only(candidate)
    if len(reference) != EXPECTED_DEVELOPMENT_PARTICIPANTS:
        raise AssertionError("Reference table must contain exactly 320 TRAIN participants.")
    if len(candidate) != EXPECTED_DEVELOPMENT_PARTICIPANTS:
        raise AssertionError(f"{name} must contain exactly 320 TRAIN participants.")
    if set(reference["participant_id"]) != set(candidate["participant_id"]):
        raise ValueError(f"{name} participant IDs do not align with the manual TRAIN table.")
    aligned = candidate.set_index("participant_id").loc[reference["participant_id"]].copy().reset_index()
    expected = reference.set_index("participant_id")[list(IDENTITY_COLUMNS)]
    observed = aligned.set_index("participant_id")[list(IDENTITY_COLUMNS)]
    if not expected.equals(observed):
        raise ValueError(f"{name} labels or Split values do not align with the manual TRAIN table.")
    return assert_train_only(aligned)


def manual_asr_compatible_feature_columns(
    manual_table: pd.DataFrame, asr_table: pd.DataFrame
) -> list[str]:
    """Use the exact plain-transcript feature schema shared by manual and ASR text."""

    selected = asr_feature_columns(asr_table)
    missing = [column for column in selected if column not in manual_table.columns]
    if missing:
        raise ValueError(f"Manual table lacks ASR-compatible features: {missing}")
    if any("pause_annotation" in column for column in selected):
        raise AssertionError("ASR-compatible representation must not include manual pause annotations.")
    return selected


def validate_asr_rate_provenance(asr_table: pd.DataFrame, deployment_table: pd.DataFrame) -> None:
    """Prove deployment rates are recomputed from ASR counts and frozen timing."""

    asr = assert_train_only(asr_table).set_index("participant_id")
    deployment = assert_train_only(deployment_table).set_index("participant_id")
    for prefix in TASK_PREFIXES:
        word_column = f"{prefix}word_count"
        duration_column = f"{prefix}audio_duration_seconds"
        voiced_column = f"{prefix}voiced_duration_seconds"
        for participant_id in asr.index:
            expected = speech_rate_features(
                asr.at[participant_id, word_column],
                float(deployment.at[participant_id, duration_column]),
                float(deployment.at[participant_id, voiced_column]),
            )
            for suffix, expected_value in expected.items():
                actual_value = float(deployment.at[participant_id, f"{prefix}{suffix}"])
                if not np.isclose(actual_value, expected_value, equal_nan=True):
                    raise ValueError(
                        "Deployment rate provenance check failed: rates must use ASR word counts, "
                        f"not manual transcript counts ({participant_id}, {prefix}{suffix})."
                    )


def validate_representation_columns(
    representations: dict[str, list[str]], sources: dict[str, pd.DataFrame]
) -> None:
    """Reject duplicated, metadata, manual-rate, or manual-pause feature selections."""

    forbidden = {"participant_id", *IDENTITY_COLUMNS, "age", "gender", "MMSE"}
    for name, columns in representations.items():
        feature_table = sources[name]
        duplicates = [column for column, count in Counter(columns).items() if count > 1]
        if duplicates:
            raise ValueError(f"{name} contains duplicate features: {duplicates}")
        missing = [column for column in columns if column not in feature_table.columns]
        if missing:
            raise ValueError(f"{name} contains missing features: {missing}")
        metadata = sorted(set(columns).intersection(forbidden))
        if metadata:
            raise ValueError(f"{name} contains metadata/label features: {metadata}")
    deployment_table = sources["DEPLOYMENT_MULTIMODAL"]
    multimodal = representations["DEPLOYMENT_MULTIMODAL"]
    if any("pause_annotation" in column for column in multimodal):
        raise AssertionError("Deployment multimodal must not contain manual pause annotations.")
    if any(
        column.endswith(RATE_SUFFIXES)
        and column not in asr_rate_feature_columns(deployment_table)
        for column in multimodal
    ):
        raise AssertionError("Deployment multimodal contains an unsupported word-rate field.")


def _result_lookup(results: Iterable[ModalityResult]) -> dict[str, ModalityResult]:
    return {result.representation: result for result in results}


def paired_f1_deltas(results: Iterable[ModalityResult]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate pre-specified paired fold-level F1 differences on shared folds."""

    by_name = _result_lookup(results)
    comparisons = (
        ("ASR vs compatible manual", "ASR_LINGUISTIC", "MANUAL_ASR_COMPATIBLE"),
        ("+ speech rate", "ASR_LINGUISTIC_PLUS_RATE", "ASR_LINGUISTIC"),
        ("+ pure acoustic", "DEPLOYMENT_MULTIMODAL", "ASR_LINGUISTIC_PLUS_RATE"),
        ("Deployment multimodal vs ASR linguistic", "DEPLOYMENT_MULTIMODAL", "ASR_LINGUISTIC"),
        ("Deployment multimodal vs manual full", "DEPLOYMENT_MULTIMODAL", "MANUAL_FULL"),
    )
    delta_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for display_name, minuend_name, subtrahend_name in comparisons:
        minuend = by_name[minuend_name].fold_metrics.set_index("fold")["f1"]
        subtrahend = by_name[subtrahend_name].fold_metrics.set_index("fold")["f1"]
        if not minuend.index.equals(subtrahend.index):
            raise AssertionError("Paired representation folds do not align.")
        delta = minuend - subtrahend
        for fold, value in delta.items():
            delta_rows.append(
                {
                    "comparison": display_name,
                    "minuend_representation": minuend_name,
                    "subtrahend_representation": subtrahend_name,
                    "fold": int(fold),
                    "f1_delta": float(value),
                }
            )
        summary_rows.append(
            {
                "comparison": display_name,
                "minuend_representation": minuend_name,
                "subtrahend_representation": subtrahend_name,
                "f1_delta_mean": float(delta.mean()),
                "f1_delta_std": float(delta.std(ddof=1)),
            }
        )
    return pd.DataFrame(delta_rows), pd.DataFrame(summary_rows)


def _build_oof_table(manual_table: pd.DataFrame, results: Iterable[ModalityResult]) -> pd.DataFrame:
    """Write one outer-fold prediction and score per participant/representation."""

    oof = manual_table[["participant_id", "screening_label"]].copy()
    oof["true_label"] = oof["screening_label"].map({"Healthy": 0, "Impaired": 1}).astype(int)
    oof = oof.drop(columns="screening_label")
    for result in results:
        name = result.representation.casefold()
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


def _plot_metric(results: list[ModalityResult], metric: str, output_path: Path) -> None:
    labels = [result.representation.replace("_", "\n") for result in results]
    means = [float(result.comparison_row[f"{metric}_mean"]) for result in results]
    stds = [float(result.comparison_row[f"{metric}_std"]) for result in results]
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(range(len(results)), means, yerr=stds, capsize=4, color="#4C78A8")
    axis.set_xticks(range(len(results)), labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel(metric.replace("_", " ").upper())
    axis.set_title(f"TRAIN-only deployment representation comparison: {metric.replace('_', ' ').upper()}")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _save_confusion_matrix(result: ModalityResult, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=result.oof_confusion_matrix,
        display_labels=["Healthy (0)", "Impaired (1)"],
    ).plot(ax=axis, colorbar=False)
    axis.set_title(f"{result.representation}: TRAIN outer-fold OOF")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    """Run the fixed, deployment-aligned representation comparison on TRAIN only."""

    paths = (MANUAL_TRAIN_PATH, ASR_LINGUISTIC_TRAIN_PATH, ACOUSTIC_TRAIN_PATH, DEPLOYMENT_TRAIN_PATH)
    for path in paths:
        if not path.is_file():
            print(f"Required TRAIN feature table not found: {path}")
            return 1
    manual = assert_train_only(pd.read_csv(MANUAL_TRAIN_PATH))
    asr = _align_train_table(manual, pd.read_csv(ASR_LINGUISTIC_TRAIN_PATH), "ASR linguistic table")
    acoustic = _align_train_table(manual, pd.read_csv(ACOUSTIC_TRAIN_PATH), "Acoustic table")
    deployment = _align_train_table(manual, pd.read_csv(DEPLOYMENT_TRAIN_PATH), "Deployment table")
    if len(manual) != EXPECTED_DEVELOPMENT_PARTICIPANTS:
        raise AssertionError("Manual table must contain exactly 320 TRAIN participants.")

    manual_full = select_feature_columns(manual, "ALL")
    manual_compatible = manual_asr_compatible_feature_columns(manual, asr)
    asr_linguistic = asr_feature_columns(asr)
    pure_acoustic = pure_acoustic_feature_columns(deployment)
    asr_rates = asr_rate_feature_columns(deployment)
    validate_asr_rate_provenance(asr, deployment)
    if not all(column in acoustic.columns for column in pure_acoustic):
        raise AssertionError("Pure acoustic columns must originate from the acoustic TRAIN schema.")
    if not all(column in deployment.columns for column in asr_linguistic):
        raise AssertionError("Deployment table must retain the ASR linguistic schema.")
    for column in asr_linguistic:
        if not asr[column].equals(deployment[column]):
            raise ValueError(f"Deployment ASR linguistic value mismatch for {column}.")

    representations = {
        "MANUAL_FULL": manual_full,
        "MANUAL_ASR_COMPATIBLE": manual_compatible,
        "ASR_LINGUISTIC": asr_linguistic,
        "ASR_LINGUISTIC_PLUS_RATE": [*asr_linguistic, *asr_rates],
        "DEPLOYMENT_MULTIMODAL": [*asr_linguistic, *asr_rates, *pure_acoustic],
    }
    expected_counts = {
        "MANUAL_FULL": 44,
        "MANUAL_ASR_COMPATIBLE": 32,
        "ASR_LINGUISTIC": 32,
        "ASR_LINGUISTIC_PLUS_RATE": 38,
        "DEPLOYMENT_MULTIMODAL": 77,
    }
    observed_counts = {name: len(columns) for name, columns in representations.items()}
    if observed_counts != expected_counts:
        raise AssertionError(f"Unexpected representation feature counts: {observed_counts}")
    target = encode_screening_labels(manual["screening_label"]).to_numpy()
    outer_splits = build_outer_splits(target)
    sources = {
        "MANUAL_FULL": manual,
        "MANUAL_ASR_COMPATIBLE": manual,
        "ASR_LINGUISTIC": asr,
        "ASR_LINGUISTIC_PLUS_RATE": deployment,
        "DEPLOYMENT_MULTIMODAL": deployment,
    }
    validate_representation_columns(representations, sources)
    print("Deployment Representation Comparison")
    print("====================================")
    print(f"Development participants: {len(manual)}")
    print("Official TEST participants used: 0")
    print("Threshold selection: not performed; default estimator predictions are reported.")

    results: list[ModalityResult] = []
    for name, columns in representations.items():
        print(f"  {name} ({len(columns)} features)...", flush=True)
        results.append(
            run_nested_logistic_representation(sources[name], name, columns, outer_splits)
        )

    deltas, delta_summary = paired_f1_deltas(results)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result.comparison_row for result in results]).to_csv(
        OUTPUT_DIRECTORY / "representation_summary.csv", index=False
    )
    pd.concat([result.fold_metrics for result in results], ignore_index=True).to_csv(
        OUTPUT_DIRECTORY / "fold_metrics.csv", index=False
    )
    deltas.to_csv(OUTPUT_DIRECTORY / "paired_f1_deltas.csv", index=False)
    delta_summary.to_csv(OUTPUT_DIRECTORY / "paired_f1_delta_summary.csv", index=False)
    _build_oof_table(manual, results).to_csv(OUTPUT_DIRECTORY / "oof_predictions.csv", index=False)
    pd.concat([result.selected_hyperparameters for result in results], ignore_index=True).to_csv(
        OUTPUT_DIRECTORY / "selected_hyperparameters.csv", index=False
    )
    _plot_metric(results, "f1", OUTPUT_DIRECTORY / "representation_f1_comparison.png")
    _plot_metric(results, "roc_auc", OUTPUT_DIRECTORY / "representation_roc_auc_comparison.png")
    by_name = _result_lookup(results)
    for name in ("MANUAL_FULL", "ASR_LINGUISTIC", "DEPLOYMENT_MULTIMODAL"):
        _save_confusion_matrix(by_name[name], OUTPUT_DIRECTORY / f"{name.casefold()}_confusion_matrix.png")

    print("\nRepresentation                   Features   F1             ROC-AUC       F1 gap")
    for result in results:
        row = result.comparison_row
        print(
            f"{result.representation:<32} {row['feature_count']:>8}   "
            f"{row['f1_mean']:.3f} ± {row['f1_std']:.3f}  "
            f"{row['roc_auc_mean']:.3f} ± {row['roc_auc_std']:.3f}  {row['f1_gap_mean']:.3f}"
        )
    print("\nPaired F1 deltas")
    for row in delta_summary.itertuples(index=False):
        print(f"{row.comparison}: {row.f1_delta_mean:.3f} ± {row.f1_delta_std:.3f}")
    print(f"\nSaved TRAIN-only comparison outputs to: {OUTPUT_DIRECTORY}")
    print("Five fold-wise deltas are descriptive only; no significance claim is made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
