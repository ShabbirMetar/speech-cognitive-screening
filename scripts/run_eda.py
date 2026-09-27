"""Run read-only, metadata-only exploratory analysis for PROCESS-2."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
for import_root in (PROJECT_ROOT, BACKEND_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from ml.data.process2_eda import (
    EDAConfigurationError,
    age_descriptive_statistics,
    age_distribution_by_diagnosis,
    binary_target_distribution,
    binary_target_distribution_by_split,
    dataset_summary,
    diagnosis_distribution,
    diagnosis_distribution_by_split,
    gender_distribution,
    gender_distribution_by_diagnosis,
    identify_eda_columns,
    metadata_warnings,
    mmse_availability,
    mmse_descriptive_statistics_by_diagnosis,
    prepare_eda_metadata,
    train_test_distribution,
)
from ml.data.process2_loader import (
    DatasetConfigurationError,
    DatasetValidationError,
    get_dataset_root,
    load_metadata,
)


EDA_OUTPUT_DIRECTORY = PROJECT_ROOT / "artifacts" / "results" / "eda"


def annotate_bars(axis: plt.Axes) -> None:
    """Add count labels to simple bar charts."""

    for container in axis.containers:
        axis.bar_label(container, fmt="%d", padding=3)


def save_bar_plot(series, title: str, x_label: str, output_path: Path) -> None:
    """Save a readable count bar chart with participant-count labels."""

    figure, axis = plt.subplots(figsize=(7, 4.5))
    series.plot(kind="bar", ax=axis, color="#4C78A8")
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Participant count")
    axis.tick_params(axis="x", rotation=0)
    annotate_bars(axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_grouped_bar_plot(table, title: str, x_label: str, output_path: Path) -> None:
    """Save a grouped categorical count chart with a legend."""

    figure, axis = plt.subplots(figsize=(8, 5))
    table.plot(kind="bar", ax=axis)
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Participant count")
    axis.tick_params(axis="x", rotation=0)
    axis.legend(title=table.columns.name or "Category")
    annotate_bars(axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_boxplot(table, value_label: str, title: str, output_path: Path) -> None:
    """Save a diagnosis-grouped box plot when numeric values are available."""

    values = [table.loc[index].dropna().to_numpy() for index in table.index]
    labels = [str(index) for index in table.index]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.boxplot(values, tick_labels=labels)
    axis.set_title(title)
    axis.set_xlabel("Diagnosis")
    axis.set_ylabel(value_label)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_figures(metadata, columns, output_directory: Path) -> None:
    """Generate the required non-decorative metadata EDA figures."""

    output_directory.mkdir(parents=True, exist_ok=True)
    diagnosis_counts = diagnosis_distribution(metadata, columns)
    binary_counts = binary_target_distribution(metadata)
    split_counts = train_test_distribution(metadata, columns)
    diagnosis_by_split = diagnosis_distribution_by_split(metadata, columns)
    binary_by_split = binary_target_distribution_by_split(metadata, columns)

    save_bar_plot(
        diagnosis_counts,
        "PROCESS-2 diagnosis distribution",
        "Diagnosis",
        output_directory / "diagnosis_distribution.png",
    )
    save_bar_plot(
        binary_counts,
        "PROCESS-2 binary screening-label distribution",
        "Screening label",
        output_directory / "binary_target_distribution.png",
    )
    save_bar_plot(
        split_counts,
        "PROCESS-2 train/test split distribution",
        "Split",
        output_directory / "split_distribution.png",
    )
    save_grouped_bar_plot(
        diagnosis_by_split,
        "Diagnosis distribution by split",
        "Split",
        output_directory / "diagnosis_by_split.png",
    )
    save_grouped_bar_plot(
        binary_by_split,
        "Binary screening-label distribution by split",
        "Split",
        output_directory / "binary_target_by_split.png",
    )

    if columns.age is not None:
        age_data = metadata[[columns.diagnosis, columns.age]].copy()
        age_data[columns.age] = age_data[columns.age].apply(lambda value: pd.to_numeric(value, errors="coerce"))
        grouped_age = [
            group[columns.age].dropna().to_numpy()
            for _, group in age_data.groupby(columns.diagnosis, dropna=False)
        ]
        labels = [str(label) for label, _ in age_data.groupby(columns.diagnosis, dropna=False)]
        if any(len(values) for values in grouped_age):
            figure, axis = plt.subplots(figsize=(8, 5))
            axis.boxplot(grouped_age, tick_labels=labels)
            axis.set_title("Age distribution by diagnosis")
            axis.set_xlabel("Diagnosis")
            axis.set_ylabel("Age")
            figure.tight_layout()
            figure.savefig(output_directory / "age_by_diagnosis.png", dpi=150, bbox_inches="tight")
            plt.close(figure)

    save_bar_plot(
        gender_distribution(metadata, columns),
        "Gender distribution",
        "Gender",
        output_directory / "gender_distribution.png",
    )
    save_grouped_bar_plot(
        gender_distribution_by_diagnosis(metadata, columns),
        "Gender distribution by diagnosis",
        "Diagnosis",
        output_directory / "gender_by_diagnosis.png",
    )
    save_bar_plot(
        mmse_availability(metadata, columns),
        "MMSE availability",
        "MMSE status",
        output_directory / "mmse_availability.png",
    )

    if columns.mmse is not None:
        mmse_data = metadata[[columns.diagnosis, columns.mmse]].copy()
        mmse_data[columns.mmse] = mmse_data[columns.mmse].apply(lambda value: pd.to_numeric(value, errors="coerce"))
        grouped_mmse = [
            group[columns.mmse].dropna().to_numpy()
            for _, group in mmse_data.groupby(columns.diagnosis, dropna=False)
        ]
        labels = [str(label) for label, _ in mmse_data.groupby(columns.diagnosis, dropna=False)]
        if any(len(values) for values in grouped_mmse):
            figure, axis = plt.subplots(figsize=(8, 5))
            axis.boxplot(grouped_mmse, tick_labels=labels)
            axis.set_title("MMSE distribution by diagnosis")
            axis.set_xlabel("Diagnosis")
            axis.set_ylabel("MMSE score")
            figure.tight_layout()
            figure.savefig(output_directory / "mmse_by_diagnosis.png", dpi=150, bbox_inches="tight")
            plt.close(figure)


def print_series(title: str, series) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(series.to_string())


def main() -> int:
    try:
        dataset_root = get_dataset_root()
        metadata = load_metadata(dataset_root)
        columns = identify_eda_columns(metadata)
    except (DatasetConfigurationError, DatasetValidationError, EDAConfigurationError, RuntimeError) as error:
        print(f"EDA could not continue: {error}")
        return 1

    prepared = prepare_eda_metadata(metadata, columns)
    summary = dataset_summary(prepared, columns)
    EDA_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    summary.to_csv(EDA_OUTPUT_DIRECTORY / "dataset_summary.csv", index=False)
    save_figures(prepared, columns, EDA_OUTPUT_DIRECTORY)

    print("PROCESS-2 Metadata EDA")
    print("======================")
    print(f"Dataset root: {dataset_root}")
    print(f"Rows: {len(prepared)}")
    print(f"Columns: {', '.join(str(column) for column in prepared.columns)}")
    print_series("Diagnosis distribution", diagnosis_distribution(prepared, columns))
    print_series("Binary screening-label distribution", binary_target_distribution(prepared))
    print_series("Train/test distribution", train_test_distribution(prepared, columns))
    print_series("Diagnosis distribution by split", diagnosis_distribution_by_split(prepared, columns))
    print_series("Binary screening-label distribution by split", binary_target_distribution_by_split(prepared, columns))
    print_series("Age descriptive statistics", age_descriptive_statistics(prepared, columns))
    print_series("Age statistics by diagnosis", age_distribution_by_diagnosis(prepared, columns))
    print_series("Gender distribution", gender_distribution(prepared, columns))
    print_series("Gender distribution by diagnosis", gender_distribution_by_diagnosis(prepared, columns))
    print_series("MMSE availability", mmse_availability(prepared, columns))
    print_series("MMSE statistics by diagnosis", mmse_descriptive_statistics_by_diagnosis(prepared, columns))

    warnings = metadata_warnings(prepared, columns)
    print("\nMetadata warnings")
    print("-----------------")
    if warnings:
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("None detected by the metadata-only checks.")
    print(f"\nSaved figures and dataset_summary.csv to: {EDA_OUTPUT_DIRECTORY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
