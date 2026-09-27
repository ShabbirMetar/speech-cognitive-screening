"""Reusable, metadata-only exploratory analysis for the external PROCESS-2 dataset.

All transformations in this module are in memory. The external metadata CSV is
loaded through ``process2_loader`` and is never changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ml.data.process2_loader import (
    DIAGNOSIS_COLUMN_CANDIDATES,
    SPLIT_COLUMN_CANDIDATES,
    identify_column,
    identify_metadata_columns,
)


SCREENING_LABEL_COLUMN = "screening_label"
SCREENING_LABELS = {"HC": "Healthy", "MCI": "Impaired", "DEMENTIA": "Impaired"}
AGE_COLUMN_CANDIDATES = ("age", "participantage")
GENDER_COLUMN_CANDIDATES = ("gender", "sex", "biologicalsex")
MMSE_COLUMN_CANDIDATES = ("mmse", "minimentalstateexamination")


class EDAConfigurationError(ValueError):
    """Raised when required metadata columns cannot be identified."""


@dataclass(frozen=True)
class EDAColumns:
    """Resolved metadata columns used by PROCESS-2 descriptive EDA."""

    diagnosis: str
    split: str
    age: str | None
    gender: str | None
    mmse: str | None


def identify_eda_columns(metadata: pd.DataFrame) -> EDAColumns:
    """Resolve actual metadata columns without depending on fixed CSV headers."""

    core_columns = identify_metadata_columns(metadata)
    diagnosis = core_columns.diagnosis or identify_column(
        metadata.columns, DIAGNOSIS_COLUMN_CANDIDATES
    )
    split = core_columns.split or identify_column(metadata.columns, SPLIT_COLUMN_CANDIDATES)
    if diagnosis is None or split is None:
        raise EDAConfigurationError(
            "PROCESS-2 EDA requires identifiable diagnosis and split columns."
        )

    return EDAColumns(
        diagnosis=diagnosis,
        split=split,
        age=identify_column(metadata.columns, AGE_COLUMN_CANDIDATES),
        gender=identify_column(metadata.columns, GENDER_COLUMN_CANDIDATES),
        mmse=identify_column(metadata.columns, MMSE_COLUMN_CANDIDATES),
    )


def create_screening_label(diagnosis: pd.Series) -> pd.Series:
    """Create the requested binary screening label without changing source values."""

    normalised = diagnosis.astype("string").str.strip().str.upper()
    return normalised.map(SCREENING_LABELS).astype("string").rename(SCREENING_LABEL_COLUMN)


def prepare_eda_metadata(metadata: pd.DataFrame, columns: EDAColumns) -> pd.DataFrame:
    """Return an in-memory copy enriched with the derived binary target."""

    prepared = metadata.copy(deep=True)
    prepared[SCREENING_LABEL_COLUMN] = create_screening_label(prepared[columns.diagnosis])
    return prepared


def _categorical_counts(values: pd.Series) -> pd.Series:
    """Count categories while retaining missing values as a visible category."""

    display_values = values.astype("string").str.strip().fillna("<missing>")
    display_values = display_values.mask(display_values.eq(""), "<missing>")
    return display_values.value_counts()


def diagnosis_distribution(metadata: pd.DataFrame, columns: EDAColumns) -> pd.Series:
    """Return the diagnosis category counts for the supplied metadata."""

    return _categorical_counts(metadata[columns.diagnosis])


def binary_target_distribution(metadata: pd.DataFrame) -> pd.Series:
    """Return counts of the in-memory Healthy/Impaired screening target."""

    counts = _categorical_counts(metadata[SCREENING_LABEL_COLUMN])
    preferred_order = ["Healthy", "Impaired", "<missing>"]
    return counts.reindex([label for label in preferred_order if label in counts.index])


def train_test_distribution(metadata: pd.DataFrame, columns: EDAColumns) -> pd.Series:
    """Return counts of the observed train/test split labels."""

    return _categorical_counts(metadata[columns.split])


def diagnosis_distribution_by_split(metadata: pd.DataFrame, columns: EDAColumns) -> pd.DataFrame:
    """Return diagnosis counts grouped by the observed split labels."""

    return pd.crosstab(
        metadata[columns.split].astype("string").str.strip().fillna("<missing>"),
        metadata[columns.diagnosis].astype("string").str.strip().fillna("<missing>"),
    )


def binary_target_distribution_by_split(
    metadata: pd.DataFrame, columns: EDAColumns
) -> pd.DataFrame:
    """Return binary screening-label counts grouped by the observed split labels."""

    return pd.crosstab(
        metadata[columns.split].astype("string").str.strip().fillna("<missing>"),
        metadata[SCREENING_LABEL_COLUMN].astype("string").str.strip().fillna("<missing>"),
    )


def _numeric_values(metadata: pd.DataFrame, column_name: str | None) -> pd.Series:
    """Read a numeric metadata column in memory, preserving invalid values as null."""

    if column_name is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(metadata[column_name], errors="coerce")


def age_descriptive_statistics(metadata: pd.DataFrame, columns: EDAColumns) -> pd.Series:
    """Return descriptive statistics for age, excluding missing numeric values."""

    return _numeric_values(metadata, columns.age).describe()


def age_distribution_by_diagnosis(metadata: pd.DataFrame, columns: EDAColumns) -> pd.DataFrame:
    """Return age descriptive statistics grouped by diagnosis."""

    if columns.age is None:
        return pd.DataFrame()
    frame = pd.DataFrame(
        {
            "diagnosis": metadata[columns.diagnosis],
            "age": _numeric_values(metadata, columns.age),
        }
    )
    return frame.groupby("diagnosis", dropna=False)["age"].describe()


def gender_distribution(metadata: pd.DataFrame, columns: EDAColumns) -> pd.Series:
    """Return overall gender counts, including missing values when present."""

    if columns.gender is None:
        return pd.Series(dtype="int64")
    return _categorical_counts(metadata[columns.gender])


def gender_distribution_by_diagnosis(metadata: pd.DataFrame, columns: EDAColumns) -> pd.DataFrame:
    """Return gender counts grouped by diagnosis."""

    if columns.gender is None:
        return pd.DataFrame()
    return pd.crosstab(
        metadata[columns.diagnosis].astype("string").str.strip().fillna("<missing>"),
        metadata[columns.gender].astype("string").str.strip().fillna("<missing>"),
    )


def mmse_availability(metadata: pd.DataFrame, columns: EDAColumns) -> pd.Series:
    """Report available and missing MMSE values without filling or changing them."""

    mmse = _numeric_values(metadata, columns.mmse)
    if columns.mmse is None:
        return pd.Series({"Available": 0, "Missing": len(metadata)}, dtype="int64")
    return pd.Series({"Available": int(mmse.notna().sum()), "Missing": int(mmse.isna().sum())})


def mmse_descriptive_statistics_by_diagnosis(
    metadata: pd.DataFrame, columns: EDAColumns
) -> pd.DataFrame:
    """Return MMSE statistics by diagnosis while preserving missing MMSE values."""

    if columns.mmse is None:
        return pd.DataFrame()
    frame = pd.DataFrame(
        {
            "diagnosis": metadata[columns.diagnosis],
            "mmse": _numeric_values(metadata, columns.mmse),
        }
    )
    return frame.groupby("diagnosis", dropna=False)["mmse"].describe()


def dataset_summary(metadata: pd.DataFrame, columns: EDAColumns) -> pd.DataFrame:
    """Build a compact, exportable metadata-only dataset summary table."""

    in_memory_label_present = SCREENING_LABEL_COLUMN in metadata.columns
    rows: list[dict[str, Any]] = [
        {"metric": "metadata_rows", "value": len(metadata)},
        {
            "metric": "source_metadata_columns",
            "value": len(metadata.columns) - int(in_memory_label_present),
        },
        {"metric": "in_memory_columns", "value": len(metadata.columns)},
        {"metric": "diagnosis_column", "value": columns.diagnosis},
        {"metric": "split_column", "value": columns.split},
        {"metric": "age_column", "value": columns.age or "<not identified>"},
        {"metric": "gender_column", "value": columns.gender or "<not identified>"},
        {"metric": "mmse_column", "value": columns.mmse or "<not identified>"},
    ]
    for label, count in diagnosis_distribution(metadata, columns).items():
        rows.append({"metric": f"diagnosis_count_{label}", "value": int(count)})
    for label, count in binary_target_distribution(metadata).items():
        rows.append({"metric": f"screening_label_count_{label}", "value": int(count)})
    for label, count in train_test_distribution(metadata, columns).items():
        rows.append({"metric": f"split_count_{label}", "value": int(count)})
    for label, count in mmse_availability(metadata, columns).items():
        rows.append({"metric": f"mmse_{str(label).lower()}", "value": int(count)})
    return pd.DataFrame(rows)


def metadata_warnings(metadata: pd.DataFrame, columns: EDAColumns) -> list[str]:
    """Return non-mutating data-quality notices for the EDA console summary."""

    warnings: list[str] = []
    unknown_labels = int(metadata[SCREENING_LABEL_COLUMN].isna().sum())
    if unknown_labels:
        warnings.append(f"{unknown_labels} rows have a missing or unmapped diagnosis label.")
    if columns.age is None:
        warnings.append("No age column was identified.")
    elif int(_numeric_values(metadata, columns.age).isna().sum()):
        warnings.append("Age contains missing or non-numeric values.")
    if columns.gender is None:
        warnings.append("No gender column was identified.")
    elif int(_categorical_counts(metadata[columns.gender]).get("<missing>", 0)):
        warnings.append("Gender contains missing values.")
    if columns.mmse is None:
        warnings.append("No MMSE column was identified.")
    else:
        missing_mmse = int(mmse_availability(metadata, columns)["Missing"])
        if missing_mmse:
            warnings.append(f"MMSE has {missing_mmse} missing or non-numeric values.")
    return warnings


# Data-leakage rule: TEST is descriptive-only here. Predictive feature analysis,
# feature selection, model fitting, and hyperparameter tuning must use TRAIN only.
