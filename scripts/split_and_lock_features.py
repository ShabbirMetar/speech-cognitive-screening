"""Create immutable-in-practice TRAIN and locked TEST views of frozen feature tables.

This script only partitions existing derived CSVs by their existing ``Split``
column.  It neither invokes an extractor nor reads any PROCESS-2 source file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


FEATURE_DIRECTORY = PROJECT_ROOT / "artifacts" / "features"
PARTICIPANT_ID_COLUMN = "participant_id"
SPLIT_COLUMN = "Split"
EXPECTED_TRAIN_ROWS = 320
EXPECTED_TEST_ROWS = 80


@dataclass(frozen=True)
class LockedSplit:
    """The source-preserving TRAIN and TEST partition for one feature modality."""

    train: pd.DataFrame
    test_locked: pd.DataFrame


def _normalized_split(feature_table: pd.DataFrame) -> pd.Series:
    return feature_table[SPLIT_COLUMN].astype("string").str.strip().str.upper()


def validate_source_table(feature_table: pd.DataFrame, table_name: str) -> None:
    """Validate split bookkeeping without deriving or modifying any features."""

    required_columns = {PARTICIPANT_ID_COLUMN, SPLIT_COLUMN}
    missing_columns = required_columns.difference(feature_table.columns)
    if missing_columns:
        raise ValueError(f"{table_name} is missing required columns: {sorted(missing_columns)}")
    if feature_table[PARTICIPANT_ID_COLUMN].isna().any():
        raise ValueError(f"{table_name} contains missing participant IDs.")
    if feature_table[PARTICIPANT_ID_COLUMN].duplicated().any():
        raise ValueError(f"{table_name} contains duplicate participant IDs.")

    splits = _normalized_split(feature_table)
    invalid = splits[~splits.isin(["TRAIN", "TEST"])].dropna().unique().tolist()
    if invalid or splits.isna().any():
        raise ValueError(f"{table_name} contains unsupported or missing Split values.")


def partition_feature_table(feature_table: pd.DataFrame, table_name: str) -> LockedSplit:
    """Partition a frozen feature table without modifying its columns or values."""

    validate_source_table(feature_table, table_name)
    splits = _normalized_split(feature_table)
    train = feature_table.loc[splits.eq("TRAIN")].copy()
    test_locked = feature_table.loc[splits.eq("TEST")].copy()
    if list(train.columns) != list(feature_table.columns):
        raise AssertionError(f"{table_name} TRAIN schema differs from its source schema.")
    if list(test_locked.columns) != list(feature_table.columns):
        raise AssertionError(f"{table_name} TEST schema differs from its source schema.")
    if set(train[PARTICIPANT_ID_COLUMN]).intersection(test_locked[PARTICIPANT_ID_COLUMN]):
        raise AssertionError(f"{table_name} TRAIN and TEST participant IDs overlap.")
    return LockedSplit(train=train, test_locked=test_locked)


def _id_set(feature_table: pd.DataFrame) -> set[str]:
    return set(feature_table[PARTICIPANT_ID_COLUMN].astype(str))


def create_locked_feature_files() -> dict[str, LockedSplit]:
    """Write locked views while leaving both full source feature tables untouched."""

    sources = {
        "linguistic": FEATURE_DIRECTORY / "manual_transcript_features.csv",
        "acoustic": FEATURE_DIRECTORY / "acoustic_features.csv",
    }
    outputs = {
        "linguistic": (
            FEATURE_DIRECTORY / "linguistic_features_train.csv",
            FEATURE_DIRECTORY / "linguistic_features_test_LOCKED.csv",
        ),
        "acoustic": (
            FEATURE_DIRECTORY / "acoustic_features_train.csv",
            FEATURE_DIRECTORY / "acoustic_features_test_LOCKED.csv",
        ),
    }
    partitions: dict[str, LockedSplit] = {}
    for modality, source_path in sources.items():
        if not source_path.is_file():
            raise FileNotFoundError(f"Feature source table not found: {source_path}")
        partitions[modality] = partition_feature_table(pd.read_csv(source_path), modality)

    if _id_set(partitions["linguistic"].train) != _id_set(partitions["acoustic"].train):
        raise AssertionError("Linguistic and acoustic TRAIN participant IDs do not match.")
    if _id_set(partitions["linguistic"].test_locked) != _id_set(partitions["acoustic"].test_locked):
        raise AssertionError("Linguistic and acoustic TEST participant IDs do not match.")
    for modality, partition in partitions.items():
        if len(partition.train) != EXPECTED_TRAIN_ROWS:
            raise AssertionError(
                f"{modality} TRAIN rows must equal {EXPECTED_TRAIN_ROWS}; found {len(partition.train)}."
            )
        if len(partition.test_locked) != EXPECTED_TEST_ROWS:
            raise AssertionError(
                f"{modality} TEST rows must equal {EXPECTED_TEST_ROWS}; found {len(partition.test_locked)}."
            )

    FEATURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for modality, partition in partitions.items():
        train_path, test_path = outputs[modality]
        partition.train.to_csv(train_path, index=False)
        partition.test_locked.to_csv(test_path, index=False)
    return partitions


def main() -> int:
    """Create the locked views and print only split-integrity counts."""

    partitions = create_locked_feature_files()
    linguistic = partitions["linguistic"]
    acoustic = partitions["acoustic"]
    train_test_overlap = len(
        _id_set(linguistic.train).intersection(_id_set(linguistic.test_locked))
    )
    print(f"Linguistic TRAIN rows: {len(linguistic.train)}")
    print(f"Linguistic TEST_LOCKED rows: {len(linguistic.test_locked)}")
    print()
    print(f"Acoustic TRAIN rows: {len(acoustic.train)}")
    print(f"Acoustic TEST_LOCKED rows: {len(acoustic.test_locked)}")
    print()
    print(f"TRAIN/TEST ID overlap: {train_test_overlap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
