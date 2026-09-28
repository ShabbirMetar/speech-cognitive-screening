"""Build deterministic participant-level features from manual PROCESS-2 transcripts."""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
for import_root in (PROJECT_ROOT, BACKEND_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from ml.data.process2_eda import create_screening_label
from ml.data.process2_loader import (
    DatasetConfigurationError,
    DatasetValidationError,
    build_directory_index,
    get_dataset_root,
    identifier_keys,
    identify_metadata_columns,
    load_metadata,
    participant_directories,
)
from ml.nlp.linguistic_features import (
    extract_linguistic_features,
    p_initial_features,
    unavailable_linguistic_features,
)
from ml.nlp.transcript_parser import (
    TASK_NAMES,
    parse_transcript_file,
    transcript_paths_by_task,
)


FEATURE_OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "features" / "manual_transcript_features.csv"
BOOKKEEPING_COLUMNS = {"participant_id", "diagnosis", "screening_label", "Split"}


def prefixed_features(task_name: str, features: dict[str, float | int]) -> dict[str, float | int]:
    """Prefix a task's features for a one-row-per-participant table."""

    return {f"{task_name.casefold()}_{name}": value for name, value in features.items()}


def features_for_task(
    transcript_path: Path | None, task_name: str
) -> tuple[dict[str, float | int], str]:
    """Return task features and a non-predictive transcript-quality status."""

    if transcript_path is None:
        features: dict[str, float | int] = unavailable_linguistic_features()
        status = "missing_transcript"
    else:
        parsed = parse_transcript_file(transcript_path)
        features = extract_linguistic_features(parsed)
        status = parsed.speaker_attribution
    if task_name == "PFT":
        if transcript_path is None or status == "no_participant_label":
            features.update({"p_initial_word_count": float("nan"), "p_initial_ratio": float("nan")})
        else:
            features.update(p_initial_features(parsed))
    return prefixed_features(task_name, features), status


def build_feature_table() -> tuple[pd.DataFrame, list[str], Counter[str]]:
    """Build one reference-and-feature row per metadata participant.

    Deterministic feature extraction is applied to all participants. No statistic is
    learned here; later scaling, imputation, selection, and modelling must use only
    TRAIN participants.
    """

    dataset_root = get_dataset_root()
    metadata = load_metadata(dataset_root)
    columns = identify_metadata_columns(metadata)
    if columns.participant_id is None or columns.diagnosis is None or columns.split is None:
        raise ValueError("Feature building requires participant ID, diagnosis, and split columns.")

    directory_index = build_directory_index(participant_directories(dataset_root))
    screening_labels = create_screening_label(metadata[columns.diagnosis])
    rows: list[dict[str, object]] = []
    warnings: list[str] = []
    quality_status_counts: Counter[str] = Counter()
    for row_index, metadata_row in metadata.iterrows():
        matching_directories = {
            directory
            for key in identifier_keys(metadata_row[columns.participant_id])
            for directory in directory_index.get(key, [])
        }
        row: dict[str, object] = {
            "participant_id": metadata_row[columns.participant_id],
            "diagnosis": metadata_row[columns.diagnosis],
            "screening_label": screening_labels.loc[row_index],
            "Split": metadata_row[columns.split],
        }
        if len(matching_directories) != 1:
            warnings.append("A metadata participant did not have a unique transcript directory.")
            transcripts = {task: [] for task in TASK_NAMES}
        else:
            transcripts = transcript_paths_by_task(next(iter(matching_directories)))

        for task_name in TASK_NAMES:
            paths = transcripts[task_name]
            if len(paths) != 1:
                warnings.append(f"A participant did not have exactly one {task_name} transcript.")
            try:
                task_features, quality_status = features_for_task(
                    paths[0] if len(paths) == 1 else None, task_name
                )
                row.update(task_features)
                quality_status_counts[f"{task_name}_{quality_status}"] += 1
            except (OSError, UnicodeError):
                warnings.append(f"A {task_name} transcript could not be read with UTF-8.")
                task_features, quality_status = features_for_task(None, task_name)
                row.update(task_features)
                quality_status_counts[f"{task_name}_{quality_status}"] += 1
        rows.append(row)

    return pd.DataFrame(rows), warnings, quality_status_counts


def feature_columns(feature_table: pd.DataFrame) -> list[str]:
    """Return only permitted task-specific numeric feature columns."""

    return [column for column in feature_table.columns if column not in BOOKKEEPING_COLUMNS]


def print_feature_quality_report(
    feature_table: pd.DataFrame, quality_status_counts: Counter[str]
) -> None:
    """Print feature integrity and TRAIN-only descriptive statistics."""

    features = feature_columns(feature_table)
    numeric_features = feature_table[features].apply(pd.to_numeric, errors="coerce")
    split_values = feature_table["Split"].astype("string").str.strip().str.upper()
    labels = feature_table["screening_label"].astype("string")
    train_mask = split_values.eq("TRAIN")
    test_mask = split_values.eq("TEST")
    missing_counts = numeric_features.isna().sum()
    constant_features = [
        column for column in features if numeric_features.loc[train_mask, column].nunique(dropna=True) <= 1
    ]
    infinite_features = [
        column
        for column in features
        if numeric_features[column].map(lambda value: math.isinf(value) if pd.notna(value) else False).any()
    ]
    train_statistics = numeric_features.loc[train_mask, features].describe().T
    train_statistics = train_statistics[["count", "mean", "std", "min", "50%", "max"]]

    print("Manual Transcript Feature Quality Report")
    print("========================================")
    print(f"Participant rows: {len(feature_table)}")
    print(f"Feature columns: {len(features)}")
    print(f"TRAIN participant count: {int(train_mask.sum())}")
    print(f"TEST participant count: {int(test_mask.sum())}")
    print(f"Healthy count: {int(labels.eq('Healthy').sum())}")
    print(f"Impaired count: {int(labels.eq('Impaired').sum())}")
    print("\nNon-predictive transcript-quality status counts")
    for status, count in sorted(quality_status_counts.items()):
        print(f"{status}: {count}")
    print("\nMissing-value count by feature")
    print(missing_counts.to_string())
    print("\nConstant TRAIN features")
    print(", ".join(constant_features) if constant_features else "None")
    print("\nFeatures containing infinite values")
    print(", ".join(infinite_features) if infinite_features else "None")
    print("\nTRAIN-only descriptive statistics")
    print(train_statistics.to_string())


def main() -> int:
    try:
        feature_table, warnings, quality_status_counts = build_feature_table()
    except (DatasetConfigurationError, DatasetValidationError, RuntimeError, ValueError) as error:
        print(f"Feature build could not continue: {error}")
        return 1

    FEATURE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(FEATURE_OUTPUT_PATH, index=False)
    print_feature_quality_report(feature_table, quality_status_counts)
    print(f"\nSaved feature table: {FEATURE_OUTPUT_PATH}")
    if warnings:
        print("\nTranscript build warnings")
        for warning in sorted(set(warnings)):
            print(f"- {warning}")
    else:
        print("\nTranscript build warnings: None")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
