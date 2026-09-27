"""Print a read-only PROCESS-2 metadata and file-layout validation summary."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
for import_root in (PROJECT_ROOT, BACKEND_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from ml.data.process2_loader import (
    DatasetConfigurationError,
    DatasetValidationError,
    collect_file_counts,
    get_dataset_root,
    get_dataset_status,
    identify_metadata_columns,
    load_metadata,
    validate_participant_integrity,
    value_counts,
)


def yes_no(value: bool) -> str:
    return "YES" if value else "NO"


def print_distribution(title: str, distribution: dict[str, int], column_name: str | None) -> None:
    print(f"{title}:")
    if column_name is None:
        print("  Not identified")
        return
    print(f"  Column: {column_name}")
    for value, count in distribution.items():
        print(f"  {value}: {count}")


def main() -> int:
    print("PROCESS-2 Dataset Validation")
    print("============================")

    try:
        dataset_root = get_dataset_root()
        status = get_dataset_status(dataset_root)
        print(f"\nDataset root: {status.dataset_root}")
        print(f"Dataset accessible: {yes_no(status.dataset_exists)}")
        print(f"Metadata found: {yes_no(status.metadata_exists)}")

        metadata = load_metadata(dataset_root)
    except (DatasetConfigurationError, DatasetValidationError, RuntimeError) as error:
        print(f"\nValidation could not continue: {error}")
        return 1

    columns = identify_metadata_columns(metadata)
    file_counts = collect_file_counts(dataset_root)
    integrity = validate_participant_integrity(dataset_root, metadata, columns)

    print("\nMetadata")
    print("--------")
    print(f"Rows: {len(metadata)}")
    print("Columns: " + ", ".join(str(column) for column in metadata.columns))
    print_distribution("\nDiagnosis distribution", value_counts(metadata, columns.diagnosis), columns.diagnosis)
    print_distribution("\nSplit distribution", value_counts(metadata, columns.split), columns.split)

    print("\nFiles")
    print("-----")
    print(f"Participant folders: {file_counts.participant_directories}")
    print(f"WAV files: {file_counts.wav_files}")
    print(f"TXT transcript files: {file_counts.txt_files}")
    for task in ("SFT", "PFT", "CTD"):
        counts = file_counts.task_files[task]
        print(f"{task}: {counts.total} (WAV: {counts.wav}, TXT: {counts.txt})")

    print("\nIntegrity")
    print("---------")
    print(f"Total participants: {integrity.total_participants}")
    print(f"Complete participants: {integrity.complete_participants}")
    print(f"Incomplete participants: {integrity.incomplete_participants}")
    print(f"Duplicate participant IDs: {integrity.duplicate_participant_ids}")
    print(f"Missing diagnosis: {integrity.missing_diagnosis}")
    print(f"Missing split: {integrity.missing_split}")
    print(
        "Participants without a unique matching directory: "
        f"{integrity.participants_without_matching_directory}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
