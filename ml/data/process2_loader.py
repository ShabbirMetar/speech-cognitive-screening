"""Read-only inspection helpers for the external PROCESS-2 dataset.

This module never writes to the dataset.  It only reads paths, metadata, and file
names needed to validate the dataset layout before any research processing begins.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import math
import re
from typing import Any, Iterable, Sequence


METADATA_FILENAME = "meta-info.csv"
EXPECTED_TASKS = ("SFT", "PFT", "CTD")
SUPPORTED_TASK_SUFFIXES = {".wav", ".txt"}

PARTICIPANT_ID_CANDIDATES = (
    "id",
    "ids",
    "participantid",
    "participant",
    "subjectid",
    "subject",
    "recordingid",
)
DIAGNOSIS_COLUMN_CANDIDATES = (
    "diagnosis",
    "diagnosticlabel",
    "diagnosticgroup",
    "label",
    "class",
)
SPLIT_COLUMN_CANDIDATES = (
    "split",
    "traintestsplit",
    "dataset split",
    "set",
    "partition",
)


class DatasetConfigurationError(RuntimeError):
    """Raised when PROCESS-2 has not been configured for local access."""


class DatasetValidationError(RuntimeError):
    """Raised when a required PROCESS-2 path is unavailable."""


@dataclass(frozen=True)
class DatasetStatus:
    """Availability of the configured dataset root and metadata file."""

    dataset_root: Path
    dataset_exists: bool
    metadata_path: Path
    metadata_exists: bool


@dataclass(frozen=True)
class MetadataColumns:
    """Resolved metadata columns, or ``None`` when a column could not be found."""

    participant_id: str | None
    diagnosis: str | None
    split: str | None


@dataclass(frozen=True)
class TaskFileCounts:
    """Counts of task-labelled files, split by media type."""

    total: int
    wav: int
    txt: int


@dataclass(frozen=True)
class FileCounts:
    """Read-only counts gathered from the dataset directory tree."""

    participant_directories: int
    wav_files: int
    txt_files: int
    task_files: dict[str, TaskFileCounts]


@dataclass(frozen=True)
class IntegritySummary:
    """Participant-level metadata and expected-task consistency results."""

    total_participants: int
    complete_participants: int
    incomplete_participants: int
    duplicate_participant_ids: int
    missing_diagnosis: int
    missing_split: int
    participants_without_matching_directory: int


def get_dataset_root() -> Path:
    """Return the configured PROCESS-2 root via the application settings.

    Importing settings lazily keeps pure helper functions testable without loading
    the backend application or a local ``.env`` file.
    """

    from app.core.config import get_settings

    dataset_root = get_settings().process2_dataset_path
    if dataset_root is None:
        raise DatasetConfigurationError(
            "PROCESS2_DATASET_PATH is not configured. Set it in the local .env file."
        )
    return Path(dataset_root)


def get_metadata_path(dataset_root: Path) -> Path:
    """Return the expected metadata path without reading or changing it."""

    return dataset_root / METADATA_FILENAME


def get_dataset_status(dataset_root: Path) -> DatasetStatus:
    """Check required paths without inspecting participant data."""

    metadata_path = get_metadata_path(dataset_root)
    return DatasetStatus(
        dataset_root=dataset_root,
        dataset_exists=dataset_root.is_dir(),
        metadata_path=metadata_path,
        metadata_exists=metadata_path.is_file(),
    )


def require_accessible_metadata(dataset_root: Path) -> Path:
    """Validate required paths and return the metadata path for read-only loading."""

    status = get_dataset_status(dataset_root)
    if not status.dataset_exists:
        raise DatasetValidationError(f"Dataset root does not exist: {dataset_root}")
    if not status.metadata_exists:
        raise DatasetValidationError(f"Metadata file does not exist: {status.metadata_path}")
    return status.metadata_path


def load_metadata(dataset_root: Path) -> Any:
    """Load ``meta-info.csv`` with pandas in read-only mode."""

    try:
        import pandas as pd
    except ImportError as error:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "pandas is required for PROCESS-2 metadata validation. "
            "Install backend/requirements.txt first."
        ) from error

    return pd.read_csv(require_accessible_metadata(dataset_root))


def normalise_column_name(column_name: object) -> str:
    """Normalise a column name for schema-tolerant matching."""

    return re.sub(r"[^a-z0-9]+", "", str(column_name).casefold())


def identify_column(
    column_names: Iterable[object], candidates: Sequence[str]
) -> str | None:
    """Find a real metadata column from a list of semantic candidate names."""

    columns = [str(column_name) for column_name in column_names]
    normalised = {normalise_column_name(column_name): column_name for column_name in columns}

    for candidate in candidates:
        match = normalised.get(normalise_column_name(candidate))
        if match is not None:
            return match

    for column_name in columns:
        normalised_column = normalise_column_name(column_name)
        if any(
            normalise_column_name(candidate) in normalised_column
            for candidate in candidates
        ):
            return column_name
    return None


def identify_metadata_columns(metadata: Any) -> MetadataColumns:
    """Resolve participant ID, diagnosis, and split columns from actual CSV headers."""

    return MetadataColumns(
        participant_id=identify_column(metadata.columns, PARTICIPANT_ID_CANDIDATES),
        diagnosis=identify_column(metadata.columns, DIAGNOSIS_COLUMN_CANDIDATES),
        split=identify_column(metadata.columns, SPLIT_COLUMN_CANDIDATES),
    )


def is_missing_value(value: object) -> bool:
    """Treat nulls and blank strings as missing without altering metadata."""

    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def non_hidden_directories(dataset_root: Path) -> list[Path]:
    """Return top-level, non-hidden directories without assuming their names."""

    if not dataset_root.is_dir():
        return []
    return sorted(
        (path for path in dataset_root.iterdir() if path.is_dir() and not path.name.startswith(".")),
        key=lambda path: path.name.casefold(),
    )


def is_in_hidden_path(path: Path, dataset_root: Path) -> bool:
    """Return whether a path is below a hidden file or directory."""

    return any(part.startswith(".") for part in path.relative_to(dataset_root).parts)


def has_expected_task_file(directory: Path) -> bool:
    """Identify a participant folder from its task-labelled audio or transcript files."""

    for path in directory.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in SUPPORTED_TASK_SUFFIXES:
            continue
        if any(task in path.name.upper() for task in EXPECTED_TASKS):
            return True
    return False


def participant_directories(dataset_root: Path) -> list[Path]:
    """Return non-hidden top-level directories containing at least one task file."""

    return [
        directory
        for directory in non_hidden_directories(dataset_root)
        if has_expected_task_file(directory)
    ]


def collect_file_counts(dataset_root: Path) -> FileCounts:
    """Count supported media/transcript files and expected task labels read-only."""

    counts = {task: Counter() for task in EXPECTED_TASKS}
    wav_files = 0
    txt_files = 0

    if dataset_root.is_dir():
        for path in dataset_root.rglob("*"):
            if not path.is_file() or is_in_hidden_path(path, dataset_root):
                continue
            suffix = path.suffix.casefold()
            if suffix == ".wav":
                wav_files += 1
            elif suffix == ".txt":
                txt_files += 1
            else:
                continue

            upper_name = path.name.upper()
            for task in EXPECTED_TASKS:
                if task in upper_name:
                    counts[task][suffix.removeprefix(".")] += 1

    task_files = {
        task: TaskFileCounts(
            total=task_counts["wav"] + task_counts["txt"],
            wav=task_counts["wav"],
            txt=task_counts["txt"],
        )
        for task, task_counts in counts.items()
    }
    return FileCounts(
        participant_directories=len(participant_directories(dataset_root)),
        wav_files=wav_files,
        txt_files=txt_files,
        task_files=task_files,
    )


def identifier_keys(value: object) -> set[str]:
    """Create tolerant matching keys for metadata IDs and folder names.

    The original compact value is retained, and a trailing numeric component is
    added when present so values such as ``001`` and a folder ending in ``001``
    can be matched without encoding a dataset-specific naming convention.
    """

    if is_missing_value(value):
        return set()
    raw_value = str(value).strip()
    compact = re.sub(r"[^A-Z0-9]+", "", raw_value.upper())
    keys = {compact} if compact else set()
    numeric_parts = re.findall(r"\d+", raw_value)
    if numeric_parts:
        keys.add(str(int(numeric_parts[-1])))
    return keys


def build_directory_index(participant_directories: Iterable[Path]) -> dict[str, list[Path]]:
    """Index participant directories with tolerant, non-identifying match keys."""

    index: dict[str, list[Path]] = {}
    for directory in participant_directories:
        for key in identifier_keys(directory.name):
            index.setdefault(key, []).append(directory)
    return index


def task_data_is_complete(participant_directory: Path) -> bool:
    """Check for WAV and TXT files for SFT, PFT, and CTD in one directory."""

    found: dict[str, set[str]] = {task: set() for task in EXPECTED_TASKS}
    for path in participant_directory.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in SUPPORTED_TASK_SUFFIXES:
            continue
        upper_name = path.name.upper()
        for task in EXPECTED_TASKS:
            if task in upper_name:
                found[task].add(path.suffix.casefold())

    return all({".wav", ".txt"}.issubset(found[task]) for task in EXPECTED_TASKS)


def value_counts(metadata: Any, column_name: str | None) -> dict[str, int]:
    """Return display-safe value counts for a resolved categorical metadata column."""

    if column_name is None:
        return {}
    values = metadata[column_name].where(~metadata[column_name].isna(), "<missing>")
    values = values.astype(str).map(lambda value: value.strip() or "<missing>")
    return {str(value): int(count) for value, count in values.value_counts().items()}


def missing_value_count(metadata: Any, column_name: str | None) -> int:
    """Count null and blank values for an optional metadata column."""

    if column_name is None:
        return len(metadata)
    values = metadata[column_name]
    null_values = values.isna()
    blank_values = values.astype("string").str.strip().eq("").fillna(False)
    return int((null_values | blank_values).sum())


def validate_participant_integrity(
    dataset_root: Path, metadata: Any, columns: MetadataColumns
) -> IntegritySummary:
    """Compare metadata participants with expected SFT/PFT/CTD task files.

    Participant IDs are never printed.  The result contains only aggregate counts.
    """

    missing_diagnosis = missing_value_count(metadata, columns.diagnosis)
    missing_split = missing_value_count(metadata, columns.split)
    if columns.participant_id is None:
        return IntegritySummary(
            total_participants=len(metadata),
            complete_participants=0,
            incomplete_participants=len(metadata),
            duplicate_participant_ids=0,
            missing_diagnosis=missing_diagnosis,
            missing_split=missing_split,
            participants_without_matching_directory=len(metadata),
        )

    participant_values = list(metadata[columns.participant_id])
    participant_key_groups: dict[tuple[str, ...], object] = {}
    duplicate_keys: set[tuple[str, ...]] = set()
    missing_identifier_rows = 0
    for value in participant_values:
        keys = tuple(sorted(identifier_keys(value)))
        if not keys:
            missing_identifier_rows += 1
            continue
        if keys in participant_key_groups:
            duplicate_keys.add(keys)
        else:
            participant_key_groups[keys] = value

    directory_index = build_directory_index(participant_directories(dataset_root))
    complete_participants = 0
    unmatched_participants = 0
    for keys in participant_key_groups:
        matching_directories = {
            directory for key in keys for directory in directory_index.get(key, [])
        }
        if len(matching_directories) != 1:
            unmatched_participants += 1
            continue
        if task_data_is_complete(next(iter(matching_directories))):
            complete_participants += 1

    total_participants = len(participant_key_groups) + missing_identifier_rows
    incomplete_participants = total_participants - complete_participants
    return IntegritySummary(
        total_participants=total_participants,
        complete_participants=complete_participants,
        incomplete_participants=incomplete_participants,
        duplicate_participant_ids=len(duplicate_keys),
        missing_diagnosis=missing_diagnosis,
        missing_split=missing_split,
        participants_without_matching_directory=unmatched_participants + missing_identifier_rows,
    )
