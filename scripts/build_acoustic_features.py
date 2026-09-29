"""Build read-only, checkpointed PROCESS-2 acoustic features with frozen settings.

The frozen extractor may deterministically process TRAIN and TEST recordings, but
TEST labels must never influence feature choices, thresholds, or configuration.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
for import_root in (PROJECT_ROOT, BACKEND_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from ml.audio.acoustic_features import (
    ACOUSTIC_EXTRACTOR_VERSION,
    ACOUSTIC_FEATURE_NAMES,
    AcousticFeatureConfig,
    empty_acoustic_features,
    extract_acoustic_feature_result,
    speech_rate_features,
)
from ml.audio.audio_loader import AudioValidationError, load_wav_read_only
from ml.data.process2_loader import (
    EXPECTED_TASKS,
    build_directory_index,
    get_dataset_root,
    identifier_keys,
    non_hidden_directories,
)


MANUAL_FEATURE_TABLE_PATH = PROJECT_ROOT / "artifacts" / "features" / "manual_transcript_features.csv"
OUTPUT_DIRECTORY = PROJECT_ROOT / "artifacts" / "features"
OUTPUT_PATH = OUTPUT_DIRECTORY / "acoustic_features.csv"
CHECKPOINT_PATH = OUTPUT_DIRECTORY / "acoustic_features_checkpoint.csv"
CHECKPOINT_CONFIG_PATH = OUTPUT_DIRECTORY / "acoustic_features_checkpoint_config.json"
CONFIG_OUTPUT_PATH = OUTPUT_DIRECTORY / "acoustic_feature_config.json"
TASKS = tuple(EXPECTED_TASKS)
CHECKPOINT_INTERVAL = 20
BOOKKEEPING_COLUMNS = ("participant_id", "diagnosis", "screening_label", "Split")
RECORDING_STATUS_COLUMNS = (
    "_status",
    "_sample_rate",
    "_channel_count",
    "_pitch_failed",
    "_pause_failed",
)


def frozen_configuration_payload(config: AcousticFeatureConfig | None = None) -> dict[str, object]:
    """Return the participant-free frozen extraction configuration."""

    active_config = config or AcousticFeatureConfig()
    return {
        "extractor_version": ACOUSTIC_EXTRACTOR_VERSION,
        "sample_rate_behavior": "Preserve the native WAV sample rate; do not resample.",
        "channel_behavior": "Retain source channel count; average channels in memory only for analysis.",
        "librosa_parameters": asdict(active_config),
        "mfcc_coefficients": active_config.mfcc_count,
        "acoustic_feature_names": list(ACOUSTIC_FEATURE_NAMES),
        "test_policy": (
            "Deterministic frozen extraction may run on TEST recordings, but TEST labels must not "
            "influence feature selection, thresholds, pitch settings, tuning, or model comparison."
        ),
    }


def configuration_fingerprint(configuration: dict[str, object]) -> str:
    """Hash participant-free frozen settings to prevent unsafe checkpoint mixing."""

    canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def write_configuration_record(configuration: dict[str, object]) -> str:
    """Persist the frozen configuration without participant data and return its fingerprint."""

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    fingerprint = configuration_fingerprint(configuration)
    record = {
        "configuration_fingerprint": fingerprint,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
    }
    CONFIG_OUTPUT_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return fingerprint


def load_checkpoint_records(expected_fingerprint: str) -> list[dict[str, object]]:
    """Load checkpointed successes only when they used the same frozen settings."""

    if not CHECKPOINT_PATH.exists():
        return []
    if not CHECKPOINT_CONFIG_PATH.is_file():
        raise RuntimeError("Checkpoint exists without a configuration record; refusing to mix outputs.")
    checkpoint_record = json.loads(CHECKPOINT_CONFIG_PATH.read_text(encoding="utf-8"))
    if checkpoint_record.get("configuration_fingerprint") != expected_fingerprint:
        raise RuntimeError("Checkpoint configuration differs from frozen extraction settings.")
    records = pd.read_csv(CHECKPOINT_PATH).to_dict(orient="records")
    return [record for record in records if record.get("_status") == "success"]


def write_checkpoint(records: list[dict[str, object]], fingerprint: str) -> None:
    """Write restart-safe derived records only inside repository artifacts."""

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(CHECKPOINT_PATH, index=False)
    CHECKPOINT_CONFIG_PATH.write_text(
        json.dumps({"configuration_fingerprint": fingerprint}, indent=2), encoding="utf-8"
    )


def task_wav_paths(participant_directory: Path) -> dict[str, list[Path]]:
    """Locate task-labelled WAV files in one participant directory read-only."""

    paths_by_task = {task: [] for task in TASKS}
    for path in participant_directory.rglob("*"):
        if not path.is_file() or path.suffix.casefold() != ".wav":
            continue
        upper_name = path.name.upper()
        for task in TASKS:
            if task in upper_name:
                paths_by_task[task].append(path)
    return {task: sorted(paths) for task, paths in paths_by_task.items()}


def participant_directory_for_id(directory_index: dict[str, list[Path]], participant_id: object) -> Path | None:
    """Resolve a participant folder using existing tolerant identifier matching."""

    matches = {
        directory
        for key in identifier_keys(participant_id)
        for directory in directory_index.get(key, [])
    }
    return next(iter(matches)) if len(matches) == 1 else None


def expected_recording_keys(feature_table: pd.DataFrame) -> list[tuple[str, str]]:
    """Return the full deterministic participant-task work list without hard-coded counts."""

    if feature_table["participant_id"].duplicated().any():
        raise ValueError("Manual transcript feature table has duplicate participant IDs.")
    return [(str(participant_id), task) for participant_id in feature_table["participant_id"] for task in TASKS]


def record_key(record: dict[str, object]) -> tuple[str, str]:
    return str(record["participant_id"]), str(record["task"])


def recording_row(participant: pd.Series, task: str) -> dict[str, object]:
    """Create one task row with bookkeeping fields and no predictive metadata extras."""

    return {
        "participant_id": participant["participant_id"],
        "diagnosis": participant["diagnosis"],
        "screening_label": participant["screening_label"],
        "Split": participant["Split"],
        "task": task,
    }


def failed_recording_row(participant: pd.Series, task: str) -> dict[str, object]:
    """Record unavailable audio as missing derived features, not fabricated zeroes."""

    row = recording_row(participant, task)
    row.update(empty_acoustic_features())
    row.update(
        {
            "_status": "failed",
            "_sample_rate": float("nan"),
            "_channel_count": float("nan"),
            "_pitch_failed": False,
            "_pause_failed": False,
        }
    )
    return row


def extract_recording_row(participant: pd.Series, task: str, wav_path: Path) -> dict[str, object]:
    """Read one WAV once and compute frozen acoustic and transcript-rate features."""

    row = recording_row(participant, task)
    audio = load_wav_read_only(wav_path)
    result = extract_acoustic_feature_result(audio.waveform, audio.sample_rate)
    features = dict(result.features)
    features.update(
        speech_rate_features(
            participant[f"{task.casefold()}_word_count"],
            features["audio_duration_seconds"],
            features["voiced_duration_seconds"],
        )
    )
    row.update(features)
    row.update(
        {
            "_status": "success",
            "_sample_rate": audio.sample_rate,
            "_channel_count": audio.channel_count,
            "_pitch_failed": not result.pitch_available,
            "_pause_failed": not result.pause_detection_success,
        }
    )
    return row


def pivot_participant_features(
    metadata_table: pd.DataFrame, recording_records: list[dict[str, object]]
) -> pd.DataFrame:
    """Pivot exactly one record per participant-task into one participant row."""

    recordings = pd.DataFrame(recording_records)
    if recordings.duplicated(["participant_id", "task"]).any():
        raise ValueError("Duplicate participant-task records prevent a safe acoustic pivot.")
    expected_keys = set(expected_recording_keys(metadata_table))
    actual_keys = {record_key(record) for record in recording_records}
    if actual_keys != expected_keys:
        raise ValueError("Recording records do not match the expected participant-task set.")

    acoustic_values = recordings.set_index(["participant_id", "task"])[list(ACOUSTIC_FEATURE_NAMES)]
    task_columns = pd.MultiIndex.from_product(
        [acoustic_values.index.get_level_values("participant_id").unique(), TASKS],
        names=["participant_id", "task"],
    )
    acoustic_values = acoustic_values.reindex(task_columns)
    pivoted = acoustic_values.unstack("task")
    pivoted.columns = [f"{task.casefold()}_{feature}" for feature, task in pivoted.columns]
    ordered_acoustic_columns = [f"{task.casefold()}_{feature}" for task in TASKS for feature in ACOUSTIC_FEATURE_NAMES]
    pivoted = pivoted.reindex(columns=ordered_acoustic_columns)

    metadata = metadata_table[list(BOOKKEEPING_COLUMNS)].copy()
    if metadata["participant_id"].duplicated().any():
        raise ValueError("Duplicate participant IDs prevent a safe acoustic pivot.")
    metadata = metadata.set_index("participant_id")
    final_table = metadata.join(pivoted, how="left").reset_index()
    if len(final_table) != len(metadata_table):
        raise ValueError("Acoustic pivot did not preserve one row per participant.")
    return final_table


def constraint_violations(feature_table: pd.DataFrame) -> dict[str, int]:
    """Count sanity-rule violations only where values are present."""

    violations: dict[str, int] = {}
    rules = {
        "silence_ratio_outside_0_1": lambda frame, prefix: (frame[f"{prefix}_silence_ratio"] < 0)
        | (frame[f"{prefix}_silence_ratio"] > 1),
        "voiced_duration_exceeds_audio_duration": lambda frame, prefix: frame[
            f"{prefix}_voiced_duration_seconds"
        ]
        > frame[f"{prefix}_audio_duration_seconds"],
        "negative_silence_duration": lambda frame, prefix: frame[f"{prefix}_silence_duration_seconds"] < 0,
        "negative_pause_count": lambda frame, prefix: frame[f"{prefix}_pause_count"] < 0,
        "negative_pause_total_seconds": lambda frame, prefix: frame[f"{prefix}_pause_total_seconds"] < 0,
        "negative_recording_word_rate": lambda frame, prefix: frame[f"{prefix}_recording_word_rate_wpm"] < 0,
        "negative_articulation_rate": lambda frame, prefix: frame[f"{prefix}_articulation_rate_wpm"] < 0,
    }
    for rule_name, rule in rules.items():
        violations[rule_name] = int(
            sum(
                rule(feature_table, task.casefold()).fillna(False).sum()
                for task in TASKS
            )
        )
    return violations


def print_final_report(
    feature_table: pd.DataFrame,
    recording_records: list[dict[str, object]],
    elapsed_seconds: float,
) -> None:
    """Print non-identifying extraction counts and pipeline-only validation."""

    records = pd.DataFrame(recording_records)
    acoustic_columns = [column for column in feature_table.columns if column not in BOOKKEEPING_COLUMNS]
    numeric_acoustic = feature_table[acoustic_columns].apply(pd.to_numeric, errors="coerce")
    successful = records[records["_status"].eq("success")]
    failed = records[records["_status"].eq("failed")]
    split = feature_table["Split"].astype("string").str.strip().str.upper()
    labels = feature_table["screening_label"].astype("string")
    sample_rates = Counter(successful["_sample_rate"].dropna().astype(int))
    channels = Counter(successful["_channel_count"].dropna().astype(int))
    missing_counts = numeric_acoustic.isna().sum()
    infinite_counts = pd.Series(
        {column: int(np.isinf(numeric_acoustic[column].dropna()).sum()) for column in acoustic_columns}
    )
    violations = constraint_violations(feature_table)
    if any(violations.values()):
        raise ValueError(f"Acoustic sanity constraints failed: {violations}")

    print("\nFull Acoustic Feature Extraction")
    print("================================")
    print(f"Participants: {len(feature_table)}")
    print(f"WAV files: {len(records)}")
    print(f"Successful files: {len(successful)}")
    print(f"Failed files: {len(failed)}")
    print(f"TRAIN participants: {int(split.eq('TRAIN').sum())}")
    print(f"TEST participants: {int(split.eq('TEST').sum())}")
    print(f"Healthy participants: {int(labels.eq('Healthy').sum())}")
    print(f"Impaired participants: {int(labels.eq('Impaired').sum())}")
    print(f"Acoustic feature columns: {len(acoustic_columns)}")
    print(f"Sample rates: {dict(sorted(sample_rates.items()))}")
    print(f"Channels: {dict(sorted(channels.items()))}")
    print(f"Pitch failures: {int(successful['_pitch_failed'].sum())}")
    print(f"Pause failures: {int(successful['_pause_failed'].sum())}")
    print("\nMissing features:")
    print(missing_counts.to_string())
    print("\nInfinite features:")
    print(infinite_counts.to_string())
    for task in TASKS:
        duration_column = f"{task.casefold()}_audio_duration_seconds"
        print(f"\n{task} duration statistics (TRAIN only):")
        print(feature_table.loc[split.eq("TRAIN"), duration_column].describe().to_string())
    print(f"\nTotal runtime: {elapsed_seconds:.1f} seconds")
    print(f"Average runtime/WAV: {elapsed_seconds / len(records):.2f} seconds")
    print(f"Saved: {OUTPUT_PATH}")


def main() -> int:
    if not MANUAL_FEATURE_TABLE_PATH.is_file():
        print(f"Manual transcript feature table not found: {MANUAL_FEATURE_TABLE_PATH}")
        return 1
    started_at = perf_counter()
    configuration = frozen_configuration_payload()
    fingerprint = write_configuration_record(configuration)
    metadata_table = pd.read_csv(MANUAL_FEATURE_TABLE_PATH)
    if not set(BOOKKEEPING_COLUMNS).issubset(metadata_table.columns):
        print("Manual feature table is missing required participant bookkeeping columns.")
        return 1
    expected_keys = expected_recording_keys(metadata_table)
    completed_records = load_checkpoint_records(fingerprint)
    completed_by_key = {record_key(record): record for record in completed_records}
    if len(completed_by_key) != len(completed_records):
        raise RuntimeError("Checkpoint has duplicate participant-task records.")
    unknown_checkpoint_keys = set(completed_by_key).difference(expected_keys)
    if unknown_checkpoint_keys:
        raise RuntimeError("Checkpoint does not match the current participant-task work list.")

    dataset_root = get_dataset_root()
    directory_index = build_directory_index(non_hidden_directories(dataset_root))
    total_files = len(expected_keys)
    print("Frozen full acoustic extraction started", flush=True)
    print("Official TEST labels used for extraction decisions: 0", flush=True)
    print(f"Reusing checkpointed successful recordings: {len(completed_records)}", flush=True)

    records = list(completed_records)
    new_records_since_checkpoint = 0
    failures = 0
    for participant in metadata_table.itertuples(index=False):
        participant_series = pd.Series(participant._asdict())
        participant_directory = participant_directory_for_id(directory_index, participant_series["participant_id"])
        paths_by_task = task_wav_paths(participant_directory) if participant_directory else {task: [] for task in TASKS}
        for task in TASKS:
            key = (str(participant_series["participant_id"]), task)
            if key in completed_by_key:
                continue
            completed_count = len(records)
            progress_number = completed_count + 1
            if progress_number == 1 or progress_number % CHECKPOINT_INTERVAL == 0 or progress_number == total_files:
                elapsed = perf_counter() - started_at
                remaining = total_files - completed_count
                average = elapsed / max(1, completed_count - len(completed_records))
                print(
                    f"[{progress_number}/{total_files}] participant {participant_series['participant_id']} {task} "
                    f"| elapsed {elapsed:.1f}s | estimated remaining {average * remaining / 60:.1f}m",
                    flush=True,
                )
            try:
                paths = paths_by_task[task]
                if len(paths) != 1:
                    raise AudioValidationError("Expected exactly one task-labelled WAV file.")
                record = extract_recording_row(participant_series, task, paths[0])
            except (AudioValidationError, OSError, RuntimeError, ValueError) as error:
                failures += 1
                record = failed_recording_row(participant_series, task)
                print(
                    f"FAILED participant {participant_series['participant_id']} {task}: {error.__class__.__name__}",
                    flush=True,
                )
            records.append(record)
            new_records_since_checkpoint += 1
            if new_records_since_checkpoint >= CHECKPOINT_INTERVAL:
                write_checkpoint(records, fingerprint)
                new_records_since_checkpoint = 0

    write_checkpoint(records, fingerprint)
    final_table = pivot_participant_features(metadata_table, records)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    final_table.to_csv(OUTPUT_PATH, index=False)
    print_final_report(final_table, records, perf_counter() - started_at)
    print("No model training was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
