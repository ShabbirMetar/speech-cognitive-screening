"""Inspect PROCESS-2 transcript structure without printing participant speech."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import statistics
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
for import_root in (PROJECT_ROOT, BACKEND_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

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
from ml.nlp.linguistic_features import tokenize_text
from ml.nlp.transcript_parser import parse_transcript_file, task_name_from_path


def token_statistics(values: list[int]) -> dict[str, float | int]:
    """Return compact token-count statistics for one task."""

    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0, "median": 0.0, "max": 0}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def main() -> int:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--all-splits",
        action="store_true",
        help="Inspect all participants; TRAIN is the safe development default.",
    )
    arguments = argument_parser.parse_args()

    try:
        dataset_root = get_dataset_root()
        metadata = load_metadata(dataset_root)
    except (DatasetConfigurationError, DatasetValidationError, RuntimeError) as error:
        print(f"Transcript inspection could not continue: {error}")
        return 1

    columns = identify_metadata_columns(metadata)
    if columns.participant_id is None or columns.split is None:
        print("Transcript inspection requires identifiable participant ID and split columns.")
        return 1

    scope = "ALL participants" if arguments.all_splits else "TRAIN participants only"
    selected_metadata = metadata
    if not arguments.all_splits:
        selected_metadata = metadata[
            metadata[columns.split].astype("string").str.strip().str.upper().eq("TRAIN")
        ]

    directory_index = build_directory_index(participant_directories(dataset_root))
    counts = Counter()
    task_token_counts: dict[str, list[int]] = defaultdict(list)
    sample: dict[str, object] | None = None
    for _, metadata_row in selected_metadata.iterrows():
        matching_directories = {
            directory
            for key in identifier_keys(metadata_row[columns.participant_id])
            for directory in directory_index.get(key, [])
        }
        if len(matching_directories) != 1:
            counts["unmatched_participants"] += 1
            continue

        for transcript_path in next(iter(matching_directories)).rglob("*.txt"):
            task_name = task_name_from_path(transcript_path)
            if task_name is None:
                continue
            try:
                parsed = parse_transcript_file(transcript_path)
            except (OSError, UnicodeError):
                counts["read_errors"] += 1
                continue

            counts["transcript_files"] += 1
            counts[f"task_{task_name}"] += 1
            counts["empty_files"] += int(not parsed.full_text.strip())
            counts["pat_files"] += int(
                any(label.casefold() == "pat" for label in parsed.speaker_labels_detected)
            )
            counts["oth_files"] += int(
                any(label.casefold() == "oth" for label in parsed.speaker_labels_detected)
            )
            counts["pause_files"] += int(parsed.pause_count > 0)
            counts["multiple_speaker_files"] += int(len(parsed.speaker_labels_detected) > 1)
            counts[f"attribution_{parsed.speaker_attribution}"] += 1
            task_token_counts[task_name].append(len(tokenize_text(parsed.participant_text)))
            if sample is None:
                sample = {
                    "task": task_name,
                    "speaker_labels_detected": ", ".join(parsed.speaker_labels_detected) or "None",
                    "pause_annotations_detected": "YES" if parsed.pause_count else "NO",
                    "number_of_lines": parsed.line_count,
                    "speaker_attribution": parsed.speaker_attribution,
                }

    print("PROCESS-2 Transcript Structure Inspection")
    print("=========================================")
    print(f"Scope: {scope}")
    print(f"Transcript files: {counts['transcript_files']}")
    print("By task: " + ", ".join(f"{task}={counts[f'task_{task}']}" for task in ("SFT", "PFT", "CTD")))
    print(f"Empty transcript files: {counts['empty_files']}")
    print(f"Files containing Pat: {counts['pat_files']}")
    print(f"Files containing Oth: {counts['oth_files']}")
    print(f"Files containing pause annotations: {counts['pause_files']}")
    print(f"Files containing multiple speakers: {counts['multiple_speaker_files']}")
    print(
        "Speaker attribution states: "
        f"explicit_pat={counts['attribution_explicit_pat']}, "
        f"unlabeled_fallback={counts['attribution_unlabeled_fallback']}, "
        f"no_participant_label={counts['attribution_no_participant_label']}"
    )
    print(f"Encoding/read errors: {counts['read_errors']}")
    print(f"Participants without a unique directory: {counts['unmatched_participants']}")
    print("\nApproximate participant-token statistics by task")
    for task in ("SFT", "PFT", "CTD"):
        print(f"{task}: {token_statistics(task_token_counts[task])}")
    print("\nSanitized structural sample (no speech content)")
    for key, value in (sample or {}).items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
