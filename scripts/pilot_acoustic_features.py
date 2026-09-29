"""Run a small, TRAIN-only, read-only acoustic feature pilot for PROCESS-2."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
for import_root in (PROJECT_ROOT, BACKEND_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from ml.audio.acoustic_features import (
    ACOUSTIC_FEATURE_NAMES,
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
from ml.models.baseline_logistic import training_rows


FEATURE_TABLE_PATH = PROJECT_ROOT / "artifacts" / "features" / "manual_transcript_features.csv"
FEATURE_OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "features" / "acoustic_pilot_features.csv"
PLOT_OUTPUT_DIRECTORY = PROJECT_ROOT / "artifacts" / "results" / "acoustic_pilot"
PILOT_PER_LABEL = 5
PILOT_RANDOM_STATE = 42
PILOT_TASKS = tuple(EXPECTED_TASKS)


def select_pilot_participants(feature_table: pd.DataFrame) -> pd.DataFrame:
    """Select five Healthy and five Impaired TRAIN participants reproducibly."""

    train_table = training_rows(feature_table)
    if not train_table["Split"].astype("string").str.strip().str.upper().eq("TRAIN").all():
        raise AssertionError("Pilot selection may use only official TRAIN participants.")
    selected_groups: list[pd.DataFrame] = []
    for label in ("Healthy", "Impaired"):
        group = train_table[train_table["screening_label"].eq(label)]
        if len(group) < PILOT_PER_LABEL:
            raise ValueError(f"Not enough TRAIN participants with screening_label={label!r}.")
        selected_groups.append(group.sample(n=PILOT_PER_LABEL, random_state=PILOT_RANDOM_STATE))
    return pd.concat(selected_groups, ignore_index=True).sort_values("participant_id").reset_index(drop=True)


def task_wav_paths(participant_directory: Path) -> dict[str, list[Path]]:
    """Find task-labelled WAVs in one selected participant directory only."""

    paths_by_task = {task: [] for task in PILOT_TASKS}
    for path in participant_directory.rglob("*"):
        if not path.is_file() or path.suffix.casefold() != ".wav":
            continue
        upper_name = path.name.upper()
        for task in PILOT_TASKS:
            if task in upper_name:
                paths_by_task[task].append(path)
    return {task: sorted(paths) for task, paths in paths_by_task.items()}


def participant_directory_for_id(directory_index: dict[str, list[Path]], participant_id: object) -> Path | None:
    """Resolve one participant directory using existing tolerant ID matching."""

    matches = {
        directory
        for key in identifier_keys(participant_id)
        for directory in directory_index.get(key, [])
    }
    return next(iter(matches)) if len(matches) == 1 else None


def _feature_row(participant: pd.Series, task: str) -> dict[str, object]:
    return {
        "participant_id": participant["participant_id"],
        "task": task,
        "diagnosis": participant["diagnosis"],
        "screening_label": participant["screening_label"],
        "Split": participant["Split"],
    }


def _plot_box_by_task(data: pd.DataFrame, column: str, title: str, ylabel: str, output_path: Path) -> None:
    values = [data.loc[data["task"].eq(task), column].dropna().to_numpy() for task in PILOT_TASKS]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.boxplot([value if len(value) else [np.nan] for value in values], tick_labels=PILOT_TASKS)
    axis.set_title(title)
    axis.set_xlabel("Task")
    axis.set_ylabel(ylabel)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_diagnostic_plots(feature_table: pd.DataFrame) -> None:
    """Save pipeline-sanity plots only; they make no disease-performance claims."""

    PLOT_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _plot_box_by_task(
        feature_table,
        "audio_duration_seconds",
        "Pilot recording duration by task",
        "Duration (seconds)",
        PLOT_OUTPUT_DIRECTORY / "duration_by_task.png",
    )
    _plot_box_by_task(
        feature_table,
        "silence_ratio",
        "Pilot energy-based silence ratio by task",
        "Silence ratio",
        PLOT_OUTPUT_DIRECTORY / "silence_ratio_by_task.png",
    )
    _plot_box_by_task(
        feature_table,
        "pause_count",
        "Pilot internal pause count by task",
        "Internal pause count",
        PLOT_OUTPUT_DIRECTORY / "pause_count_by_task.png",
    )
    for column, title, xlabel, filename in (
        ("f0_mean_hz", "Pilot voiced F0 estimates", "Mean F0 (Hz)", "f0_distribution.png"),
        ("rms_mean", "Pilot RMS energy estimates", "Mean RMS", "rms_distribution.png"),
    ):
        values = feature_table[column].dropna().to_numpy()
        figure, axis = plt.subplots(figsize=(7, 4))
        if len(values):
            axis.hist(values, bins=min(10, len(values)), color="#4C72B0", edgecolor="white")
        else:
            axis.text(0.5, 0.5, "No valid values", ha="center", va="center", transform=axis.transAxes)
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Recording count")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        figure.tight_layout()
        figure.savefig(PLOT_OUTPUT_DIRECTORY / filename, dpi=150, bbox_inches="tight")
        plt.close(figure)


def print_pilot_summary(
    feature_table: pd.DataFrame,
    failures: int,
    sample_rates: Counter[int],
    pitch_failures: int,
    pause_failures: int,
    elapsed_seconds: float,
) -> None:
    """Print aggregate, non-identifying pilot quality checks."""

    numeric_features = feature_table[list(ACOUSTIC_FEATURE_NAMES)].apply(pd.to_numeric, errors="coerce")
    infinite_counts = pd.Series(
        {
            column: int(np.isinf(numeric_features[column].dropna()).sum())
            for column in numeric_features.columns
        }
    )
    print("\nAcoustic Pilot Summary")
    print("======================")
    print(f"Participants processed: {feature_table['participant_id'].nunique()}")
    print(f"WAV files processed: {len(feature_table) - failures}")
    print(f"Failed WAV files: {failures}")
    print(f"Sample-rate distribution: {dict(sorted(sample_rates.items()))}")
    print("\nDuration statistics by task")
    print(feature_table.groupby("task")["audio_duration_seconds"].describe().to_string())
    print("\nMissing values by feature")
    print(numeric_features.isna().sum().to_string())
    print("\nInfinite values by feature")
    print(infinite_counts.to_string())
    print(f"Pitch extraction failures: {pitch_failures}")
    print(f"Pause extraction failures: {pause_failures}")
    print(f"Total runtime: {elapsed_seconds:.1f} seconds")
    print(f"Average runtime per WAV: {elapsed_seconds / len(feature_table):.2f} seconds")


def main() -> int:
    if not FEATURE_TABLE_PATH.is_file():
        print(f"Manual transcript feature table not found: {FEATURE_TABLE_PATH}")
        return 1
    started_at = perf_counter()
    feature_table = pd.read_csv(FEATURE_TABLE_PATH)
    pilot_participants = select_pilot_participants(feature_table)
    dataset_root = get_dataset_root()
    directory_index = build_directory_index(non_hidden_directories(dataset_root))
    total_recordings = len(pilot_participants) * len(PILOT_TASKS)
    records: list[dict[str, object]] = []
    failures = 0
    pitch_failures = 0
    pause_failures = 0
    sample_rates: Counter[int] = Counter()

    for participant in pilot_participants.itertuples(index=False):
        participant_series = pd.Series(participant._asdict())
        participant_directory = participant_directory_for_id(directory_index, participant_series["participant_id"])
        paths_by_task = task_wav_paths(participant_directory) if participant_directory else {task: [] for task in PILOT_TASKS}
        for task in PILOT_TASKS:
            progress_number = len(records) + 1
            print(f"[{progress_number}/{total_recordings}] participant {participant_series['participant_id']} {task}", flush=True)
            row = _feature_row(participant_series, task)
            paths = paths_by_task[task]
            if len(paths) != 1:
                failures += 1
                row.update(empty_acoustic_features())
                records.append(row)
                print("feature extraction: FAILED (expected one task WAV)", flush=True)
                continue
            try:
                audio = load_wav_read_only(paths[0])
                result = extract_acoustic_feature_result(audio.waveform, audio.sample_rate)
                word_count = participant_series[f"{task.casefold()}_word_count"]
                features = dict(result.features)
                features.update(
                    speech_rate_features(
                        word_count,
                        features["audio_duration_seconds"],
                        features["voiced_duration_seconds"],
                    )
                )
                row.update(features)
                records.append(row)
                sample_rates[audio.sample_rate] += 1
                pitch_failures += int(not result.pitch_available)
                pause_failures += int(not result.pause_detection_success)
                print(f"duration: {features['audio_duration_seconds']:.2f} seconds", flush=True)
                print(f"sample rate: {audio.sample_rate} Hz; source channels: {audio.channel_count}", flush=True)
                print("feature extraction: OK", flush=True)
            except (AudioValidationError, OSError, RuntimeError, ValueError) as error:
                failures += 1
                row.update(empty_acoustic_features())
                records.append(row)
                print(f"feature extraction: FAILED ({error.__class__.__name__})", flush=True)

    pilot_features = pd.DataFrame(records)
    FEATURE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pilot_features.to_csv(FEATURE_OUTPUT_PATH, index=False)
    save_diagnostic_plots(pilot_features)
    print_pilot_summary(
        pilot_features,
        failures,
        sample_rates,
        pitch_failures,
        pause_failures,
        perf_counter() - started_at,
    )
    print(f"\nSaved pilot features: {FEATURE_OUTPUT_PATH}")
    print(f"Saved diagnostic plots: {PLOT_OUTPUT_DIRECTORY}")
    print("This is a small engineering pilot; it does not evaluate TEST or train a model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
