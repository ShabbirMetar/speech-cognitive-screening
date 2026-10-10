"""Resume-safe local base.en transcription for the official TRAIN recordings only."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
for import_root in (PROJECT_ROOT, BACKEND_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


from ml.asr.asr_linguistic_features import (
    asr_feature_columns,
    build_asr_feature_tables,
    feature_quality_report,
)
from ml.asr.whisper_transcriber import (
    WhisperTranscriptionConfig,
    load_whisper_model,
    transcribe_audio,
)
from ml.data.development_guard import assert_train_only
from ml.data.process2_loader import (
    EXPECTED_TASKS,
    build_directory_index,
    get_dataset_root,
    identifier_keys,
    non_hidden_directories,
)


LINGUISTIC_TRAIN_PATH = PROJECT_ROOT / "artifacts" / "features" / "linguistic_features_train.csv"
ACOUSTIC_TRAIN_PATH = PROJECT_ROOT / "artifacts" / "features" / "acoustic_features_train.csv"
TRANSCRIPT_DIRECTORY = PROJECT_ROOT / "artifacts" / "transcripts" / "asr_train"
CONFIG_PATH = TRANSCRIPT_DIRECTORY / "asr_config.json"
CHECKPOINT_PATH = TRANSCRIPT_DIRECTORY / "base_en_train_checkpoint.csv"
FINAL_TRANSCRIPT_PATH = TRANSCRIPT_DIRECTORY / "base_en_train_transcripts.csv"
ASR_FEATURE_PATH = PROJECT_ROOT / "artifacts" / "features" / "asr_linguistic_features_train.csv"
DEPLOYMENT_FEATURE_PATH = PROJECT_ROOT / "artifacts" / "features" / "deployment_features_train.csv"
QUALITY_REPORT_PATH = PROJECT_ROOT / "artifacts" / "results" / "asr_train_feature_quality.csv"
TASKS = tuple(EXPECTED_TASKS)
EXPECTED_TRAIN_PARTICIPANTS = 320
TRANSCRIPT_COLUMNS = (
    "participant_id",
    "task",
    "transcript_text",
    "audio_duration_seconds",
    "transcription_runtime_seconds",
    "model",
)
FROZEN_CONFIG = WhisperTranscriptionConfig(
    model_size="base.en", device="cpu", compute_type="int8", language="en"
)


def _atomic_csv_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def _atomic_json_write(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary_path.replace(path)


def format_timing_messages(
    wall_clock_seconds: float, inference_seconds: float, completed_count: int, failures: int
) -> tuple[str, str, str]:
    """Label wall-clock loop duration separately from summed ASR inference time."""

    if completed_count <= 0:
        raise ValueError("completed_count must be positive when formatting ASR timing.")
    return (
        f"Wall-clock transcription loop runtime: {wall_clock_seconds:.1f} seconds; "
        f"failures requiring retry: {failures}",
        f"Summed ASR inference runtime: {inference_seconds:.1f} seconds",
        f"Average ASR inference runtime/WAV: {inference_seconds / completed_count:.2f} seconds",
    )


def frozen_configuration_payload(config: WhisperTranscriptionConfig) -> dict[str, object]:
    """Produce the versioned configuration fingerprint required for checkpoint reuse."""

    return {"extractor": "faster-whisper", "pipeline_version": 1, **asdict(config)}


def write_or_validate_frozen_config(config_path: Path, config: WhisperTranscriptionConfig) -> None:
    """Write the config once or reject a resume attempt with different settings."""

    expected = frozen_configuration_payload(config)
    if config_path.exists():
        found = json.loads(config_path.read_text(encoding="utf-8"))
        if found != expected:
            raise RuntimeError("Existing ASR checkpoint configuration does not match frozen base.en settings.")
        return
    _atomic_json_write(expected, config_path)


def build_split_worklist(feature_table: pd.DataFrame, expected_split: str) -> pd.DataFrame:
    """Create exactly one required base.en item per participant in one strict split."""

    if {"participant_id", "Split"}.difference(feature_table.columns):
        raise ValueError("Feature table is missing participant_id or Split for ASR worklist creation.")
    normalized_expected = expected_split.strip().upper()
    observed = feature_table["Split"].astype("string").str.strip().str.upper()
    if not observed.eq(normalized_expected).all():
        found = sorted(observed.dropna().unique().tolist())
        raise ValueError(f"ASR worklist requires only Split == {normalized_expected}; found {found}.")
    if feature_table["participant_id"].duplicated().any():
        raise ValueError("ASR feature table contains duplicate participant IDs.")
    return pd.DataFrame(
        [
            {"participant_id": participant_id, "task": task}
            for participant_id in feature_table["participant_id"]
            for task in TASKS
        ]
    )


def build_train_worklist(train_table: pd.DataFrame) -> pd.DataFrame:
    """Create exactly one required base.en transcription item per TRAIN participant/task."""

    train = assert_train_only(train_table)
    return build_split_worklist(train, "TRAIN")


def validate_transcript_frame(frame: pd.DataFrame, worklist: pd.DataFrame) -> None:
    """Validate checkpoint/final rows as a subset of one frozen split worklist."""

    missing = set(TRANSCRIPT_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"ASR transcript rows are missing required columns: {sorted(missing)}")
    if frame.duplicated(["participant_id", "task"]).any():
        raise ValueError("ASR transcript rows contain duplicate participant/task pairs.")
    if not frame["model"].eq(FROZEN_CONFIG.model_size).all():
        raise ValueError("ASR transcript checkpoint contains a non-base.en model.")
    expected = set(map(tuple, worklist[["participant_id", "task"]].to_records(index=False)))
    actual = set(map(tuple, frame[["participant_id", "task"]].to_records(index=False)))
    if not actual.issubset(expected):
        raise ValueError("ASR transcript checkpoint contains unknown participant/task rows.")


def load_checkpoint(checkpoint_path: Path, worklist: pd.DataFrame) -> pd.DataFrame:
    """Load completed successful rows only; missing rows will be transcribed on resume."""

    if not checkpoint_path.exists():
        return pd.DataFrame(columns=TRANSCRIPT_COLUMNS)
    completed = pd.read_csv(checkpoint_path)
    validate_transcript_frame(completed, worklist)
    return completed.loc[:, TRANSCRIPT_COLUMNS].copy()


def remaining_worklist(worklist: pd.DataFrame, completed: pd.DataFrame) -> pd.DataFrame:
    """Return work not present in the completed checkpoint without retranscribing rows."""

    completed_keys = set(map(tuple, completed[["participant_id", "task"]].to_records(index=False)))
    return worklist.loc[
        ~worklist.apply(lambda row: (row["participant_id"], row["task"]) in completed_keys, axis=1)
    ].reset_index(drop=True)


def participant_directory_for_id(directory_index: dict[str, list[Path]], participant_id: object) -> Path | None:
    matches = {
        directory
        for key in identifier_keys(participant_id)
        for directory in directory_index.get(key, [])
    }
    return next(iter(matches)) if len(matches) == 1 else None


def task_wav_paths(participant_directory: Path) -> dict[str, list[Path]]:
    paths = {task: [] for task in TASKS}
    for path in participant_directory.rglob("*.wav"):
        for task in TASKS:
            if task in path.name.upper():
                paths[task].append(path)
    return {task: sorted(task_paths) for task, task_paths in paths.items()}


def _feature_report(asr_linguistic: pd.DataFrame, per_recording: pd.DataFrame) -> pd.DataFrame:
    quality = feature_quality_report(asr_linguistic, asr_feature_columns(asr_linguistic))
    word_summary = (
        per_recording.groupby("task")["word_count"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
    )
    print("\nASR word-count statistics by task")
    print(word_summary.to_string(index=False))
    print("\nASR-sensitive features retained for later TRAIN-only evaluation: filler_count, repeated_word_count, word_count.")
    print("Missing/infinite/constant ASR linguistic features:")
    print(quality.to_string(index=False))
    return quality


def main() -> int:
    """Transcribe all and only TRAIN recordings, then build deployment-aligned features."""

    for path in (LINGUISTIC_TRAIN_PATH, ACOUSTIC_TRAIN_PATH):
        if not path.is_file():
            print(f"Required TRAIN feature table not found: {path}")
            return 1
    train_table = assert_train_only(pd.read_csv(LINGUISTIC_TRAIN_PATH))
    acoustic_train = assert_train_only(pd.read_csv(ACOUSTIC_TRAIN_PATH))
    if len(train_table) != EXPECTED_TRAIN_PARTICIPANTS:
        raise AssertionError(f"Expected {EXPECTED_TRAIN_PARTICIPANTS} TRAIN participants; found {len(train_table)}.")
    worklist = build_train_worklist(train_table)
    if len(worklist) != EXPECTED_TRAIN_PARTICIPANTS * len(TASKS):
        raise AssertionError("TRAIN transcription worklist does not contain exactly three tasks per participant.")

    write_or_validate_frozen_config(CONFIG_PATH, FROZEN_CONFIG)
    completed = load_checkpoint(CHECKPOINT_PATH, worklist)
    remaining = remaining_worklist(worklist, completed)
    print("Frozen base.en TRAIN Transcription")
    print("==================================")
    print(f"TRAIN participants: {len(train_table)}")
    print("TEST participants processed: 0")
    print(f"Expected WAV files: {len(worklist)}")
    print(f"Previously completed recordings: {len(completed)}")
    print(f"Remaining recordings: {len(remaining)}")

    if len(remaining):
        model = load_whisper_model(FROZEN_CONFIG)
        dataset_root = get_dataset_root()
        directory_index = build_directory_index(non_hidden_directories(dataset_root))
        started_at = perf_counter()
        failures = 0
        for progress, item in enumerate(remaining.itertuples(index=False), start=1):
            participant_directory = participant_directory_for_id(directory_index, item.participant_id)
            wavs = task_wav_paths(participant_directory)[item.task] if participant_directory else []
            if len(wavs) != 1:
                failures += 1
                print(f"[{progress}/{len(remaining)}] {item.task}: failed (expected one WAV)", flush=True)
                continue
            try:
                transcription = transcribe_audio(model, wavs[0], FROZEN_CONFIG)
                row = pd.DataFrame(
                    [
                        {
                            "participant_id": item.participant_id,
                            "task": item.task,
                            "transcript_text": transcription.transcript_text,
                            "audio_duration_seconds": transcription.audio_duration_seconds,
                            "transcription_runtime_seconds": transcription.transcription_runtime_seconds,
                            "model": FROZEN_CONFIG.model_size,
                        }
                    ]
                )
                completed = pd.concat([completed, row], ignore_index=True)
                validate_transcript_frame(completed, worklist)
                _atomic_csv_write(completed.loc[:, TRANSCRIPT_COLUMNS], CHECKPOINT_PATH)
                print(f"[{progress}/{len(remaining)}] {item.task}: complete", flush=True)
            except (OSError, RuntimeError, ValueError) as error:
                failures += 1
                print(f"[{progress}/{len(remaining)}] {item.task}: failed ({error.__class__.__name__})", flush=True)
        elapsed = perf_counter() - started_at
        loop_message, _, _ = format_timing_messages(elapsed, 0.0, 1, failures)
        print(loop_message)

    validate_transcript_frame(completed, worklist)
    if len(completed) != len(worklist):
        print("ASR TRAIN transcription is incomplete; retained checkpoint can be resumed safely.")
        return 1

    completed = completed.sort_values(["participant_id", "task"]).reset_index(drop=True)
    _atomic_csv_write(completed.loc[:, TRANSCRIPT_COLUMNS], FINAL_TRANSCRIPT_PATH)
    asr_linguistic, deployment, per_recording = build_asr_feature_tables(
        completed, train_table, acoustic_train
    )
    _atomic_csv_write(asr_linguistic, ASR_FEATURE_PATH)
    _atomic_csv_write(deployment, DEPLOYMENT_FEATURE_PATH)
    quality = _feature_report(asr_linguistic, per_recording)
    _atomic_csv_write(quality, QUALITY_REPORT_PATH)
    total_runtime = float(completed["transcription_runtime_seconds"].sum())
    print("\nFull TRAIN ASR Summary")
    print("========================")
    print(f"Transcriptions successful: {len(completed)}")
    print("Transcriptions failed: 0")
    _, inference_message, average_message = format_timing_messages(
        0.0, total_runtime, len(completed), 0
    )
    print(inference_message)
    print(average_message)
    print(f"Saved transcripts: {FINAL_TRANSCRIPT_PATH}")
    print(f"Saved ASR linguistic features: {ASR_FEATURE_PATH}")
    print(f"Saved deployment features: {DEPLOYMENT_FEATURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
