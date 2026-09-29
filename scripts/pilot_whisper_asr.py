"""Run a small local-only, TRAIN-only faster-whisper comparison pilot."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import sys
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
for import_root in (PROJECT_ROOT, BACKEND_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


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
from ml.nlp.linguistic_features import extract_linguistic_features, tokenize_text
from ml.nlp.transcript_parser import (
    parse_transcript_file,
    parse_transcript_text,
    remove_pause_annotations,
    transcript_paths_by_task,
)


FEATURE_TABLE_PATH = PROJECT_ROOT / "artifacts" / "features" / "linguistic_features_train.csv"
TRANSCRIPT_OUTPUT_DIRECTORY = PROJECT_ROOT / "artifacts" / "transcripts" / "asr_pilot"
RESULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "artifacts" / "results" / "asr_pilot"
PILOT_PER_LABEL = 5
PILOT_RANDOM_STATE = 42
PILOT_TASKS = tuple(EXPECTED_TASKS)
PILOT_MODELS = ("base.en", "small.en")
STABILITY_FEATURES = (
    "word_count",
    "unique_word_count",
    "type_token_ratio",
    "average_word_length",
    "repeated_word_count",
    "repetition_ratio",
    "filler_count",
    "filler_ratio",
    "brunet_index",
    "honore_statistic",
)
TRANSCRIPT_OUTPUT_COLUMNS = (
    "participant_id",
    "task",
    "model",
    "transcript_text",
    "detected_language",
    "audio_duration_seconds",
    "transcription_runtime_seconds",
    "success",
)


def select_pilot_participants(feature_table: pd.DataFrame) -> pd.DataFrame:
    """Select a reproducible balanced ten-participant subset from TRAIN only."""

    train_table = assert_train_only(feature_table)
    selected_groups: list[pd.DataFrame] = []
    for label in ("Healthy", "Impaired"):
        group = train_table[train_table["screening_label"].eq(label)]
        if len(group) < PILOT_PER_LABEL:
            raise ValueError(f"Not enough TRAIN participants with screening_label={label!r}.")
        selected_groups.append(group.sample(n=PILOT_PER_LABEL, random_state=PILOT_RANDOM_STATE))
    return pd.concat(selected_groups, ignore_index=True).sort_values("participant_id").reset_index(drop=True)


def participant_directory_for_id(
    directory_index: dict[str, list[Path]], participant_id: object
) -> Path | None:
    """Resolve one directory with the existing tolerant participant-ID matching."""

    matches = {
        directory
        for key in identifier_keys(participant_id)
        for directory in directory_index.get(key, [])
    }
    return next(iter(matches)) if len(matches) == 1 else None


def task_wav_paths(participant_directory: Path) -> dict[str, list[Path]]:
    """Return task-labelled WAV paths without reading or modifying their content."""

    paths = {task: [] for task in PILOT_TASKS}
    for path in participant_directory.rglob("*.wav"):
        for task in PILOT_TASKS:
            if task in path.name.upper():
                paths[task].append(path)
    return {task: sorted(task_paths) for task, task_paths in paths.items()}


def normalise_for_wer(text: str) -> str:
    """Normalize text consistently for reference/hypothesis WER calculation."""

    without_pauses = remove_pause_annotations(text)
    without_speaker_labels = re.sub(
        r"(?im)^\s*(?:pat|oth)\s*:\s*", "", without_pauses
    )
    return " ".join(tokenize_text(without_speaker_labels))


def word_error_rate(reference_text: str, hypothesis_text: str) -> float:
    """Compute token-level WER with a local Levenshtein dynamic-programming implementation."""

    reference = normalise_for_wer(reference_text).split()
    hypothesis = normalise_for_wer(hypothesis_text).split()
    if not reference:
        return 0.0 if not hypothesis else float("nan")
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_token in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_token in enumerate(hypothesis, start=1):
            substitution = previous[hypothesis_index - 1] + (reference_token != hypothesis_token)
            insertion = current[hypothesis_index - 1] + 1
            deletion = previous[hypothesis_index] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return float(previous[-1] / len(reference))


def feature_stability_rows(
    model: str,
    asr_text: str,
    manual_text: str,
) -> list[dict[str, object]]:
    """Return non-annotation linguistic absolute differences for one text pair."""

    asr_features = extract_linguistic_features(parse_transcript_text(asr_text))
    manual_features = extract_linguistic_features(parse_transcript_text(manual_text))
    rows: list[dict[str, object]] = []
    for feature in STABILITY_FEATURES:
        asr_value = float(asr_features[feature])
        manual_value = float(manual_features[feature])
        absolute_difference = (
            abs(asr_value - manual_value)
            if np.isfinite(asr_value) and np.isfinite(manual_value)
            else float("nan")
        )
        rows.append(
            {
                "model": model,
                "feature": feature,
                "asr_value": asr_value,
                "manual_value": manual_value,
                "absolute_difference": absolute_difference,
            }
        )
    return rows


def summarise_feature_stability(rows: Iterable[dict[str, object]]) -> pd.DataFrame:
    """Summarise ASR/manual feature absolute differences without outcome labels."""

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["model", "feature", "comparison_count", "mean_absolute_difference"])
    return (
        frame.groupby(["model", "feature"], as_index=False)["absolute_difference"]
        .agg(comparison_count="count", mean_absolute_difference="mean")
        .sort_values(["model", "feature"])
        .reset_index(drop=True)
    )


def empty_transcript_row(participant_id: object, task: str, model: str) -> dict[str, object]:
    """Provide an explicit failed-transcription row without recording raw error text."""

    return {
        "participant_id": participant_id,
        "task": task,
        "model": model,
        "transcript_text": "",
        "detected_language": None,
        "audio_duration_seconds": float("nan"),
        "transcription_runtime_seconds": float("nan"),
        "success": False,
    }


def output_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Return the fixed, diagnosis-free ASR transcript artifact schema."""

    return pd.DataFrame(rows, columns=TRANSCRIPT_OUTPUT_COLUMNS)


def _manual_text_for_task(participant_directory: Path, task: str) -> str | None:
    paths = transcript_paths_by_task(participant_directory)[task]
    if len(paths) != 1:
        return None
    parsed = parse_transcript_file(paths[0])
    if parsed.speaker_attribution == "no_participant_label":
        return None
    return parsed.participant_text


def _wer_summary(records: list[dict[str, object]], model: str) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    rows: list[dict[str, object]] = []
    for task in (*PILOT_TASKS, "OVERALL"):
        subset = frame if task == "OVERALL" else frame[frame["task"].eq(task)]
        valid = pd.to_numeric(subset["wer"], errors="coerce").dropna()
        rows.append(
            {
                "model": model,
                "task": task,
                "reference_count": int(len(valid)),
                "wer": float(valid.mean()) if len(valid) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _model_summary(records: list[dict[str, object]], model: str) -> dict[str, object]:
    frame = pd.DataFrame(records)
    successful = frame[frame["success"].eq(True)]
    total_runtime = pd.to_numeric(successful["transcription_runtime_seconds"], errors="coerce").sum()
    total_duration = pd.to_numeric(successful["audio_duration_seconds"], errors="coerce").sum()
    overall_wer = pd.to_numeric(frame["wer"], errors="coerce").mean()
    return {
        "model": model,
        "successful": int(len(successful)),
        "failed": int(len(frame) - len(successful)),
        "total_transcription_runtime_seconds": float(total_runtime),
        "average_runtime_per_wav_seconds": float(total_runtime / len(successful)) if len(successful) else float("nan"),
        "real_time_factor": float(total_runtime / total_duration) if total_duration else float("nan"),
        "overall_wer": float(overall_wer) if pd.notna(overall_wer) else float("nan"),
    }


def main() -> int:
    """Run the explicit local ASR pilot; no TEST audio or model training is involved."""

    if not FEATURE_TABLE_PATH.is_file():
        print(f"TRAIN linguistic feature table not found: {FEATURE_TABLE_PATH}")
        return 1
    train_table = assert_train_only(pd.read_csv(FEATURE_TABLE_PATH))
    pilot_participants = select_pilot_participants(train_table)
    dataset_root = get_dataset_root()
    directory_index = build_directory_index(non_hidden_directories(dataset_root))
    total_wavs = len(pilot_participants) * len(PILOT_TASKS)

    print("Whisper ASR Pilot")
    print("=================")
    print(f"TRAIN participants: {len(pilot_participants)}")
    print("TEST participants used: 0")
    print(f"WAV files: {total_wavs}")

    TRANSCRIPT_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    RESULT_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    model_summaries: list[dict[str, object]] = []
    wer_summaries: list[pd.DataFrame] = []
    stability_records: list[dict[str, object]] = []

    for model_name in PILOT_MODELS:
        config = WhisperTranscriptionConfig(model_size=model_name)
        model = load_whisper_model(config)
        records: list[dict[str, object]] = []
        transcript_rows: list[dict[str, object]] = []
        progress = 0
        for participant in pilot_participants.itertuples(index=False):
            participant_series = pd.Series(participant._asdict())
            participant_directory = participant_directory_for_id(
                directory_index, participant_series["participant_id"]
            )
            wavs_by_task = task_wav_paths(participant_directory) if participant_directory else {
                task: [] for task in PILOT_TASKS
            }
            for task in PILOT_TASKS:
                progress += 1
                print(f"[{progress}/{total_wavs}] {model_name} {task}", flush=True)
                paths = wavs_by_task[task]
                if len(paths) != 1:
                    transcript_row = empty_transcript_row(participant_series["participant_id"], task, model_name)
                    transcript_rows.append(transcript_row)
                    records.append({**transcript_row, "wer": float("nan")})
                    continue
                try:
                    transcription = transcribe_audio(model, paths[0], config)
                    transcript_row = {
                        "participant_id": participant_series["participant_id"],
                        "task": task,
                        "model": model_name,
                        "transcript_text": transcription.transcript_text,
                        "detected_language": transcription.detected_language,
                        "audio_duration_seconds": transcription.audio_duration_seconds,
                        "transcription_runtime_seconds": transcription.transcription_runtime_seconds,
                        "success": True,
                    }
                    manual_text = _manual_text_for_task(participant_directory, task)
                    wer = word_error_rate(manual_text, transcription.transcript_text) if manual_text is not None else float("nan")
                    if manual_text is not None:
                        stability_records.extend(
                            feature_stability_rows(model_name, transcription.transcript_text, manual_text)
                        )
                    transcript_rows.append(transcript_row)
                    records.append({**transcript_row, "wer": wer})
                except (OSError, RuntimeError, ValueError) as error:
                    transcript_row = empty_transcript_row(participant_series["participant_id"], task, model_name)
                    transcript_rows.append(transcript_row)
                    records.append({**transcript_row, "wer": float("nan")})
                    print(f"  failed: {error.__class__.__name__}", flush=True)

        output_frame(transcript_rows).to_csv(
            TRANSCRIPT_OUTPUT_DIRECTORY / f"{model_name.replace('.', '_')}.csv", index=False
        )
        model_summaries.append(_model_summary(records, model_name))
        wer_summaries.append(_wer_summary(records, model_name))

    model_comparison = pd.DataFrame(model_summaries)
    wer_by_task = pd.concat(wer_summaries, ignore_index=True)
    feature_stability = summarise_feature_stability(stability_records)
    model_comparison.to_csv(RESULT_OUTPUT_DIRECTORY / "model_comparison.csv", index=False)
    wer_by_task.to_csv(RESULT_OUTPUT_DIRECTORY / "wer_by_task.csv", index=False)
    feature_stability.to_csv(RESULT_OUTPUT_DIRECTORY / "feature_stability.csv", index=False)

    for model_name in PILOT_MODELS:
        summary = model_comparison.set_index("model").loc[model_name]
        model_wer = wer_by_task[wer_by_task["model"].eq(model_name)].set_index("task")
        print(f"\n{model_name.upper()}")
        print("-" * len(model_name))
        print(f"Successful: {int(summary['successful'])}")
        print(f"Failed: {int(summary['failed'])}")
        print(f"Average runtime/WAV: {summary['average_runtime_per_wav_seconds']:.2f} seconds")
        print(f"Real-time factor: {summary['real_time_factor']:.3f}")
        print(f"Overall WER: {summary['overall_wer']:.3f}")
        for task in PILOT_TASKS:
            print(f"{task} WER: {model_wer.loc[task, 'wer']:.3f}")
    print("\nFeature stability: saved mean absolute differences without pause-annotation features.")
    print(f"Saved ASR transcripts: {TRANSCRIPT_OUTPUT_DIRECTORY}")
    print(f"Saved ASR pilot results: {RESULT_OUTPUT_DIRECTORY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
