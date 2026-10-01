"""Deployment-aligned linguistic features derived from ASR text only."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

from ml.audio.acoustic_features import speech_rate_features
from ml.data.development_guard import assert_train_only
from ml.models.modality_comparison import select_pure_acoustic_feature_columns
from ml.nlp.linguistic_features import extract_linguistic_features, p_initial_features
from ml.nlp.transcript_parser import parse_transcript_text


TASKS = ("SFT", "PFT", "CTD")
# Manual pause annotations are deliberately absent: ASR output does not carry
# compatible annotation markup, so zeroes would be fabricated values.
ASR_COMPATIBLE_FEATURE_NAMES = (
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
PFT_FEATURE_NAMES = ("p_initial_word_count", "p_initial_ratio")
BOOKKEEPING_COLUMNS = ("participant_id", "diagnosis", "screening_label", "Split")


def extract_asr_linguistic_features(transcript_text: str, task: str) -> dict[str, float | int]:
    """Use the established linguistic formulas while excluding pause-annotation features."""

    normalized_task = task.upper()
    if normalized_task not in TASKS:
        raise ValueError(f"Unsupported ASR task: {task}")
    parsed = parse_transcript_text(transcript_text)
    all_features = extract_linguistic_features(parsed)
    features = {name: all_features[name] for name in ASR_COMPATIBLE_FEATURE_NAMES}
    if normalized_task == "PFT":
        features.update(p_initial_features(parsed))
    return features


def validate_asr_transcripts(transcripts: pd.DataFrame, participant_ids: Iterable[object]) -> None:
    """Require a complete, unique TRAIN participant/task ASR transcription set."""

    required = {"participant_id", "task", "transcript_text", "model"}
    missing = required.difference(transcripts.columns)
    if missing:
        raise ValueError(f"ASR transcripts are missing required columns: {sorted(missing)}")
    if transcripts.duplicated(["participant_id", "task"]).any():
        raise ValueError("ASR transcripts contain duplicate participant/task rows.")
    expected_keys = {(str(participant_id), task) for participant_id in participant_ids for task in TASKS}
    actual_keys = {(str(row.participant_id), str(row.task).upper()) for row in transcripts.itertuples()}
    if actual_keys != expected_keys:
        raise ValueError("ASR transcripts do not contain exactly one SFT/PFT/CTD row per TRAIN participant.")


def _pivot_task_features(per_recording: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    values = per_recording.set_index(["participant_id", "task"])[
        list(ASR_COMPATIBLE_FEATURE_NAMES) + list(PFT_FEATURE_NAMES)
    ]
    expected_index = pd.MultiIndex.from_product(
        [metadata["participant_id"], TASKS], names=["participant_id", "task"]
    )
    values = values.reindex(expected_index)
    pivoted = values.unstack("task")
    pivoted.columns = [f"{task.casefold()}_{feature}" for feature, task in pivoted.columns]
    # P-initial measurements are meaningful only for PFT; drop invented SFT/CTD NaN columns.
    unwanted = [
        column
        for column in pivoted.columns
        if column.endswith(PFT_FEATURE_NAMES) and not column.startswith("pft_")
    ]
    pivoted = pivoted.drop(columns=unwanted)
    return metadata.set_index("participant_id").join(pivoted, how="left").reset_index()


def build_asr_feature_tables(
    transcripts: pd.DataFrame,
    train_metadata: pd.DataFrame,
    acoustic_train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build ASR-only and deployment-aligned TRAIN tables from complete ASR results.

    The combined deployment table has ASR linguistic features, the established
    pure acoustic subset, and new ASR-count-derived rates. Existing manual
    transcript rates in ``acoustic_train`` are intentionally not carried over.
    """

    metadata = assert_train_only(train_metadata)
    acoustic = assert_train_only(acoustic_train)
    if metadata["participant_id"].duplicated().any() or acoustic["participant_id"].duplicated().any():
        raise ValueError("TRAIN metadata/acoustic tables must have unique participant IDs.")
    if set(metadata["participant_id"]) != set(acoustic["participant_id"]):
        raise ValueError("TRAIN metadata and acoustic participant IDs do not match.")
    validate_asr_transcripts(transcripts, metadata["participant_id"])

    transcript_frame = transcripts.copy()
    transcript_frame["task"] = transcript_frame["task"].astype(str).str.upper()
    feature_rows: list[dict[str, object]] = []
    for row in transcript_frame.itertuples(index=False):
        feature_rows.append(
            {
                "participant_id": row.participant_id,
                "task": row.task,
                **extract_asr_linguistic_features(str(row.transcript_text), str(row.task)),
            }
        )
    per_recording = pd.DataFrame(feature_rows)
    for column in PFT_FEATURE_NAMES:
        if column not in per_recording:
            per_recording[column] = np.nan
    metadata_columns = metadata[list(BOOKKEEPING_COLUMNS)].copy()
    asr_linguistic = _pivot_task_features(per_recording, metadata_columns)
    if len(asr_linguistic) != len(metadata) or asr_linguistic["participant_id"].duplicated().any():
        raise AssertionError("ASR linguistic pivot must yield one unique row per TRAIN participant.")

    pure_acoustic_columns = select_pure_acoustic_feature_columns(acoustic, "ALL")
    acoustic_values = acoustic.set_index("participant_id")[pure_acoustic_columns]
    per_recording = per_recording.set_index(["participant_id", "task"])
    rate_rows: list[dict[str, object]] = []
    for participant_id in metadata["participant_id"]:
        row: dict[str, object] = {"participant_id": participant_id}
        for task in TASKS:
            prefix = task.casefold()
            word_count = per_recording.loc[(participant_id, task), "word_count"]
            rates = speech_rate_features(
                word_count,
                float(acoustic_values.loc[participant_id, f"{prefix}_audio_duration_seconds"]),
                float(acoustic_values.loc[participant_id, f"{prefix}_voiced_duration_seconds"]),
            )
            row.update({f"{prefix}_{name}": value for name, value in rates.items()})
        rate_rows.append(row)
    rate_table = pd.DataFrame(rate_rows).set_index("participant_id")
    deployment = (
        asr_linguistic.set_index("participant_id")
        .join(acoustic_values, how="left")
        .join(rate_table, how="left")
        .reset_index()
    )
    if len(deployment) != len(metadata) or deployment["participant_id"].duplicated().any():
        raise AssertionError("Deployment feature table must yield one unique row per TRAIN participant.")
    return asr_linguistic, deployment, per_recording.reset_index()


def feature_quality_report(feature_table: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    """Return TRAIN-only missing, infinite, and constant-feature diagnostics."""

    rows: list[dict[str, object]] = []
    for column in feature_columns:
        values = pd.to_numeric(feature_table[column], errors="coerce")
        finite_values = values[np.isfinite(values)]
        rows.append(
            {
                "feature": column,
                "missing_count": int(values.isna().sum()),
                "infinite_count": int(np.isinf(values).sum()),
                "constant": bool(len(finite_values) > 0 and finite_values.nunique() <= 1),
            }
        )
    return pd.DataFrame(rows)


def asr_feature_columns(feature_table: pd.DataFrame) -> list[str]:
    """Return ASR-linguistic predictors without bookkeeping labels/IDs."""

    return [column for column in feature_table.columns if column not in BOOKKEEPING_COLUMNS]
