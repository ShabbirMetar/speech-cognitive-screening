"""Reusable, non-clinical helpers for frozen screening-model explanations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np

from ml.models.modality_comparison import PURE_ACOUSTIC_SUFFIXES


FROZEN_REPRESENTATION = "ASR_RATE_PLUS_CTD"
RESEARCH_OPERATING_THRESHOLD = 0.53
ASR_RATE_SUFFIXES = ("recording_word_rate_wpm", "articulation_rate_wpm")

FeatureGroup = Literal["ASR_LINGUISTIC", "ASR_RATE", "CTD_ACOUSTIC"]


def feature_group(feature: str) -> FeatureGroup:
    """Return the frozen modality group for one selected predictor."""

    if feature.endswith(ASR_RATE_SUFFIXES):
        return "ASR_RATE"
    if feature.startswith("ctd_") and feature.removeprefix("ctd_") in PURE_ACOUSTIC_SUFFIXES:
        return "CTD_ACOUSTIC"
    return "ASR_LINGUISTIC"


def feature_groups(features: Iterable[str]) -> dict[str, FeatureGroup]:
    """Map all selected predictors to their three frozen feature groups."""

    return {feature: feature_group(feature) for feature in features}


def human_feature_label(feature: str) -> str:
    """Return a readable display label without adding a clinical interpretation."""

    task, separator, suffix = feature.partition("_")
    if not separator:
        return feature
    task_label = task.upper()
    labels = {
        "word_count": "Word Count",
        "unique_word_count": "Unique Word Count",
        "type_token_ratio": "Lexical Diversity (TTR)",
        "average_word_length": "Average Word Length",
        "repeated_word_count": "Repeated Word Count",
        "repetition_ratio": "Repetition Ratio",
        "filler_count": "Filler Count",
        "filler_ratio": "Filler Ratio",
        "brunet_index": "Brunet Vocabulary Richness",
        "honore_statistic": "Honoré Vocabulary Richness",
        "p_initial_word_count": "P-initial Word Count",
        "p_initial_ratio": "P-initial Word Ratio",
        "recording_word_rate_wpm": "Speech Rate",
        "articulation_rate_wpm": "Articulation Rate",
        "audio_duration_seconds": "Recording Duration",
        "voiced_duration_seconds": "Voiced Duration",
        "silence_duration_seconds": "Silence Duration",
        "silence_ratio": "Silence Ratio",
        "pause_count": "Pause Count",
        "pause_total_seconds": "Total Pause Duration",
        "pause_mean_seconds": "Mean Pause Duration",
        "pause_max_seconds": "Maximum Pause Duration",
        "f0_mean_hz": "Mean Pitch",
        "f0_std_hz": "Pitch Variability",
        "f0_range_hz": "Pitch Range",
        "rms_mean": "Mean Voice Energy",
        "rms_std": "Voice-Energy Variability",
    }
    return f"{task_label} {labels.get(suffix, suffix.replace('_', ' ').title())}"


def coefficient_direction(coefficient: float) -> str:
    """Describe a standardized coefficient direction, not a causal effect."""

    if coefficient > 0:
        return "Toward Possible impairment-like class"
    if coefficient < 0:
        return "Toward Healthy-like class"
    return "No directional effect"


def screening_score(raw_model_score: float) -> float:
    """Map a raw classifier score to the 0–100 display-only screening score."""

    score = float(raw_model_score)
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("raw_model_score must be a finite value between 0 and 1.")
    return score * 100.0


def screening_classification(
    raw_model_score: float, threshold: float = RESEARCH_OPERATING_THRESHOLD
) -> str:
    """Apply the TRAIN-derived research operating point to a raw model score."""

    score = float(raw_model_score)
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("raw_model_score must be a finite value between 0 and 1.")
    return (
        "Possible impairment-like speech pattern"
        if score >= threshold
        else "Healthy-like speech pattern"
    )


def format_feature_contribution(
    feature: str,
    contribution_value: float,
    observed_value: float,
    reference_median: float,
) -> str:
    """Format one model contribution using observed-vs-TRAIN context and its sign.

    The contribution sign determines the model-output direction. Higher/lower wording is
    emitted only when a finite observed value can be compared with the TRAIN
    reference median; no clinical meaning is attached to the feature itself.
    """

    label = human_feature_label(feature)
    contribution = float(contribution_value)
    if contribution > 0:
        destination = "possible-impairment-like screening output"
    elif contribution < 0:
        destination = "healthy-like screening output"
    else:
        return f"{label} had negligible influence on this model output."

    value = float(observed_value)
    reference = float(reference_median)
    if np.isfinite(value) and np.isfinite(reference):
        if value > reference:
            relation = "Higher"
        elif value < reference:
            relation = "Lower"
        else:
            relation = "An observed"
        if relation == "An observed":
            subject = f"{relation} {label}"
        else:
            subject = f"{relation} {label} relative to the TRAIN reference"
    else:
        subject = f"The median-imputed {label} value"
    return f"{subject} contributed toward the model's {destination}."


def select_example_indices(
    participant_ids: Sequence[object], scores: Sequence[float], threshold: float = RESEARCH_OPERATING_THRESHOLD
) -> dict[str, int]:
    """Choose deterministic score-position examples without inspecting labels."""

    ids = [str(participant_id) for participant_id in participant_ids]
    values = np.asarray(scores, dtype=float)
    if len(ids) != len(values) or not len(values):
        raise ValueError("participant_ids and scores must be non-empty and have the same length.")
    if not np.isfinite(values).all():
        raise ValueError("Example selection requires finite model scores.")

    targets = {
        "low_score_quantile": float(np.quantile(values, 0.10)),
        "near_research_threshold": float(threshold),
        "high_score_quantile": float(np.quantile(values, 0.90)),
    }
    selected: dict[str, int] = {}
    used: set[int] = set()
    for name, target in targets.items():
        ordered = sorted(range(len(values)), key=lambda index: (abs(values[index] - target), ids[index]))
        index = next(index for index in ordered if index not in used)
        selected[name] = index
        used.add(index)
    return selected
