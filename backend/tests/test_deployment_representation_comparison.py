"""Synthetic safeguards for the deployment representation comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.models.modality_comparison import PURE_ACOUSTIC_SUFFIXES
from scripts.compare_deployment_representations import (
    _align_train_table,
    asr_rate_feature_columns,
    manual_asr_compatible_feature_columns,
    pure_acoustic_feature_columns,
    validate_asr_rate_provenance,
)


def metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_id": ["participant-1", "participant-2"],
            "diagnosis": ["HC", "MCI"],
            "screening_label": ["Healthy", "Impaired"],
            "Split": ["TRAIN", "TRAIN"],
        }
    )


def asr_and_deployment_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    asr = metadata()
    deployment = metadata()
    for prefix, word_counts in (("sft_", [3.0, 6.0]), ("pft_", [2.0, 4.0]), ("ctd_", [5.0, 10.0])):
        asr[f"{prefix}word_count"] = word_counts
        deployment[f"{prefix}word_count"] = word_counts
        deployment[f"{prefix}audio_duration_seconds"] = 60.0
        deployment[f"{prefix}voiced_duration_seconds"] = 30.0
        deployment[f"{prefix}recording_word_rate_wpm"] = word_counts
        deployment[f"{prefix}articulation_rate_wpm"] = np.asarray(word_counts) * 2
    return asr, deployment


def test_selectors_keep_manual_pause_annotations_and_rates_out_of_asr_schema() -> None:
    asr, _ = asr_and_deployment_tables()
    manual = asr.copy()
    manual["sft_pause_annotation_count"] = 1.0
    assert manual_asr_compatible_feature_columns(manual, asr) == [
        "sft_word_count", "pft_word_count", "ctd_word_count"
    ]
    deployment = asr.copy()
    for task in ("sft", "pft", "ctd"):
        for suffix in PURE_ACOUSTIC_SUFFIXES:
            deployment[f"{task}_{suffix}"] = 1.0
        deployment[f"{task}_recording_word_rate_wpm"] = 1.0
        deployment[f"{task}_articulation_rate_wpm"] = 2.0
    assert len(pure_acoustic_feature_columns(deployment)) == 39
    assert len(asr_rate_feature_columns(deployment)) == 6


def test_rate_provenance_requires_asr_word_counts() -> None:
    asr, deployment = asr_and_deployment_tables()
    validate_asr_rate_provenance(asr, deployment)
    deployment.loc[0, "sft_recording_word_rate_wpm"] = 999.0
    with pytest.raises(ValueError, match="provenance"):
        validate_asr_rate_provenance(asr, deployment)


def test_alignment_rejects_test_rows() -> None:
    reference = metadata()
    candidate = metadata()
    candidate.loc[0, "Split"] = "TEST"
    with pytest.raises(ValueError, match="only Split == TRAIN"):
        _align_train_table(reference, candidate, "candidate")
