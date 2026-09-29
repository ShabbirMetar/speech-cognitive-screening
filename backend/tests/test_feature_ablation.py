"""Synthetic tests for the deterministic linguistic redundancy ablation."""

from __future__ import annotations

import pandas as pd
import pytest

from ml.models.classical_comparison import resolve_linguistic_feature_columns
from scripts.compare_reduced_linguistic_features import REDUNDANT_FEATURES, reduced_feature_columns


def all_manual_feature_names() -> list[str]:
    """Construct the current 44-column task schema without real participant data."""

    common_suffixes = (
        "word_count",
        "unique_word_count",
        "type_token_ratio",
        "average_word_length",
        "repeated_word_count",
        "repetition_ratio",
        "filler_count",
        "filler_ratio",
        "pause_annotation_count",
        "pause_annotation_total_seconds",
        "pause_annotation_mean_seconds",
        "pause_annotation_max_seconds",
        "brunet_index",
        "honore_statistic",
    )
    columns = [f"{task}_{suffix}" for task in ("sft", "pft", "ctd") for suffix in common_suffixes]
    columns.extend(("pft_p_initial_word_count", "pft_p_initial_ratio"))
    return columns


def test_reduced_feature_set_removes_only_the_six_prespecified_columns() -> None:
    original = all_manual_feature_names()
    reduced = reduced_feature_columns(original)
    assert len(original) == 44
    assert len(reduced) == 38
    assert set(original).difference(reduced) == REDUNDANT_FEATURES
    assert "sft_type_token_ratio" in reduced
    assert "pft_p_initial_word_count" in reduced
    assert "ctd_brunet_index" in reduced
    assert "ctd_honore_statistic" in reduced


def test_reduced_feature_set_requires_every_expected_redundant_column() -> None:
    with pytest.raises(ValueError, match="missing expected redundant columns"):
        reduced_feature_columns(
            [
                column
                for column in all_manual_feature_names()
                if column != "sft_repeated_word_count"
            ]
        )


def test_feature_override_accepts_only_existing_linguistic_columns() -> None:
    table = pd.DataFrame(
        {
            "participant_id": ["synthetic-1"],
            "diagnosis": ["HC"],
            "screening_label": ["Healthy"],
            "Split": ["TRAIN"],
            "sft_word_count": [2],
            "pft_word_count": [2],
            "ctd_word_count": [2],
        }
    )
    assert resolve_linguistic_feature_columns(table, ["sft_word_count", "ctd_word_count"]) == [
        "sft_word_count",
        "ctd_word_count",
    ]
    with pytest.raises(ValueError, match="only existing linguistic columns"):
        resolve_linguistic_feature_columns(table, ["participant_id"])
