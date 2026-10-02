"""Synthetic safeguards for compact and late fusion strategy comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.compare_fusion_strategies import (
    ALPHA_VALUES,
    _align_to_asr,
    late_fusion_probabilities,
)


def train_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_id": ["one", "two"],
            "diagnosis": ["HC", "MCI"],
            "screening_label": ["Healthy", "Impaired"],
            "Split": ["TRAIN", "TRAIN"],
        }
    )


def test_late_fusion_uses_the_requested_probability_formula() -> None:
    result = late_fusion_probabilities(np.array([0.8, 0.2]), np.array([0.3, 0.7]), 0.7)
    assert np.allclose(result, [0.65, 0.35])
    assert ALPHA_VALUES == (0.6, 0.7, 0.8, 0.9)


def test_late_fusion_rejects_unsupported_alpha_or_mismatched_arrays() -> None:
    with pytest.raises(ValueError, match="alpha"):
        late_fusion_probabilities(np.array([0.2]), np.array([0.3]), 0.5)
    with pytest.raises(ValueError, match="same shape"):
        late_fusion_probabilities(np.array([0.2]), np.array([0.3, 0.4]), 0.6)


def test_alignment_rejects_test_rows_before_fusion() -> None:
    asr = train_table()
    deployment = train_table()
    deployment.loc[1, "Split"] = "TEST"
    with pytest.raises(ValueError, match="only Split == TRAIN"):
        _align_to_asr(asr, deployment)
