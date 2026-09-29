"""Tests for the strict TRAIN-only model-development guard."""

from __future__ import annotations

import pandas as pd
import pytest

from ml.data.development_guard import DevelopmentDataGuardError, assert_train_only


def feature_rows(splits: list[str], identifiers: list[str] | None = None) -> pd.DataFrame:
    identifiers = identifiers or [f"participant-{index}" for index in range(len(splits))]
    return pd.DataFrame({"participant_id": identifiers, "Split": splits, "sft_word_count": 1.0})


def test_assert_train_only_accepts_unique_train_rows() -> None:
    table = feature_rows(["TRAIN", "TRAIN"])
    assert assert_train_only(table) is table


def test_assert_train_only_rejects_locked_test_rows() -> None:
    with pytest.raises(DevelopmentDataGuardError, match="only Split == TRAIN"):
        assert_train_only(feature_rows(["TEST"]))


def test_assert_train_only_rejects_mixed_rows() -> None:
    with pytest.raises(DevelopmentDataGuardError, match="only Split == TRAIN"):
        assert_train_only(feature_rows(["TRAIN", "TEST"]))


def test_assert_train_only_rejects_duplicate_participant_ids() -> None:
    with pytest.raises(DevelopmentDataGuardError, match="must be unique"):
        assert_train_only(feature_rows(["TRAIN", "TRAIN"], ["same", "same"]))
