"""Guards that keep model-development inputs inside the official TRAIN split."""

from __future__ import annotations

import pandas as pd


SPLIT_COLUMN = "Split"
PARTICIPANT_ID_COLUMN = "participant_id"


class DevelopmentDataGuardError(ValueError):
    """Raised when a model-development table is not a unique TRAIN-only table."""


def assert_train_only(feature_table: pd.DataFrame) -> pd.DataFrame:
    """Validate and return a unique official-TRAIN feature table.

    Model-development code must call this before selecting features, fitting
    preprocessing, cross-validating, or tuning hyperparameters.  The guard is
    intentionally stricter than filtering: a mixed or locked-TEST input is an
    error rather than something a caller can silently reduce to TRAIN rows.
    """

    required_columns = {SPLIT_COLUMN, PARTICIPANT_ID_COLUMN}
    missing_columns = required_columns.difference(feature_table.columns)
    if missing_columns:
        raise DevelopmentDataGuardError(
            f"Feature table is missing required development columns: {sorted(missing_columns)}"
        )

    if feature_table[PARTICIPANT_ID_COLUMN].isna().any():
        raise DevelopmentDataGuardError("participant_id contains missing values.")
    if feature_table[PARTICIPANT_ID_COLUMN].duplicated().any():
        raise DevelopmentDataGuardError(
            "participant_id values must be unique for model development."
        )

    normalized_split = feature_table[SPLIT_COLUMN].astype("string").str.strip().str.upper()
    if not normalized_split.eq("TRAIN").all():
        found_splits = sorted(normalized_split.dropna().unique().tolist())
        raise DevelopmentDataGuardError(
            "Model development accepts only Split == TRAIN rows; "
            f"found split values: {found_splits}."
        )
    return feature_table
