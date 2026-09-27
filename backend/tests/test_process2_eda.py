from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.process2_eda import (
    SCREENING_LABEL_COLUMN,
    create_screening_label,
    identify_eda_columns,
    mmse_availability,
    prepare_eda_metadata,
)


def test_create_screening_label_maps_expected_diagnoses() -> None:
    diagnoses = pd.Series(["HC", "MCI", "Dementia"])

    labels = create_screening_label(diagnoses)

    assert labels.tolist() == ["Healthy", "Impaired", "Impaired"]


def test_prepare_eda_metadata_adds_binary_label_without_mutating_source() -> None:
    metadata = pd.DataFrame(
        {
            "diagnosis": ["HC", "MCI", "Dementia"],
            "Split": ["TRAIN", "TRAIN", "TEST"],
            "MMSE": [29, 25, 18],
        }
    )
    columns = identify_eda_columns(metadata)

    prepared = prepare_eda_metadata(metadata, columns)

    assert SCREENING_LABEL_COLUMN not in metadata.columns
    assert prepared[SCREENING_LABEL_COLUMN].tolist() == ["Healthy", "Impaired", "Impaired"]


def test_mmse_availability_preserves_and_reports_missing_values() -> None:
    metadata = pd.DataFrame(
        {
            "diagnosis": ["HC", "MCI", "Dementia"],
            "Split": ["TRAIN", "TRAIN", "TEST"],
            "MMSE": [29, None, 18],
        }
    )
    columns = identify_eda_columns(metadata)

    availability = mmse_availability(metadata, columns)

    assert availability.to_dict() == {"Available": 2, "Missing": 1}
