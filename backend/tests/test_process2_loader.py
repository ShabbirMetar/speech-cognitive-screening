from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.process2_loader import (
    collect_file_counts,
    get_dataset_status,
    identifier_keys,
    identify_column,
    task_data_is_complete,
)


def test_identify_column_matches_normalised_candidate() -> None:
    columns = ["Participant ID", "Clinical Diagnosis", "Train/Test Split"]

    assert identify_column(columns, ("participantid", "ids")) == "Participant ID"
    assert identify_column(columns, ("diagnosis",)) == "Clinical Diagnosis"
    assert identify_column(columns, ("split",)) == "Train/Test Split"


def test_identifier_keys_match_zero_padded_folder_suffix() -> None:
    assert "1" in identifier_keys("001")
    assert "1" in identifier_keys("PROCESS-2_rec__001")


def test_dataset_status_and_file_counts_ignore_hidden_directories(tmp_path: Path) -> None:
    (tmp_path / "meta-info.csv").write_text("IDs,diagnosis,Split\n", encoding="utf-8")
    (tmp_path / ".cache").mkdir()
    participant = tmp_path / "recording-001"
    participant.mkdir()
    (participant / "recording-001__SFT.wav").touch()
    (participant / "recording-001__SFT.txt").touch()
    (participant / "notes.csv").touch()

    status = get_dataset_status(tmp_path)
    counts = collect_file_counts(tmp_path)

    assert status.dataset_exists is True
    assert status.metadata_exists is True
    assert counts.participant_directories == 1
    assert counts.wav_files == 1
    assert counts.txt_files == 1
    assert counts.task_files["SFT"].total == 2


def test_complete_task_data_requires_wav_and_transcript_for_every_task(tmp_path: Path) -> None:
    for task in ("SFT", "PFT", "CTD"):
        (tmp_path / f"participant__{task}.wav").touch()
        (tmp_path / f"participant__{task}.txt").touch()

    assert task_data_is_complete(tmp_path) is True

    (tmp_path / "participant__CTD.txt").unlink()

    assert task_data_is_complete(tmp_path) is False
