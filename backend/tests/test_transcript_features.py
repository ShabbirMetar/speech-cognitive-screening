from pathlib import Path
import math
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.nlp.linguistic_features import (
    brunet_index,
    extract_linguistic_features,
    honore_statistic,
    p_initial_features,
)
from ml.nlp.transcript_parser import parse_transcript_text


def test_parser_extracts_pat_and_excludes_oth_with_pauses() -> None:
    parsed = parse_transcript_text(
        "Pat: Um pizza pizza (2 seconds)\nOth: unrelated prompt (1 second)"
    )

    assert parsed.participant_text.strip() == "Um pizza pizza"
    assert parsed.other_speaker_text.strip() == "unrelated prompt"
    assert parsed.speaker_labels_detected == ("Pat", "Oth")
    assert parsed.pause_durations_seconds == (2.0,)
    assert parsed.pause_count == 1
    assert parsed.speaker_attribution == "explicit_pat"


def test_linguistic_counts_and_p_initial_features() -> None:
    parsed = parse_transcript_text("Pat: Um pizza pizza pear")

    features = extract_linguistic_features(parsed)
    p_features = p_initial_features(parsed)

    assert features["word_count"] == 4
    assert features["unique_word_count"] == 3
    assert features["repeated_word_count"] == 1
    assert features["type_token_ratio"] == 0.75
    assert features["filler_count"] == 1
    assert features["filler_ratio"] == 0.25
    assert p_features == {"p_initial_word_count": 3, "p_initial_ratio": 0.75}


def test_empty_transcript_returns_safe_counts_and_nan_ratios() -> None:
    features = extract_linguistic_features(parse_transcript_text(""))

    assert features["word_count"] == 0
    assert features["unique_word_count"] == 0
    assert features["repeated_word_count"] == 0
    assert math.isnan(features["type_token_ratio"])
    assert math.isnan(features["average_word_length"])
    assert math.isnan(features["honore_statistic"])


def test_unattributable_speaker_text_becomes_missing_features() -> None:
    parsed = parse_transcript_text("Oth: words that cannot be attributed")
    features = extract_linguistic_features(parsed)

    assert parsed.speaker_attribution == "no_participant_label"
    assert math.isnan(features["word_count"])
    assert math.isnan(features["pause_annotation_count"])


def test_successfully_parsed_transcript_without_pauses_uses_zero_pause_summary() -> None:
    features = extract_linguistic_features(parse_transcript_text("Pat: verified response"))

    assert features["pause_annotation_count"] == 0
    assert features["pause_annotation_total_seconds"] == 0
    assert features["pause_annotation_mean_seconds"] == 0
    assert features["pause_annotation_max_seconds"] == 0


def test_unlabeled_transcript_uses_documented_fallback() -> None:
    parsed = parse_transcript_text("apple pear (1.5 seconds)")

    assert parsed.speaker_labels_detected == ()
    assert parsed.speaker_attribution == "unlabeled_fallback"
    assert parsed.participant_text.strip() == "apple pear"
    assert parsed.pause_durations_seconds == (1.5,)


def test_brunet_and_honore_edge_cases() -> None:
    assert math.isnan(brunet_index(["only"]))
    assert math.isnan(honore_statistic(["only"]))
    assert math.isfinite(brunet_index(["cat", "cat", "dog"]))
    assert math.isfinite(honore_statistic(["cat", "cat", "dog"]))
