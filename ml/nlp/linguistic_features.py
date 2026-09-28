"""Small, interpretable linguistic features for parsed manual transcripts."""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import Iterable

from ml.nlp.transcript_parser import ParsedTranscript


DEFAULT_FILLERS = frozenset({"um", "uh", "erm", "er", "hmm"})
TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
LINGUISTIC_FEATURE_NAMES = (
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


def tokenize_text(text: str) -> list[str]:
    """Tokenize English-like words, lowercasing and excluding annotations/punctuation."""

    return [match.group(0).casefold() for match in TOKEN_PATTERN.finditer(text)]


def brunet_index(tokens: Iterable[str]) -> float:
    """Return Brunet's index when at least two tokens are available, else NaN."""

    token_list = list(tokens)
    token_count = len(token_list)
    unique_count = len(set(token_list))
    if token_count < 2 or unique_count == 0:
        return float("nan")
    return float(token_count ** (unique_count ** -0.165))


def honore_statistic(tokens: Iterable[str]) -> float:
    """Return Honoré's statistic when its denominator is non-zero, else NaN."""

    token_list = list(tokens)
    token_count = len(token_list)
    frequencies = Counter(token_list)
    unique_count = len(frequencies)
    hapax_count = sum(count == 1 for count in frequencies.values())
    denominator = 1 - (hapax_count / unique_count) if unique_count else 0
    if token_count < 2 or denominator <= 0:
        return float("nan")
    return float(100 * math.log(token_count) / denominator)


def unavailable_linguistic_features() -> dict[str, float]:
    """Return NaN features for a missing or unattributable participant transcript."""

    return {feature_name: float("nan") for feature_name in LINGUISTIC_FEATURE_NAMES}


def extract_linguistic_features(
    parsed_transcript: ParsedTranscript,
    filler_tokens: Iterable[str] = DEFAULT_FILLERS,
) -> dict[str, float | int]:
    """Extract deterministic participant-only features from a parsed transcript."""

    if parsed_transcript.speaker_attribution == "no_participant_label":
        return unavailable_linguistic_features()

    tokens = tokenize_text(parsed_transcript.participant_text)
    token_count = len(tokens)
    frequencies = Counter(tokens)
    unique_count = len(frequencies)
    repeated_count = token_count - unique_count
    fillers = {token.casefold() for token in filler_tokens}
    filler_count = sum(token in fillers for token in tokens)

    return {
        "word_count": token_count,
        "unique_word_count": unique_count,
        "type_token_ratio": unique_count / token_count if token_count else float("nan"),
        "average_word_length": (
            sum(len(token) for token in tokens) / token_count if token_count else float("nan")
        ),
        "repeated_word_count": repeated_count,
        "repetition_ratio": repeated_count / token_count if token_count else float("nan"),
        "filler_count": filler_count,
        "filler_ratio": filler_count / token_count if token_count else float("nan"),
        "pause_annotation_count": parsed_transcript.pause_count,
        "pause_annotation_total_seconds": parsed_transcript.pause_total_seconds,
        # Zero means a successfully parsed participant transcript had no detected
        # pause annotation. Unavailable participant attribution remains NaN above.
        "pause_annotation_mean_seconds": (
            parsed_transcript.pause_mean_seconds if parsed_transcript.pause_count else 0.0
        ),
        "pause_annotation_max_seconds": (
            parsed_transcript.pause_max_seconds if parsed_transcript.pause_count else 0.0
        ),
        "brunet_index": brunet_index(tokens),
        "honore_statistic": honore_statistic(tokens),
    }


def p_initial_features(parsed_transcript: ParsedTranscript) -> dict[str, float | int]:
    """Return simple P-initial token features for the phonemic fluency task."""

    tokens = tokenize_text(parsed_transcript.participant_text)
    token_count = len(tokens)
    p_initial_count = sum(token.startswith("p") for token in tokens)
    return {
        "p_initial_word_count": p_initial_count,
        "p_initial_ratio": p_initial_count / token_count if token_count else float("nan"),
    }
