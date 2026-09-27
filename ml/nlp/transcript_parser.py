"""Read-only parsing for manually prepared PROCESS-2 transcripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


SPEAKER_LINE_PATTERN = re.compile(
    r"^\s*(?P<label>[A-Za-z][A-Za-z0-9 _-]{0,30})\s*:\s*(?P<text>.*)$"
)
PAUSE_PATTERN = re.compile(
    r"\(\s*(?P<seconds>\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\s*\)",
    re.IGNORECASE,
)
TASK_NAMES = ("SFT", "PFT", "CTD")


@dataclass(frozen=True)
class ParsedTranscript:
    """A parsed transcript with source text preserved in memory."""

    participant_text: str
    other_speaker_text: str
    full_text: str
    speaker_labels_detected: tuple[str, ...]
    pause_durations_seconds: tuple[float, ...]
    pause_count: int
    speaker_attribution: str
    line_count: int

    @property
    def pause_total_seconds(self) -> float:
        return float(sum(self.pause_durations_seconds))

    @property
    def pause_mean_seconds(self) -> float:
        if not self.pause_durations_seconds:
            return float("nan")
        return self.pause_total_seconds / self.pause_count

    @property
    def pause_max_seconds(self) -> float:
        if not self.pause_durations_seconds:
            return float("nan")
        return max(self.pause_durations_seconds)


def extract_pause_durations(text: str) -> tuple[float, ...]:
    """Extract parenthesised second-based pause annotations before text cleaning."""

    return tuple(float(match.group("seconds")) for match in PAUSE_PATTERN.finditer(text))


def remove_pause_annotations(text: str) -> str:
    """Remove recognised pause annotations so they never count as words."""

    return PAUSE_PATTERN.sub(" ", text)


def _normalise_speaker_label(label: str) -> str:
    return re.sub(r"\s+", " ", label).strip()


def parse_transcript_text(raw_text: str) -> ParsedTranscript:
    """Parse labelled participant/other text without changing the source file.

    Fallback policy: when a transcript has no speaker labels at all, its text is
    used as a participant-text candidate and marked ``unlabeled_fallback``. This
    is needed because structural inspection found many unlabeled task files. If
    any labels are present but no ``Pat`` label is present, participant text is
    empty: no speaker identity is inferred in that situation.
    """

    labels: list[str] = []
    participant_segments: list[str] = []
    other_segments: list[str] = []
    participant_pause_durations: list[float] = []
    active_label: str | None = None
    saw_any_label = False

    for line in raw_text.splitlines():
        match = SPEAKER_LINE_PATTERN.match(line)
        if match:
            saw_any_label = True
            active_label = _normalise_speaker_label(match.group("label"))
            if active_label not in labels:
                labels.append(active_label)
            content = match.group("text")
        else:
            content = line

        label_key = active_label.casefold() if active_label else None
        if label_key == "pat":
            participant_segments.append(content)
            participant_pause_durations.extend(extract_pause_durations(content))
        elif label_key == "oth":
            other_segments.append(content)

    if not saw_any_label:
        participant_text = remove_pause_annotations(raw_text)
        pause_durations = extract_pause_durations(raw_text)
        attribution = "unlabeled_fallback"
    elif any(label.casefold() == "pat" for label in labels):
        participant_text = remove_pause_annotations("\n".join(participant_segments))
        pause_durations = tuple(participant_pause_durations)
        attribution = "explicit_pat"
    else:
        participant_text = ""
        pause_durations = ()
        attribution = "no_participant_label"

    return ParsedTranscript(
        participant_text=participant_text,
        other_speaker_text=remove_pause_annotations("\n".join(other_segments)),
        full_text=raw_text,
        speaker_labels_detected=tuple(labels),
        pause_durations_seconds=tuple(pause_durations),
        pause_count=len(pause_durations),
        speaker_attribution=attribution,
        line_count=len(raw_text.splitlines()),
    )


def parse_transcript_file(transcript_path: Path) -> ParsedTranscript:
    """Read and parse one UTF-8 transcript without modifying it."""

    return parse_transcript_text(transcript_path.read_text(encoding="utf-8-sig"))


def task_name_from_path(transcript_path: Path) -> str | None:
    """Identify the known task label from a filename without relying on folder names."""

    upper_name = transcript_path.name.upper()
    return next((task for task in TASK_NAMES if task in upper_name), None)


def transcript_paths_by_task(participant_directory: Path) -> dict[str, list[Path]]:
    """Return deterministic transcript paths grouped by recognised task name."""

    paths = {task: [] for task in TASK_NAMES}
    for path in sorted(participant_directory.rglob("*.txt"), key=lambda item: item.name.casefold()):
        task_name = task_name_from_path(path)
        if task_name is not None:
            paths[task_name].append(path)
    return paths
