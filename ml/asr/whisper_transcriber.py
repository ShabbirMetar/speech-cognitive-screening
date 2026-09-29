"""Local, read-only faster-whisper transcription helpers for PROCESS-2 pilots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import soundfile as sf


@dataclass(frozen=True)
class WhisperTranscriptionConfig:
    """CPU-safe local faster-whisper configuration for a reproducible pilot."""

    model_size: str
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "en"
    cpu_threads: int = 4
    num_workers: int = 1


@dataclass(frozen=True)
class WhisperTranscription:
    """A local transcription result with timing metadata and no source mutation."""

    transcript_text: str
    detected_language: str | None
    transcription_runtime_seconds: float
    audio_duration_seconds: float


def load_whisper_model(config: WhisperTranscriptionConfig) -> Any:
    """Instantiate a local faster-whisper model.

    The import is intentionally lazy so pure text/unit-test utilities do not
    download or require a model.  faster-whisper resolves model files locally
    and downloads them only when this function is explicitly used by the pilot.
    """

    try:
        from faster_whisper import WhisperModel
    except ImportError as error:  # pragma: no cover - depends on local installation
        raise RuntimeError(
            "faster-whisper is required for ASR transcription. "
            "Install backend/requirements.txt before running the pilot."
        ) from error
    return WhisperModel(
        config.model_size,
        device=config.device,
        compute_type=config.compute_type,
        cpu_threads=config.cpu_threads,
        num_workers=config.num_workers,
    )


def audio_duration_seconds(audio_path: Path) -> float:
    """Read only WAV header metadata to determine duration without altering audio."""

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
    info = sf.info(audio_path)
    if info.samplerate <= 0:
        raise ValueError(f"Audio file has an invalid sample rate: {audio_path}")
    return float(info.frames / info.samplerate)


def transcribe_audio(
    model: Any, audio_path: Path, config: WhisperTranscriptionConfig
) -> WhisperTranscription:
    """Transcribe one audio file locally and return text plus elapsed-time metadata."""

    duration = audio_duration_seconds(audio_path)
    started_at = perf_counter()
    segments, information = model.transcribe(
        str(audio_path),
        language=config.language,
    )
    transcript = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
    return WhisperTranscription(
        transcript_text=transcript,
        detected_language=getattr(information, "language", None),
        transcription_runtime_seconds=float(perf_counter() - started_at),
        audio_duration_seconds=duration,
    )
