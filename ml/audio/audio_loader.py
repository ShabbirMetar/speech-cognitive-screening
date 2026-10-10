"""Read-only WAV loading and validation helpers for the acoustic pilot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


class AudioValidationError(RuntimeError):
    """Raised when a WAV file cannot safely enter acoustic feature extraction."""


@dataclass(frozen=True)
class AudioData:
    """In-memory analysis waveform and source-file audio metadata."""

    waveform: np.ndarray
    sample_rate: int
    channel_count: int


def validate_waveform(waveform: np.ndarray, sample_rate: int) -> None:
    """Validate an in-memory waveform without modifying its source file."""

    values = np.asarray(waveform)
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise AudioValidationError(f"Invalid audio sample rate: {sample_rate!r}")
    if values.size == 0:
        raise AudioValidationError("Audio waveform is empty.")
    if not np.issubdtype(values.dtype, np.number):
        raise AudioValidationError("Audio waveform must contain numeric samples.")
    if not np.isfinite(values).all():
        raise AudioValidationError("Audio waveform contains non-finite numeric values.")


def load_wav_read_only(path: Path) -> AudioData:
    """Read a WAV file without resampling, rewriting, or modifying it.

    Multi-channel recordings are averaged only in memory for mono acoustic
    analysis.  The original source channel count is retained in ``AudioData``.
    """

    if not path.is_file():
        raise AudioValidationError(f"Audio file does not exist: {path}")
    try:
        info = sf.info(path)
        samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as error:
        raise AudioValidationError(f"Audio file is not readable: {path}") from error

    channel_count = int(info.channels)
    if channel_count <= 0:
        raise AudioValidationError("Audio source reported no channels.")
    validate_waveform(samples, int(sample_rate))
    waveform = samples.mean(axis=1, dtype=np.float32)
    validate_waveform(waveform, int(sample_rate))
    return AudioData(
        waveform=np.asarray(waveform, dtype=np.float32),
        sample_rate=int(sample_rate),
        channel_count=channel_count,
    )


def load_and_normalize_audio(path: Path, target_sample_rate: int = 16_000) -> AudioData:
    """Decode audio, average channels, and resample only in memory for runtime use.

    The source file is never altered.  No denoising, gain adjustment, or other
    speech enhancement is applied; only mono conversion and the runtime's
    required sample-rate conversion are performed.
    """

    if not isinstance(target_sample_rate, (int, np.integer)) or target_sample_rate <= 0:
        raise AudioValidationError(f"Invalid target audio sample rate: {target_sample_rate!r}")
    decoded = load_wav_read_only(path)
    waveform = decoded.waveform
    if decoded.sample_rate != int(target_sample_rate):
        try:
            waveform = librosa.resample(
                waveform,
                orig_sr=decoded.sample_rate,
                target_sr=int(target_sample_rate),
                res_type="kaiser_best",
            )
        except Exception as error:
            raise AudioValidationError("Audio sample-rate conversion failed.") from error
    normalized = np.asarray(waveform, dtype=np.float32)
    validate_waveform(normalized, int(target_sample_rate))
    return AudioData(
        waveform=normalized,
        sample_rate=int(target_sample_rate),
        channel_count=decoded.channel_count,
    )
