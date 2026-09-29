"""Small, interpretable librosa-based acoustic features for the pilot."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from ml.audio.audio_loader import AudioValidationError, validate_waveform


ACOUSTIC_EXTRACTOR_VERSION = "librosa-acoustic-pilot-v1"


@dataclass(frozen=True)
class AcousticFeatureConfig:
    """Engineering parameters for the initial acoustic pilot, not clinical settings."""

    frame_length: int = 2048
    hop_length: int = 512
    mfcc_count: int = 13
    silence_top_db: float = 30.0
    minimum_pause_seconds: float = 0.25
    f0_min_hz: float = 50.0
    f0_max_hz: float = 500.0


@dataclass(frozen=True)
class AcousticFeatureResult:
    """Features plus non-predictive extraction-status information."""

    features: dict[str, float]
    pitch_available: bool
    pause_detection_success: bool


MFCC_FEATURE_NAMES = tuple(
    name
    for index in range(1, 14)
    for name in (f"mfcc_{index:02d}_mean", f"mfcc_{index:02d}_std")
)
ACOUSTIC_FEATURE_NAMES = (
    "audio_duration_seconds",
    "rms_mean",
    "rms_std",
    "zero_crossing_rate_mean",
    "zero_crossing_rate_std",
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",
    *MFCC_FEATURE_NAMES,
    "f0_mean_hz",
    "f0_std_hz",
    "f0_min_hz",
    "f0_max_hz",
    "f0_range_hz",
    "voiced_duration_seconds",
    "silence_duration_seconds",
    "silence_ratio",
    "number_of_non_silent_segments",
    "pause_count",
    "pause_total_seconds",
    "pause_mean_seconds",
    "pause_max_seconds",
    "recording_word_rate_wpm",
    "articulation_rate_wpm",
)


def empty_acoustic_features() -> dict[str, float]:
    """Return a missing-value row for an unavailable recording."""

    return {feature_name: float("nan") for feature_name in ACOUSTIC_FEATURE_NAMES}


def _mean_and_std(values: np.ndarray) -> tuple[float, float]:
    flattened = np.asarray(values, dtype=float).ravel()
    return float(np.mean(flattened)), float(np.std(flattened, ddof=0))


def pitch_statistics(frequencies: np.ndarray) -> tuple[dict[str, float], bool]:
    """Summarize finite voiced F0 values, preserving missing pitch as NaN."""

    valid_frequencies = np.asarray(frequencies, dtype=float)
    valid_frequencies = valid_frequencies[np.isfinite(valid_frequencies) & (valid_frequencies > 0)]
    names = ("f0_mean_hz", "f0_std_hz", "f0_min_hz", "f0_max_hz", "f0_range_hz")
    if valid_frequencies.size == 0:
        return {name: float("nan") for name in names}, False
    minimum = float(np.min(valid_frequencies))
    maximum = float(np.max(valid_frequencies))
    return {
        "f0_mean_hz": float(np.mean(valid_frequencies)),
        "f0_std_hz": float(np.std(valid_frequencies, ddof=0)),
        "f0_min_hz": minimum,
        "f0_max_hz": maximum,
        "f0_range_hz": maximum - minimum,
    }, True


def silence_and_pause_features(
    waveform: np.ndarray, sample_rate: int, config: AcousticFeatureConfig
) -> dict[str, float]:
    """Estimate energy-based silence and internal pauses for pilot diagnostics.

    ``silence_top_db`` and ``minimum_pause_seconds`` are engineering parameters;
    this first baseline is not a clinically validated voice-activity detector.
    """

    duration = len(waveform) / sample_rate
    if np.max(np.abs(waveform)) <= np.finfo(np.float32).eps:
        intervals = np.empty((0, 2), dtype=int)
    else:
        intervals = librosa.effects.split(
            waveform,
            top_db=config.silence_top_db,
            frame_length=config.frame_length,
            hop_length=config.hop_length,
        )
    voiced_samples = int(np.sum(intervals[:, 1] - intervals[:, 0])) if len(intervals) else 0
    voiced_duration = voiced_samples / sample_rate
    pause_durations = [
        (next_start - previous_end) / sample_rate
        for (_, previous_end), (next_start, _) in zip(intervals[:-1], intervals[1:], strict=True)
        if (next_start - previous_end) / sample_rate >= config.minimum_pause_seconds
    ]
    return {
        "voiced_duration_seconds": float(voiced_duration),
        "silence_duration_seconds": float(max(duration - voiced_duration, 0.0)),
        "silence_ratio": float(max(duration - voiced_duration, 0.0) / duration) if duration > 0 else float("nan"),
        "number_of_non_silent_segments": float(len(intervals)),
        "pause_count": float(len(pause_durations)),
        "pause_total_seconds": float(sum(pause_durations)),
        "pause_mean_seconds": float(np.mean(pause_durations)) if pause_durations else 0.0,
        "pause_max_seconds": float(np.max(pause_durations)) if pause_durations else 0.0,
    }


def speech_rate_features(
    word_count: object, audio_duration_seconds: float, voiced_duration_seconds: float
) -> dict[str, float]:
    """Calculate transcript-derived rates when the required inputs are valid."""

    try:
        words = float(word_count)
    except (TypeError, ValueError):
        words = float("nan")
    valid_words = bool(np.isfinite(words) and words >= 0)
    return {
        "recording_word_rate_wpm": (
            words / audio_duration_seconds * 60
            if valid_words and np.isfinite(audio_duration_seconds) and audio_duration_seconds > 0
            else float("nan")
        ),
        "articulation_rate_wpm": (
            words / voiced_duration_seconds * 60
            if valid_words and np.isfinite(voiced_duration_seconds) and voiced_duration_seconds > 0
            else float("nan")
        ),
    }


def extract_acoustic_feature_result(
    waveform: np.ndarray, sample_rate: int, config: AcousticFeatureConfig | None = None
) -> AcousticFeatureResult:
    """Extract all non-rate acoustic features from a validated in-memory waveform."""

    active_config = config or AcousticFeatureConfig()
    values = np.asarray(waveform, dtype=np.float32).ravel()
    validate_waveform(values, sample_rate)
    n_fft = min(active_config.frame_length, len(values))
    if n_fft < 2:
        raise AudioValidationError("Audio waveform is too short for spectral feature extraction.")
    hop_length = min(active_config.hop_length, max(1, n_fft // 2))
    features: dict[str, float] = {"audio_duration_seconds": float(len(values) / sample_rate)}

    rms = librosa.feature.rms(y=values, frame_length=n_fft, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(values, frame_length=n_fft, hop_length=hop_length)
    centroid = librosa.feature.spectral_centroid(y=values, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
    bandwidth = librosa.feature.spectral_bandwidth(y=values, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
    for prefix, values_for_feature in (
        ("rms", rms),
        ("zero_crossing_rate", zcr),
        ("spectral_centroid", centroid),
        ("spectral_bandwidth", bandwidth),
    ):
        mean, std = _mean_and_std(values_for_feature)
        features[f"{prefix}_mean"] = mean
        features[f"{prefix}_std"] = std

    mfcc = librosa.feature.mfcc(
        y=values,
        sr=sample_rate,
        n_mfcc=active_config.mfcc_count,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    for index, coefficient in enumerate(mfcc, start=1):
        mean, std = _mean_and_std(coefficient)
        features[f"mfcc_{index:02d}_mean"] = mean
        features[f"mfcc_{index:02d}_std"] = std

    try:
        f0, _, _ = librosa.pyin(
            values,
            fmin=active_config.f0_min_hz,
            fmax=active_config.f0_max_hz,
            sr=sample_rate,
            frame_length=n_fft,
            hop_length=hop_length,
        )
        pitch_features, pitch_available = pitch_statistics(f0)
    except Exception:  # librosa can reject short/unvoiced signals; retain missing pitch.
        pitch_features, pitch_available = pitch_statistics(np.array([], dtype=float))
    features.update(pitch_features)

    try:
        features.update(silence_and_pause_features(values, sample_rate, active_config))
        pause_detection_success = True
    except Exception:
        features.update(
            {
                "voiced_duration_seconds": float("nan"),
                "silence_duration_seconds": float("nan"),
                "silence_ratio": float("nan"),
                "number_of_non_silent_segments": float("nan"),
                "pause_count": float("nan"),
                "pause_total_seconds": float("nan"),
                "pause_mean_seconds": float("nan"),
                "pause_max_seconds": float("nan"),
            }
        )
        pause_detection_success = False
    return AcousticFeatureResult(
        features=features,
        pitch_available=pitch_available,
        pause_detection_success=pause_detection_success,
    )


def extract_acoustic_features(
    waveform: np.ndarray, sample_rate: int, config: AcousticFeatureConfig | None = None
) -> dict[str, float]:
    """Return the pilot acoustic features without extraction-status fields."""

    return extract_acoustic_feature_result(waveform, sample_rate, config).features
