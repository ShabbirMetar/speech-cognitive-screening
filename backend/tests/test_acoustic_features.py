"""Synthetic-only tests for the initial librosa acoustic pilot."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from ml.audio.acoustic_features import (
    MFCC_FEATURE_NAMES,
    AcousticFeatureConfig,
    extract_acoustic_features,
    extract_acoustic_feature_result,
    pitch_statistics,
    silence_and_pause_features,
)
from ml.audio.audio_loader import AudioValidationError, load_wav_read_only, validate_waveform


SAMPLE_RATE = 16_000


def sine_wave(seconds: float = 1.0, frequency: float = 180.0) -> np.ndarray:
    samples = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * frequency * samples)).astype(np.float32)


def test_read_only_loader_reads_synthetic_wav_and_preserves_sample_rate(tmp_path) -> None:
    path = tmp_path / "synthetic.wav"
    sf.write(path, sine_wave(), SAMPLE_RATE)
    audio = load_wav_read_only(path)
    assert audio.sample_rate == SAMPLE_RATE
    assert audio.channel_count == 1
    assert audio.waveform.ndim == 1
    assert len(audio.waveform) == SAMPLE_RATE


def test_sine_wave_duration_rms_and_mfcc_summary_shape() -> None:
    features = extract_acoustic_features(sine_wave(), SAMPLE_RATE)
    assert features["audio_duration_seconds"] == pytest.approx(1.0)
    assert features["rms_mean"] > 0
    assert len(MFCC_FEATURE_NAMES) == 26
    assert all(name in features for name in MFCC_FEATURE_NAMES)


def test_silence_and_internal_pause_features() -> None:
    waveform = np.concatenate((sine_wave(0.4), np.zeros(int(0.5 * SAMPLE_RATE)), sine_wave(0.4)))
    features = silence_and_pause_features(waveform, SAMPLE_RATE, AcousticFeatureConfig())
    assert features["number_of_non_silent_segments"] >= 2
    assert features["pause_count"] >= 1
    assert features["pause_total_seconds"] >= 0.25


def test_continuous_sine_has_no_internal_pause() -> None:
    features = silence_and_pause_features(sine_wave(), SAMPLE_RATE, AcousticFeatureConfig())
    assert features["pause_count"] == 0
    assert features["pause_mean_seconds"] == 0
    assert features["pause_max_seconds"] == 0


def test_silent_waveform_has_nan_safe_pitch_and_full_silence() -> None:
    waveform = np.zeros(SAMPLE_RATE, dtype=np.float32)
    result = extract_acoustic_feature_result(waveform, SAMPLE_RATE)
    assert result.features["silence_ratio"] == pytest.approx(1.0)
    assert result.features["number_of_non_silent_segments"] == 0
    assert np.isnan(result.features["f0_mean_hz"])
    pitch, available = pitch_statistics(np.array([np.nan, np.nan]))
    assert not available
    assert all(np.isnan(value) for value in pitch.values())


def test_invalid_empty_and_non_finite_audio_is_rejected() -> None:
    with pytest.raises(AudioValidationError, match="empty"):
        validate_waveform(np.array([], dtype=np.float32), SAMPLE_RATE)
    with pytest.raises(AudioValidationError, match="non-finite"):
        validate_waveform(np.array([0.0, np.nan], dtype=np.float32), SAMPLE_RATE)
    with pytest.raises(AudioValidationError, match="too short"):
        extract_acoustic_features(np.array([0.0], dtype=np.float32), SAMPLE_RATE)
