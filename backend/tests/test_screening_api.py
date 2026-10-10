"""Synthetic-audio tests for the production runtime, without PROCESS-2 data."""

from __future__ import annotations

from io import BytesIO
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from fastapi.testclient import TestClient


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import create_app
from app.services import screening_service
from app.services.screening_service import AudioInputError, ScreeningService
from ml.asr.asr_linguistic_features import ASR_COMPATIBLE_FEATURE_NAMES, PFT_FEATURE_NAMES
from ml.asr.whisper_transcriber import WhisperTranscription, WhisperTranscriptionConfig
from ml.audio.acoustic_features import AcousticFeatureResult
from ml.models.explainability import feature_group


PURE_CTD_ACOUSTIC = (
    "audio_duration_seconds",
    "voiced_duration_seconds",
    "silence_duration_seconds",
    "silence_ratio",
    "pause_count",
    "pause_total_seconds",
    "pause_mean_seconds",
    "pause_max_seconds",
    "f0_mean_hz",
    "f0_std_hz",
    "f0_range_hz",
    "rms_mean",
    "rms_std",
)


def _frozen_schema() -> dict[str, object]:
    ordered = [
        f"{task}_{feature}"
        for feature in ASR_COMPATIBLE_FEATURE_NAMES
        for task in ("ctd", "pft", "sft")
    ]
    ordered.extend(f"pft_{feature}" for feature in PFT_FEATURE_NAMES)
    ordered.extend(
        f"{task}_{feature}"
        for task in ("sft", "pft", "ctd")
        for feature in ("recording_word_rate_wpm", "articulation_rate_wpm")
    )
    ordered.extend(f"ctd_{feature}" for feature in PURE_CTD_ACOUSTIC)
    assert len(ordered) == 51 and len(set(ordered)) == 51
    return {
        "schema_version": "1.0.0",
        "feature_count": 51,
        "ordered_feature_names": ordered,
        "feature_groups": {feature: feature_group(feature) for feature in ordered},
    }


def _synthetic_wav_bytes() -> bytes:
    sample_rate = 8_000
    samples = np.arange(sample_rate, dtype=np.float32) / sample_rate
    mono = 0.2 * np.sin(2 * np.pi * 220.0 * samples)
    stereo = np.column_stack([mono, mono])
    destination = BytesIO()
    sf.write(destination, stereo, sample_rate, format="WAV", subtype="PCM_16")
    return destination.getvalue()


def _fake_acoustic_result() -> AcousticFeatureResult:
    return AcousticFeatureResult(
        features={
            "audio_duration_seconds": 1.0,
            "voiced_duration_seconds": 0.8,
            "silence_duration_seconds": 0.2,
            "silence_ratio": 0.2,
            "pause_count": 1.0,
            "pause_total_seconds": 0.3,
            "pause_mean_seconds": 0.3,
            "pause_max_seconds": 0.3,
            "f0_mean_hz": 180.0,
            "f0_std_hz": 12.0,
            "f0_range_hz": 40.0,
            "rms_mean": 0.1,
            "rms_std": 0.01,
        },
        pitch_available=True,
        pause_detection_success=True,
    )


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> tuple[ScreeningService, dict[str, Any]]:
    schema = _frozen_schema()
    metadata = {
        "model_type": "Logistic Regression",
        "model_version": "1.0.0",
        "feature_count": 51,
        "feature_representation": "ASR_RATE_PLUS_CTD",
        "research_threshold": 0.53,
    }
    artifacts = SimpleNamespace(schema=schema, metadata=metadata)
    captured: dict[str, Any] = {"temporary_directories": []}

    def fake_transcribe(_model: object, path: Path, _config: WhisperTranscriptionConfig) -> WhisperTranscription:
        captured["temporary_directories"].append(path.parent)
        info = sf.info(path)
        captured.setdefault("normalized_audio", []).append((info.samplerate, info.channels))
        return WhisperTranscription(
            transcript_text="parrot panda pear pause",
            detected_language="en",
            transcription_runtime_seconds=0.01,
            audio_duration_seconds=1.0,
        )

    def fake_predict(feature_table: pd.DataFrame, _artifacts: object) -> list[dict[str, object]]:
        captured["feature_table"] = feature_table.copy()
        return [
            {
                "raw_model_score": 0.53,
                "cognitive_speech_screening_score": 53.0,
                "classification": "Possible impairment-like speech pattern",
                "research_threshold": 0.53,
            }
        ]

    def fake_attributions(feature_table: pd.DataFrame, _artifacts: object) -> SimpleNamespace:
        count = feature_table.shape[1]
        return SimpleNamespace(
            feature_names=tuple(feature_table.columns),
            observed_values=feature_table.to_numpy(dtype=float),
            train_standardization_means=np.zeros(count, dtype=float),
            log_odds_contributions=np.linspace(-0.51, 0.51, count, dtype=float).reshape(1, -1),
        )

    monkeypatch.setattr(screening_service, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(screening_service, "extract_acoustic_feature_result", lambda *_: _fake_acoustic_result())
    monkeypatch.setattr(screening_service, "predict_screening", fake_predict)
    monkeypatch.setattr(screening_service, "linear_model_attributions", fake_attributions)
    service = ScreeningService(
        artifacts=artifacts,
        whisper_config=WhisperTranscriptionConfig(model_size="base.en"),
        whisper_model=object(),
        max_upload_bytes=100_000,
        max_duration_seconds=5.0,
        target_sample_rate=16_000,
    )
    return service, captured


def test_service_assembles_the_exact_schema_and_cleans_temporary_audio(runtime) -> None:
    service, captured = runtime
    output = service.analyze_uploads(
        "synthetic-request",
        {"SFT": _synthetic_wav_bytes(), "PFT": _synthetic_wav_bytes(), "CTD": _synthetic_wav_bytes()},
    )

    feature_table = captured["feature_table"]
    assert feature_table.shape == (1, 51)
    assert list(feature_table.columns) == service._artifacts.schema["ordered_feature_names"]
    assert output["result"]["classification"] == "Possible impairment-like speech pattern"
    assert output["result"]["raw_model_score"] == 0.53
    assert output["result"]["cognitive_speech_screening_score"] == 53.0
    assert output["tasks"]["ctd"]["word_count"] == 4
    assert output["explanation"]["scale"] == "log-odds"
    assert output["explanation"]["toward_impairment_like"]
    assert output["explanation"]["toward_healthy_like"]
    assert captured["normalized_audio"] == [(16_000, 1)] * 3
    assert all(not directory.exists() for directory in captured["temporary_directories"])


def test_service_rejects_corrupt_audio_without_processing_it(runtime) -> None:
    service, _ = runtime
    with pytest.raises(AudioInputError, match="could not be decoded"):
        service.analyze_uploads(
            "corrupt-request",
            {"SFT": b"not audio", "PFT": _synthetic_wav_bytes(), "CTD": _synthetic_wav_bytes()},
        )


def test_service_rejects_oversized_and_overlong_uploads(runtime) -> None:
    service, _ = runtime
    with pytest.raises(AudioInputError, match="upload limit"):
        service.analyze_uploads(
            "oversized-request",
            {"SFT": b"x" * 100_001, "PFT": _synthetic_wav_bytes(), "CTD": _synthetic_wav_bytes()},
        )
    service._max_duration_seconds = 0.5
    with pytest.raises(AudioInputError, match="duration limit"):
        service.analyze_uploads(
            "overlong-request",
            {"SFT": _synthetic_wav_bytes(), "PFT": _synthetic_wav_bytes(), "CTD": _synthetic_wav_bytes()},
        )


def test_screening_api_validates_uploads_and_uses_research_terminology(runtime) -> None:
    service, _ = runtime
    app = create_app(lambda: service)
    files = {
        "sft_audio": ("ignored.wav", _synthetic_wav_bytes(), "audio/wav"),
        "pft_audio": ("ignored.wav", _synthetic_wav_bytes(), "audio/wav"),
        "ctd_audio": ("ignored.wav", _synthetic_wav_bytes(), "audio/wav"),
    }
    with TestClient(app) as client:
        info = client.get("/api/v1/screening/model-info")
        openapi = client.get("/openapi.json")
        response = client.post("/api/v1/screening/analyze", files=files)
        missing = client.post("/api/v1/screening/analyze", files={"sft_audio": files["sft_audio"]})
        corrupt = client.post(
            "/api/v1/screening/analyze",
            files={**files, "ctd_audio": ("ignored.wav", b"bad audio", "audio/wav")},
        )

    assert info.status_code == 200
    assert info.json()["feature_count"] == 51
    assert openapi.status_code == 200
    assert "/api/v1/screening/analyze" in openapi.json()["paths"]
    assert response.status_code == 200
    payload = response.json()
    assert 0.0 <= payload["result"]["raw_model_score"] <= 1.0
    assert 0.0 <= payload["result"]["cognitive_speech_screening_score"] <= 100.0
    assert payload["result"]["classification"] == "Possible impairment-like speech pattern"
    assert "not a medical diagnosis" in payload["disclaimer"]
    assert missing.status_code == 400
    assert corrupt.status_code == 422


def test_runtime_has_no_model_fit_or_dataset_runtime_dependency() -> None:
    source = inspect.getsource(screening_service)
    assert ".fit(" not in source
    for forbidden in ("development_features", "official_test", "get_dataset_root", "process2_dataset_path"):
        assert forbidden not in source
