"""Runtime-only inference service for the frozen speech-screening pipeline.

This module has no PROCESS-2 dataset dependency. It processes a single set of
transient uploaded recordings, uses frozen artifacts and ASR settings, and
never fits or persists user speech data.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import pandas as pd
import soundfile as sf

from ml.asr.asr_linguistic_features import extract_asr_linguistic_features
from ml.asr.whisper_transcriber import (
    WhisperTranscriptionConfig,
    load_whisper_model,
    transcribe_audio,
)
from ml.audio.acoustic_features import extract_acoustic_feature_result, speech_rate_features
from ml.audio.audio_loader import AudioData, AudioValidationError, load_and_normalize_audio
from ml.models.explainability import format_feature_contribution, human_feature_label
from ml.models.frozen_pipeline import (
    FrozenScreeningArtifacts,
    linear_model_attributions,
    load_frozen_artifacts,
    predict_screening,
    validate_and_order_features,
)

from ..core.config import Settings, get_settings


LOGGER = logging.getLogger(__name__)
TASKS = ("SFT", "PFT", "CTD")
PURE_CTD_ACOUSTIC_SUFFIXES = (
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


class ScreeningRuntimeError(RuntimeError):
    """Base class for errors whose messages are safe to return to clients."""


class ModelRuntimeUnavailable(ScreeningRuntimeError):
    """Raised when frozen artifacts or the frozen ASR runtime cannot initialize."""


class AudioInputError(ScreeningRuntimeError):
    """Raised for empty, oversized, unreadable, or unreasonable audio uploads."""


class TranscriptionError(ScreeningRuntimeError):
    """Raised when the frozen ASR model cannot transcribe a task recording."""


class FeatureExtractionError(ScreeningRuntimeError):
    """Raised when the frozen feature pathway cannot produce an inference vector."""


@dataclass(frozen=True)
class TaskProcessingResult:
    """Transient task values used to form one prediction response."""

    task: str
    transcript: str
    duration_seconds: float
    transcription_runtime_seconds: float
    word_count: int
    linguistic_features: dict[str, float | int]
    rate_features: dict[str, float]
    acoustic_features: dict[str, float]


def _finite_or_none(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _frozen_asr_config(artifacts: FrozenScreeningArtifacts) -> WhisperTranscriptionConfig:
    """Create the exact ASR config from frozen model metadata only."""

    source = artifacts.metadata.get("asr_deployment_configuration")
    if not isinstance(source, dict):
        raise ModelRuntimeUnavailable("Frozen ASR configuration is unavailable.")
    required = {"model_size", "device", "compute_type", "language", "cpu_threads", "num_workers"}
    if required.difference(source):
        raise ModelRuntimeUnavailable("Frozen ASR configuration is incomplete.")
    config = WhisperTranscriptionConfig(
        model_size=str(source["model_size"]),
        device=str(source["device"]),
        compute_type=str(source["compute_type"]),
        language=str(source["language"]),
        cpu_threads=int(source["cpu_threads"]),
        num_workers=int(source["num_workers"]),
    )
    expected = ("base.en", "cpu", "int8", "en", 4, 1)
    observed = (
        config.model_size,
        config.device,
        config.compute_type,
        config.language,
        config.cpu_threads,
        config.num_workers,
    )
    if observed != expected:
        raise ModelRuntimeUnavailable("Frozen ASR configuration does not match the approved deployment settings.")
    return config


class ScreeningService:
    """One-at-a-time local runtime for frozen audio-to-screening inference."""

    def __init__(
        self,
        artifacts: FrozenScreeningArtifacts,
        whisper_config: WhisperTranscriptionConfig,
        whisper_model: Any,
        *,
        max_upload_bytes: int,
        max_duration_seconds: float,
        target_sample_rate: int,
    ) -> None:
        self._artifacts = artifacts
        self._whisper_config = whisper_config
        self._whisper_model = whisper_model
        self._max_upload_bytes = int(max_upload_bytes)
        self._max_duration_seconds = float(max_duration_seconds)
        self._target_sample_rate = int(target_sample_rate)
        self._inference_lock = Lock()

    @classmethod
    def load_default(cls, settings: Settings | None = None) -> "ScreeningService":
        """Load and validate all heavy frozen runtime assets exactly once."""

        active_settings = settings or get_settings()
        try:
            artifacts = load_frozen_artifacts()
            config = _frozen_asr_config(artifacts)
            model = load_whisper_model(config)
        except Exception as error:
            LOGGER.exception("Frozen screening runtime initialization failed: %s", error.__class__.__name__)
            raise ModelRuntimeUnavailable("Frozen screening runtime could not be initialized.") from error
        return cls(
            artifacts,
            config,
            model,
            max_upload_bytes=active_settings.screening_max_upload_bytes,
            max_duration_seconds=active_settings.screening_max_duration_seconds,
            target_sample_rate=active_settings.screening_target_sample_rate,
        )

    @property
    def is_ready(self) -> bool:
        """Indicate that frozen artifacts and one ASR model instance are loaded."""

        return self._whisper_model is not None

    @property
    def max_upload_bytes(self) -> int:
        """Return the request-byte cap used by the API before temporary writes."""

        return self._max_upload_bytes

    def model_info(self) -> dict[str, object]:
        """Return non-sensitive frozen metadata suitable for the public API."""

        metadata = self._artifacts.metadata
        return {
            "model_type": str(metadata.get("model_type", "Logistic Regression")),
            "model_version": str(metadata.get("model_version", "1.0.0")),
            "feature_representation": str(metadata.get("feature_representation", "ASR_RATE_PLUS_CTD")),
            "feature_count": int(metadata.get("feature_count", 51)),
            "asr": f"faster-whisper {self._whisper_config.model_size}",
            "research_threshold": float(metadata.get("research_threshold", 0.53)),
            "disclaimer": "Research screening prototype; this result is not a medical diagnosis.",
        }

    def analyze_uploads(self, request_id: str, uploads: Mapping[str, bytes]) -> dict[str, object]:
        """Process exactly SFT, PFT, and CTD audio bytes without retaining them."""

        normalized_uploads = {str(name).upper(): value for name, value in uploads.items()}
        if set(normalized_uploads) != set(TASKS):
            raise AudioInputError("Exactly one SFT, PFT, and CTD recording is required.")
        started_at = perf_counter()
        # The shared CPU ASR model is serialized for this demo hardware.
        with self._inference_lock:
            try:
                with TemporaryDirectory(prefix="speech-screening-") as directory_name:
                    temporary_directory = Path(directory_name)
                    processed = {
                        task: self._process_task(task, normalized_uploads[task], temporary_directory)
                        for task in TASKS
                    }
                    feature_frame = self._assemble_frozen_feature_frame(processed)
                    prediction = predict_screening(feature_frame, self._artifacts)[0]
                    explanation = self._build_explanation(feature_frame)
                    response = {
                        "request_id": request_id,
                        "result": {
                            "classification": prediction["classification"],
                            "cognitive_speech_screening_score": prediction["cognitive_speech_screening_score"],
                            "raw_model_score": prediction["raw_model_score"],
                            "research_threshold": prediction["research_threshold"],
                        },
                        "tasks": {
                            task.casefold(): {
                                "transcript": item.transcript,
                                "duration_seconds": item.duration_seconds,
                                "transcription_runtime_seconds": item.transcription_runtime_seconds,
                                "word_count": item.word_count,
                            }
                            for task, item in processed.items()
                        },
                        "observed_biomarkers": self._observed_speech_characteristics(feature_frame.iloc[0]),
                        "explanation": explanation,
                        "model": self.model_info(),
                        "disclaimer": "Research screening prototype; this result is not a medical diagnosis.",
                    }
            except ScreeningRuntimeError:
                raise
            except Exception as error:
                LOGGER.warning(
                    "screening_request_failed request_id=%s stage=processing category=%s",
                    request_id,
                    error.__class__.__name__,
                )
                raise FeatureExtractionError("Speech features could not be extracted from the uploaded recordings.") from error
        LOGGER.info("Screening request %s completed in %.2f seconds.", request_id, perf_counter() - started_at)
        return response

    def _process_task(self, task: str, audio_bytes: bytes, temporary_directory: Path) -> TaskProcessingResult:
        """Decode, normalize, transcribe, and extract existing features for one task."""

        if not isinstance(audio_bytes, bytes) or not audio_bytes:
            raise AudioInputError(f"{task} audio is empty.")
        if len(audio_bytes) > self._max_upload_bytes:
            raise AudioInputError(f"{task} audio exceeds the server upload limit.")
        source_path = temporary_directory / f"{task.casefold()}_upload.wav"
        normalized_path = temporary_directory / f"{task.casefold()}_normalized.wav"
        try:
            source_path.write_bytes(audio_bytes)
            audio = load_and_normalize_audio(source_path, self._target_sample_rate)
        except (AudioValidationError, OSError, RuntimeError, ValueError) as error:
            raise AudioInputError(f"{task} audio could not be decoded as supported audio.") from error
        duration = len(audio.waveform) / audio.sample_rate
        if duration <= 0.0:
            raise AudioInputError(f"{task} audio is empty.")
        if duration > self._max_duration_seconds:
            raise AudioInputError(f"{task} audio exceeds the server duration limit.")
        try:
            sf.write(normalized_path, audio.waveform, audio.sample_rate, format="WAV", subtype="FLOAT")
        except (OSError, RuntimeError, ValueError) as error:
            raise AudioInputError(f"{task} audio could not be prepared for processing.") from error
        return self._extract_task_features(task, audio, normalized_path)

    def _extract_task_features(self, task: str, audio: AudioData, normalized_path: Path) -> TaskProcessingResult:
        """Apply frozen ASR and existing linguistic/acoustic helpers once."""

        try:
            transcription = transcribe_audio(self._whisper_model, normalized_path, self._whisper_config)
        except Exception as error:
            LOGGER.warning("ASR failure for task %s: %s", task, error.__class__.__name__)
            raise TranscriptionError(f"Speech transcription failed for the {task} recording.") from error
        if not transcription.transcript_text.strip():
            raise TranscriptionError(f"The {task} recording did not produce recognized speech.")
        try:
            linguistic = extract_asr_linguistic_features(transcription.transcript_text, task)
            acoustic_result = extract_acoustic_feature_result(audio.waveform, audio.sample_rate)
            acoustic = dict(acoustic_result.features)
            rates = speech_rate_features(
                linguistic["word_count"],
                acoustic["audio_duration_seconds"],
                acoustic["voiced_duration_seconds"],
            )
        except Exception as error:
            LOGGER.warning("Feature extraction failure for task %s: %s", task, error.__class__.__name__)
            raise FeatureExtractionError(f"Speech features could not be extracted for the {task} recording.") from error
        return TaskProcessingResult(
            task=task,
            transcript=transcription.transcript_text,
            duration_seconds=float(acoustic["audio_duration_seconds"]),
            transcription_runtime_seconds=float(transcription.transcription_runtime_seconds),
            word_count=int(linguistic["word_count"]),
            linguistic_features=linguistic,
            rate_features=rates,
            acoustic_features=acoustic,
        )

    def _assemble_frozen_feature_frame(self, processed: Mapping[str, TaskProcessingResult]) -> pd.DataFrame:
        """Build the exact 32 + 6 + 13 feature set, then apply schema order."""

        values: dict[str, float | int] = {}
        for task in TASKS:
            item = processed[task]
            prefix = task.casefold()
            for name, value in item.linguistic_features.items():
                values[f"{prefix}_{name}"] = value
            for name, value in item.rate_features.items():
                values[f"{prefix}_{name}"] = value
        ctd_acoustic = processed["CTD"].acoustic_features
        for suffix in PURE_CTD_ACOUSTIC_SUFFIXES:
            values[f"ctd_{suffix}"] = ctd_acoustic[suffix]
        candidate = pd.DataFrame([values])
        try:
            ordered = validate_and_order_features(candidate, self._artifacts.schema)
        except (TypeError, ValueError) as error:
            raise FeatureExtractionError("Generated speech features do not match the frozen model schema.") from error
        if ordered.shape != (1, 51):
            raise FeatureExtractionError("Generated speech features do not contain the required 51 predictors.")
        return ordered

    def _build_explanation(self, feature_frame: pd.DataFrame) -> dict[str, object]:
        """Provide exact linear log-odds attributions without retaining TRAIN rows."""

        try:
            attribution = linear_model_attributions(feature_frame, self._artifacts)
        except Exception as error:
            raise FeatureExtractionError("Model explanation could not be generated safely.") from error
        contributions = attribution.log_odds_contributions[0]
        return {
            "scale": "log-odds",
            "method": "exact frozen linear-model contributions relative to the zero-centred TRAIN standardized reference",
            "toward_impairment_like": self._contribution_rows(attribution, contributions, positive=True),
            "toward_healthy_like": self._contribution_rows(attribution, contributions, positive=False),
        }

    @staticmethod
    def _contribution_rows(attribution: Any, contributions: np.ndarray, *, positive: bool) -> list[dict[str, object]]:
        """Select the largest signed exact contributions for one response section."""

        if positive:
            indices = [index for index, value in enumerate(contributions) if float(value) > 0.0]
            direction = "toward_possible_impairment_like"
        else:
            indices = [index for index, value in enumerate(contributions) if float(value) < 0.0]
            direction = "toward_healthy_like"
        selected = sorted(indices, key=lambda index: abs(float(contributions[index])), reverse=True)[:5]
        rows: list[dict[str, object]] = []
        for index in selected:
            observed = float(attribution.observed_values[0, index])
            reference = float(attribution.train_standardization_means[index])
            value = float(contributions[index])
            rows.append(
                {
                    "feature_key": attribution.feature_names[index],
                    "label": human_feature_label(attribution.feature_names[index]),
                    "observed_value": _finite_or_none(observed),
                    "train_reference_mean": _finite_or_none(reference),
                    "was_median_imputed": not np.isfinite(observed),
                    "direction": direction,
                    "log_odds_contribution": value,
                    "contribution_magnitude": abs(value),
                    "explanation": format_feature_contribution(
                        attribution.feature_names[index], value, observed, reference
                    ),
                }
            )
        return rows

    @staticmethod
    def _observed_speech_characteristics(feature_row: pd.Series) -> list[dict[str, object]]:
        """Expose a small, readable set of observed speech measurements."""

        selected = (
            ("sft_type_token_ratio", "ratio"),
            ("pft_type_token_ratio", "ratio"),
            ("ctd_type_token_ratio", "ratio"),
            ("sft_recording_word_rate_wpm", "words per minute"),
            ("pft_recording_word_rate_wpm", "words per minute"),
            ("ctd_recording_word_rate_wpm", "words per minute"),
            ("ctd_pause_total_seconds", "seconds"),
            ("ctd_silence_ratio", "ratio"),
            ("ctd_f0_std_hz", "hertz"),
            ("ctd_rms_mean", "normalized RMS"),
        )
        return [
            {
                "feature_key": feature,
                "label": human_feature_label(feature),
                "observed_value": _finite_or_none(feature_row[feature]),
                "unit": unit,
            }
            for feature, unit in selected
        ]
