"""Pydantic response contracts for the research screening API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ScreeningClass = Literal[
    "Healthy-like speech pattern",
    "Possible impairment-like speech pattern",
]
ContributionDirection = Literal["toward_possible_impairment_like", "toward_healthy_like"]


class ScreeningResult(BaseModel):
    """Frozen-model output expressed as a research screening signal."""

    classification: ScreeningClass
    cognitive_speech_screening_score: float = Field(ge=0.0, le=100.0)
    raw_model_score: float = Field(ge=0.0, le=1.0)
    research_threshold: float = Field(ge=0.0, le=1.0)


class TaskTranscript(BaseModel):
    transcript: str
    duration_seconds: float = Field(gt=0.0)
    word_count: int = Field(ge=0)
    transcription_runtime_seconds: float = Field(ge=0.0)


class TaskTranscripts(BaseModel):
    sft: TaskTranscript
    pft: TaskTranscript
    ctd: TaskTranscript


class ObservedSpeechCharacteristic(BaseModel):
    feature_key: str
    label: str
    observed_value: float | None
    unit: str


class ExplanationContribution(BaseModel):
    feature_key: str
    label: str
    observed_value: float | None
    train_reference_mean: float | None
    was_median_imputed: bool
    log_odds_contribution: float
    contribution_magnitude: float = Field(ge=0.0)
    direction: ContributionDirection
    explanation: str


class ScreeningExplanation(BaseModel):
    method: str
    scale: Literal["log-odds"]
    toward_impairment_like: list[ExplanationContribution]
    toward_healthy_like: list[ExplanationContribution]


class ModelInfo(BaseModel):
    model_type: str
    model_version: str | None = None
    asr: str
    feature_representation: str
    feature_count: Literal[51]
    research_threshold: float = Field(ge=0.0, le=1.0)
    disclaimer: str


class ScreeningAnalysisResponse(BaseModel):
    request_id: str
    result: ScreeningResult
    tasks: TaskTranscripts
    observed_biomarkers: list[ObservedSpeechCharacteristic]
    explanation: ScreeningExplanation
    model: ModelInfo
    disclaimer: str
