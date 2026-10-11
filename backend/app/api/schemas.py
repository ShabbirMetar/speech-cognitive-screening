"""Pydantic response contracts for the research screening API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ScreeningClass = Literal[
    "Healthy-like speech pattern",
    "Possible impairment-like speech pattern",
]
ContributionDirection = Literal["toward_possible_impairment_like", "toward_healthy_like"]
ModelDirection = Literal[
    "higher_values_toward_possible_impairment_like",
    "higher_values_toward_healthy_like",
    "weak_or_negligible_linear_effect",
]


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


class FeatureReferenceStatistics(BaseModel):
    """Aggregate-only descriptive statistics from one TRAIN class group."""

    valid_n: int = Field(ge=0, le=160)
    mean: float | None = None
    median: float | None = None
    q25: float | None = None
    q75: float | None = None


class FeatureProfileEntry(BaseModel):
    """One current feature value plus aggregate-only TRAIN reference metadata."""

    feature_key: str
    label: str
    task: Literal["SFT", "PFT", "CTD"]
    group: Literal["ASR_LINGUISTIC", "ASR_RATE", "CTD_ACOUSTIC"]
    unit: str
    raw_value: float | None = None
    model_input_value: float | None = None
    was_median_imputed: bool
    standardized_value: float
    healthy_reference: FeatureReferenceStatistics
    impaired_reference: FeatureReferenceStatistics
    healthy_reference_standardized: float
    impaired_reference_standardized: float
    coefficient: float
    model_direction: ModelDirection
    local_log_odds_contribution: float


class FeatureProfile(BaseModel):
    """Optional, aggregate-only visualization data that never affects prediction."""

    available: Literal[True] = True
    source_description: str
    source_participants: Literal[320]
    healthy_train_participants: Literal[160]
    impaired_train_participants: Literal[160]
    official_test_participants_used: Literal[0]
    standardized_value_method: str
    features: list[FeatureProfileEntry] = Field(min_length=51, max_length=51)


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
    feature_profile: FeatureProfile | None = None
    explanation: ScreeningExplanation
    model: ModelInfo
    disclaimer: str
