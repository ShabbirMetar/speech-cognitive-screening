"""Unit tests for reusable, non-clinical explanation helpers."""

from __future__ import annotations

import pytest

from ml.models.explainability import (
    RESEARCH_OPERATING_THRESHOLD,
    feature_group,
    feature_task,
    feature_unit,
    format_feature_contribution,
    human_feature_label,
    model_direction,
    model_direction_text,
    screening_classification,
    screening_score,
    select_example_indices,
)


def test_feature_group_and_labels_keep_the_frozen_modalities_readable() -> None:
    assert feature_group("sft_type_token_ratio") == "ASR_LINGUISTIC"
    assert feature_group("ctd_articulation_rate_wpm") == "ASR_RATE"
    assert feature_group("ctd_pause_total_seconds") == "CTD_ACOUSTIC"
    assert human_feature_label("ctd_pause_total_seconds") == "CTD Total Pause Duration"
    assert human_feature_label("sft_type_token_ratio") == "SFT Lexical Diversity (TTR)"
    assert feature_task("ctd_pause_total_seconds") == "CTD"
    assert feature_unit("ctd_pause_total_seconds") == "seconds"
    assert feature_unit("pft_p_initial_word_count") == "words"


def test_model_direction_describes_only_the_fitted_coefficient_sign() -> None:
    assert model_direction(0.01) == "higher_values_toward_possible_impairment_like"
    assert model_direction(-0.01) == "higher_values_toward_healthy_like"
    assert model_direction(0.0) == "weak_or_negligible_linear_effect"
    assert "Possible impairment-like output" in model_direction_text(model_direction(0.01))
    assert "Healthy-like output" in model_direction_text(model_direction(-0.01))


def test_screening_score_and_research_threshold_are_display_only_helpers() -> None:
    assert screening_score(0.53) == 53.0
    assert screening_classification(RESEARCH_OPERATING_THRESHOLD) == "Possible impairment-like speech pattern"
    assert screening_classification(0.5299) == "Healthy-like speech pattern"
    with pytest.raises(ValueError, match="between 0 and 1"):
        screening_score(1.1)


def test_contribution_formatter_uses_observed_train_reference_and_shap_direction() -> None:
    positive = format_feature_contribution("ctd_pause_count", 0.4, 8.0, 4.0)
    negative = format_feature_contribution("sft_type_token_ratio", -0.3, 0.2, 0.4)
    assert "Higher CTD Pause Count" in positive
    assert "possible-impairment-like screening output" in positive
    assert "Lower SFT Lexical Diversity (TTR)" in negative
    assert "healthy-like screening output" in negative


def test_example_selection_is_label_blind_score_position_selection() -> None:
    selected = select_example_indices(["p3", "p1", "p2", "p4"], [0.10, 0.50, 0.54, 0.90])
    assert set(selected) == {"low_score_quantile", "near_research_threshold", "high_score_quantile"}
    assert len(set(selected.values())) == 3
    assert selected["near_research_threshold"] == 2
