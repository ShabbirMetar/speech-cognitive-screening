"""Synthetic safeguards for TRAIN-only screening threshold analysis."""

from __future__ import annotations

import numpy as np

from scripts.analyze_screening_threshold import (
    THRESHOLDS,
    build_threshold_sweep,
    select_threshold_candidates,
    threshold_metrics,
)


def test_threshold_sweep_has_the_requested_inclusive_range() -> None:
    target = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])
    sweep = build_threshold_sweep(target, probabilities)
    assert len(sweep) == 61
    assert sweep["threshold"].iloc[0] == THRESHOLDS[0] == 0.20
    assert sweep["threshold"].iloc[-1] == THRESHOLDS[-1] == 0.80


def test_threshold_metrics_uses_fixed_half_threshold_when_requested() -> None:
    metrics = threshold_metrics(np.array([0, 0, 1, 1]), np.array([0.2, 0.6, 0.4, 0.9]), 0.5)
    assert metrics["sensitivity"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["youden_j"] == 0.0


def test_candidate_selection_includes_objective_and_available_sensitivity_rules() -> None:
    sweep = build_threshold_sweep(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9]))
    candidates = select_threshold_candidates(sweep)
    assert {"max_f1", "max_balanced_accuracy", "max_youden_j"}.issubset(candidates["rule"])
    assert candidates["rule"].str.startswith("sensitivity_at_least_").any()
