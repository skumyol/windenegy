"""Unit tests for evaluation metrics."""

from __future__ import annotations

import pytest

from windenegy.application.evaluation import (
    build_metrics,
    interval_coverage,
    mean_absolute_error,
    root_mean_squared_error,
    symmetric_mape,
)


def test_core_metrics() -> None:
    """Metric functions return expected values for simple arrays."""
    actual = [100.0, 200.0, 300.0]
    predicted = [110.0, 190.0, 330.0]

    assert mean_absolute_error(actual, predicted) == pytest.approx(16.666, rel=1e-3)
    assert root_mean_squared_error(actual, predicted) == pytest.approx(19.149, rel=1e-3)
    assert symmetric_mape(actual, predicted) > 0.0


def test_interval_coverage() -> None:
    """Coverage is the share of actuals inside the interval."""
    assert interval_coverage([10.0, 20.0], [9.0, 25.0], [11.0, 30.0]) == 0.5


def test_build_metrics_includes_skill_score() -> None:
    """Metrics include skill score when a baseline is supplied."""
    metrics = build_metrics(
        model_id="model",
        horizon_hours=1,
        actual=[100.0, 200.0],
        predicted=[100.0, 190.0],
        baseline_predicted=[80.0, 170.0],
    )

    assert metrics.mae == 5.0
    assert metrics.skill_score is not None
    assert metrics.skill_score > 0.0
