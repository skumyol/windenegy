"""Forecast evaluation metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from windenegy.domain.models import ModelMetrics


def mean_absolute_error(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Calculate mean absolute error."""
    actual_arr, predicted_arr = _as_arrays(actual, predicted)
    return float(np.mean(np.abs(actual_arr - predicted_arr)))


def root_mean_squared_error(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Calculate root mean squared error."""
    actual_arr, predicted_arr = _as_arrays(actual, predicted)
    return float(math.sqrt(float(np.mean(np.square(actual_arr - predicted_arr)))))


def symmetric_mape(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Calculate symmetric mean absolute percentage error."""
    actual_arr, predicted_arr = _as_arrays(actual, predicted)
    denominator = np.abs(actual_arr) + np.abs(predicted_arr)
    mask = denominator > 0
    if not bool(mask.any()):
        return 0.0
    percentage_error = 2.0 * np.abs(predicted_arr[mask] - actual_arr[mask])
    return float(np.mean(percentage_error / denominator[mask]) * 100.0)


def skill_score(model_mae: float, baseline_mae: float) -> float | None:
    """Calculate skill score versus a baseline MAE."""
    if baseline_mae <= 0:
        return None
    return 1.0 - model_mae / baseline_mae


def interval_coverage(
    actual: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> float:
    """Calculate empirical interval coverage."""
    actual_arr, lower_arr = _as_arrays(actual, lower)
    _, upper_arr = _as_arrays(actual, upper)
    return float(np.mean((actual_arr >= lower_arr) & (actual_arr <= upper_arr)))


def build_metrics(
    model_id: str,
    horizon_hours: int,
    actual: Sequence[float],
    predicted: Sequence[float],
    baseline_predicted: Sequence[float] | None = None,
    lower: Sequence[float] | None = None,
    upper: Sequence[float] | None = None,
) -> ModelMetrics:
    """Build a domain metrics object from raw arrays."""
    mae = mean_absolute_error(actual, predicted)
    baseline_mae = (
        mean_absolute_error(actual, baseline_predicted) if baseline_predicted is not None else 0.0
    )
    coverage = (
        interval_coverage(actual, lower, upper) if lower is not None and upper is not None else None
    )
    return ModelMetrics(
        model_id=model_id,
        horizon_hours=horizon_hours,
        mae=mae,
        rmse=root_mean_squared_error(actual, predicted),
        mape=symmetric_mape(actual, predicted),
        skill_score=skill_score(mae, baseline_mae),
        coverage_p90=coverage,
    )


def _as_arrays(left: Sequence[float], right: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Convert two equally-sized sequences to finite float arrays."""
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    if left_arr.shape != right_arr.shape:
        msg = (
            f"Metric inputs must have identical shapes, got {left_arr.shape} and {right_arr.shape}"
        )
        raise ValueError(msg)
    if not np.isfinite(left_arr).all() or not np.isfinite(right_arr).all():
        msg = "Metric inputs must be finite"
        raise ValueError(msg)
    return left_arr, right_arr
