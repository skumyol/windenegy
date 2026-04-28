"""Uncertainty quantification via conformal prediction and calibration.

Provides prediction intervals with statistical coverage guarantees
using split conformal prediction on held-out calibration data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from structlog import get_logger

if TYPE_CHECKING:
    from windenegy.domain.sequence import SequenceSample

logger = get_logger(__name__)


@dataclass(frozen=True)
class PredictionInterval:
    """A prediction interval with point forecast and bounds.

    Attributes:
        p50: Median (point) forecast
        p10: Lower bound (10th percentile)
        p90: Upper bound (90th percentile)
        width: Interval width (p90 - p10)
        coverage_target: Target coverage probability
    """

    p50: float
    p10: float
    p90: float
    width: float
    coverage_target: float = 0.80

    def __post_init__(self) -> None:
        """Validate interval bounds after initialization."""
        if self.p10 > self.p90:
            raise ValueError(f"Lower bound {self.p10} > upper bound {self.p90}")


@dataclass(frozen=True)
class CalibrationResult:
    """Results from conformal calibration.

    Attributes:
        quantile_level: Non-conformity quantile used (e.g., 0.9 for 80% coverage)
        nonconformity_scores: Scores on calibration set
        median_residual: Median absolute residual
        max_residual: Maximum absolute residual on calibration set
    """

    quantile_level: float
    nonconformity_scores: np.ndarray
    median_residual: float
    max_residual: float

    def get_interval_width(self, point_forecast: float) -> tuple[float, float]:
        """Get lower and upper bounds for a point forecast.

        Args:
            point_forecast: Point prediction value.

        Returns:
            Tuple of (lower, upper) bounds.
        """
        q = np.quantile(self.nonconformity_scores, self.quantile_level)
        return (point_forecast - q, point_forecast + q)


class ConformalPredictor:
    """Split conformal predictor for prediction intervals.

    Uses absolute residuals as non-conformity scores on a calibration set.
    """

    def __init__(self, coverage: float = 0.80) -> None:
        """Initialize with target coverage.

        Args:
            coverage: Target coverage probability (e.g., 0.80 for 80%).
        """
        if not 0 < coverage < 1:
            raise ValueError(f"Coverage must be in (0,1), got {coverage}")
        self.coverage = coverage
        self.quantile_level = coverage + (1 - coverage) / 2  # Two-sided
        self._calibration: CalibrationResult | None = None

    def fit(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> ConformalPredictor:
        """Calibrate using held-out data.

        Args:
            predictions: Model predictions on calibration set.
            actuals: Ground truth on calibration set.

        Returns:
            Self for chaining.
        """
        predictions, actuals = self._validate_inputs(predictions, actuals)
        scores = np.abs(actuals - predictions)

        self._calibration = CalibrationResult(
            quantile_level=self.quantile_level,
            nonconformity_scores=scores,
            median_residual=float(np.median(scores)),
            max_residual=float(np.max(scores)),
        )

        logger.info(
            "Calibrated conformal predictor",
            coverage=self.coverage,
            n_cal=len(scores),
            median_residual=self._calibration.median_residual,
            quantile_value=float(np.quantile(scores, self.quantile_level)),
        )
        return self

    def predict_interval(
        self,
        point_forecast: float | np.ndarray,
    ) -> PredictionInterval | list[PredictionInterval]:
        """Generate prediction interval(s).

        Args:
            point_forecast: Single value or array of point forecasts.

        Returns:
            PredictionInterval(s) with bounds.
        """
        if self._calibration is None:
            raise RuntimeError("Predictor must be calibrated before prediction")

        q = float(np.quantile(self._calibration.nonconformity_scores, self.quantile_level))

        if isinstance(point_forecast, np.ndarray):
            intervals = []
            for pf in point_forecast:
                lower = float(pf) - q
                upper = float(pf) + q
                # Ensure non-negative power and valid interval
                lower = max(0.0, lower)
                upper = max(lower + 1.0, upper)  # Ensure upper > lower
                intervals.append(
                    PredictionInterval(
                        p50=float(pf),
                        p10=lower,
                        p90=upper,
                        width=upper - lower,
                        coverage_target=self.coverage,
                    )
                )
            return intervals

        lower = float(point_forecast) - q
        upper = float(point_forecast) + q
        # Ensure non-negative power and valid interval
        lower = max(0.0, lower)
        upper = max(lower + 1.0, upper)  # Ensure upper > lower

        return PredictionInterval(
            p50=float(point_forecast),
            p10=lower,
            p90=upper,
            width=upper - lower,
            coverage_target=self.coverage,
        )

    def evaluate_coverage(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> dict[str, float]:
        """Evaluate empirical coverage on a test set.

        Args:
            predictions: Point predictions.
            actuals: Ground truth.

        Returns:
            Dictionary with coverage metrics.
        """
        predictions, actuals = self._validate_inputs(predictions, actuals)
        intervals = self.predict_interval(predictions)

        covered = sum(
            1 for i, interval in enumerate(intervals)
            if interval.p10 <= actuals[i] <= interval.p90
        )

        widths = [interval.width for interval in intervals]

        return {
            "empirical_coverage": covered / len(actuals),
            "target_coverage": self.coverage,
            "mean_width": float(np.mean(widths)),
            "median_width": float(np.median(widths)),
            "max_width": float(np.max(widths)),
        }

    @staticmethod
    def _validate_inputs(
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate and align inputs."""
        predictions = np.asarray(predictions).flatten()
        actuals = np.asarray(actuals).flatten()

        if len(predictions) != len(actuals):
            raise ValueError(
                f"Length mismatch: predictions={len(predictions)}, actuals={len(actuals)}"
            )
        if len(predictions) == 0:
            raise ValueError("Empty input arrays")

        # Filter non-finite values
        mask = np.isfinite(predictions) & np.isfinite(actuals)
        if not mask.all():
            n_bad = (~mask).sum()
            logger.warning(f"Filtering {n_bad} non-finite values")
            predictions = predictions[mask]
            actuals = actuals[mask]

        return predictions, actuals


def calibrate_from_val_samples(
    model,
    val_samples: list[SequenceSample],
    coverage: float = 0.80,
) -> ConformalPredictor:
    """Convenience: calibrate conformal predictor from validation sequences.

    Args:
        model: Model with predict() method.
        val_samples: Validation sequence samples.
        coverage: Target coverage.

    Returns:
        Calibrated ConformalPredictor.
    """
    predictions = []
    actuals = []

    for sample in val_samples:
        pred = model.predict(sample.input_sequence)
        # Handle multi-step: take first step
        if hasattr(pred, "__len__") and len(pred) > 1:
            pred = pred[0]
        predictions.append(float(pred))
        actuals.append(float(sample.target[0] if hasattr(sample.target, "__len__") else sample.target))

    predictor = ConformalPredictor(coverage=coverage)
    predictor.fit(np.array(predictions), np.array(actuals))
    return predictor
