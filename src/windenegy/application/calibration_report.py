"""Calibration reports for uncertainty quantification.

Generates reports measuring interval coverage, sharpness, and calibration.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from structlog import get_logger

from windenegy.application.uncertainty import ConformalPredictor

if TYPE_CHECKING:
    from windenegy.domain.sequence import SequenceSample

logger = get_logger(__name__)


@dataclass(frozen=True)
class CalibrationMetrics:
    """Calibration quality metrics.

    Attributes:
        target_coverage: Target coverage level (e.g., 0.80).
        empirical_coverage: Achieved coverage on test set.
        mean_interval_width: Average prediction interval width.
        median_interval_width: Median interval width.
        mean_absolute_error: MAE of point forecasts.
        rmse: RMSE of point forecasts.
        sharpness: 1 / mean_width (higher is sharper).
        is_well_calibrated: Whether coverage matches target within tolerance.
    """

    target_coverage: float
    empirical_coverage: float
    mean_interval_width: float
    median_interval_width: float
    mean_absolute_error: float
    rmse: float
    sharpness: float
    is_well_calibrated: bool

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class HorizonCalibrationReport:
    """Calibration report for a single forecast horizon.

    Attributes:
        horizon_hours: Forecast horizon.
        n_samples: Number of test samples.
        metrics: Calibration metrics.
        coverage_by_power_bin: Coverage in different power regimes.
    """

    horizon_hours: int
    n_samples: int
    metrics: CalibrationMetrics
    coverage_by_power_bin: dict[str, float]


@dataclass(frozen=True)
class ModelCalibrationReport:
    """Full calibration report for a model across horizons.

    Attributes:
        model_id: Model identifier.
        model_type: Type of model (e.g., "patchtst", "gradient_boosting").
        generated_at: ISO timestamp of report generation.
        horizon_reports: Reports by horizon.
        summary: Overall summary statistics.
    """

    model_id: str
    model_type: str
    generated_at: str
    horizon_reports: dict[str, HorizonCalibrationReport]
    summary: dict[str, float]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "generated_at": self.generated_at,
            "horizon_reports": {
                k: {
                    "horizon_hours": v.horizon_hours,
                    "n_samples": v.n_samples,
                    "metrics": v.metrics.to_dict(),
                    "coverage_by_power_bin": v.coverage_by_power_bin,
                }
                for k, v in self.horizon_reports.items()
            },
            "summary": self.summary,
        }

    def save(self, path: Path) -> None:
        """Save report to JSON."""
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Calibration report saved", path=str(path))


class CalibrationReporter:
    """Generate calibration reports for models with uncertainty."""

    def __init__(self, coverage_target: float = 0.80, tolerance: float = 0.05) -> None:
        """Initialize reporter.

        Args:
            coverage_target: Target coverage level.
            tolerance: Acceptable deviation from target.
        """
        self.coverage_target = coverage_target
        self.tolerance = tolerance

    def generate_report(
        self,
        model_id: str,
        model_type: str,
        model,
        val_samples: list[SequenceSample],
        test_samples: list[SequenceSample],
        horizon_hours: int = 1,
    ) -> HorizonCalibrationReport:
        """Generate calibration report for a model.

        Args:
            model_id: Model identifier.
            model_type: Type of model.
            model: Model with predict() method.
            val_samples: Validation samples for conformal calibration.
            test_samples: Test samples for evaluation.
            horizon_hours: Forecast horizon.

        Returns:
            Calibration report for this horizon.
        """
        # Calibrate on validation set
        val_predictions = []
        val_actuals = []
        for sample in val_samples:
            pred = model.predict(sample.input_sequence)
            if hasattr(pred, "__len__") and len(pred) > 1:
                pred = pred[0]
            val_predictions.append(float(pred))
            val_actuals.append(
                float(sample.target[0] if hasattr(sample.target, "__len__") else sample.target)
            )

        predictor = ConformalPredictor(coverage=self.coverage_target)
        predictor.fit(np.array(val_predictions), np.array(val_actuals))

        # Evaluate on test set
        test_predictions = []
        test_actuals = []
        for sample in test_samples:
            pred = model.predict(sample.input_sequence)
            if hasattr(pred, "__len__") and len(pred) > 1:
                pred = pred[0]
            test_predictions.append(float(pred))
            test_actuals.append(
                float(sample.target[0] if hasattr(sample.target, "__len__") else sample.target)
            )

        test_predictions = np.array(test_predictions)
        test_actuals = np.array(test_actuals)

        # Calculate coverage metrics
        intervals = predictor.predict_interval(test_predictions)
        covered = sum(
            1 for i, interval in enumerate(intervals)
            if interval.p10 <= test_actuals[i] <= interval.p90
        )
        empirical_coverage = covered / len(test_actuals)

        widths = [interval.width for interval in intervals]
        mean_width = float(np.mean(widths))

        # Calculate point forecast errors
        mae = float(np.mean(np.abs(test_predictions - test_actuals)))
        rmse = float(np.sqrt(np.mean((test_predictions - test_actuals) ** 2)))

        # Coverage by power bin
        power_bins = self._bin_by_power(test_actuals, test_predictions, intervals)

        metrics = CalibrationMetrics(
            target_coverage=self.coverage_target,
            empirical_coverage=empirical_coverage,
            mean_interval_width=mean_width,
            median_interval_width=float(np.median(widths)),
            mean_absolute_error=mae,
            rmse=rmse,
            sharpness=1.0 / mean_width if mean_width > 0 else 0,
            is_well_calibrated=abs(empirical_coverage - self.coverage_target) <= self.tolerance,
        )

        report = HorizonCalibrationReport(
            horizon_hours=horizon_hours,
            n_samples=len(test_actuals),
            metrics=metrics,
            coverage_by_power_bin=power_bins,
        )

        logger.info(
            "Calibration report generated",
            model_id=model_id,
            horizon=horizon_hours,
            empirical_coverage=empirical_coverage,
            mean_width=mean_width,
        )

        return report

    def _bin_by_power(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray,
        intervals: list,
    ) -> dict[str, float]:
        """Calculate coverage by power regime."""
        # Define bins
        bins = [
            (0, 500, "low_power"),
            (500, 1500, "medium_power"),
            (1500, 3000, "high_power"),
        ]

        coverage_by_bin: dict[str, float] = {}

        for low, high, name in bins:
            mask = (actuals >= low) & (actuals < high)
            if mask.sum() > 0:
                covered = sum(
                    1 for i in range(len(actuals))
                    if mask[i] and intervals[i].p10 <= actuals[i] <= intervals[i].p90
                )
                coverage_by_bin[name] = covered / mask.sum()
            else:
                coverage_by_bin[name] = 1.0

        return coverage_by_bin


def generate_full_report(
    model_id: str,
    model_type: str,
    model,
    val_samples: list[SequenceSample],
    test_samples: list[SequenceSample],
    horizon_hours: int = 1,
    output_path: Path | None = None,
) -> ModelCalibrationReport:
    """Generate complete calibration report.

    Args:
        model_id: Model identifier.
        model_type: Model type.
        model: Model instance.
        val_samples: Validation samples.
        test_samples: Test samples.
        horizon_hours: Forecast horizon.
        output_path: Optional path to save report.

    Returns:
        Complete calibration report.
    """
    reporter = CalibrationReporter()

    horizon_report = reporter.generate_report(
        model_id=model_id,
        model_type=model_type,
        model=model,
        val_samples=val_samples,
        test_samples=test_samples,
        horizon_hours=horizon_hours,
    )

    summary = {
        "overall_coverage": horizon_report.metrics.empirical_coverage,
        "mean_interval_width": horizon_report.metrics.mean_interval_width,
        "sharpness": horizon_report.metrics.sharpness,
    }

    report = ModelCalibrationReport(
        model_id=model_id,
        model_type=model_type,
        generated_at=datetime.now(UTC).isoformat(),
        horizon_reports={f"{horizon_hours}h": horizon_report},
        summary=summary,
    )

    if output_path:
        report.save(output_path)

    return report
