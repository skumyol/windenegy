"""Baseline forecasting models.

Simple baselines that anchor model credibility:
- Persistence: future power equals last observed power
- Power Curve: lookup from manufacturer theoretical curve
- Rolling Mean: recent average power
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
from structlog import get_logger

from windenegy.domain.models import ForecastHorizon, ForecastPoint, PowerForecast, PowerUnit

logger = get_logger(__name__)


@dataclass
class PersistenceBaseline:
    """Persistence baseline model.

    The simplest baseline: forecast future power equals
    the most recently observed power.

    Attributes:
        residual_p90: Estimated P90 residual from validation data.
    """

    residual_p90: float = 200.0  # Default estimate
    _model_version: str = "persistence-baseline-0.1.0"

    def fit(self, df: pl.DataFrame) -> PersistenceBaseline:
        """Calculate residual from training data.

        Args:
            df: Training dataframe with 'active_power_kw' column.

        Returns:
            Self for chaining.
        """
        if "active_power_kw" not in df.columns:
            logger.warning("No power column found, using default residual")
            return self

        # Calculate typical power change magnitude
        power = df["active_power_kw"].to_numpy()
        if len(power) > 1:
            diffs = np.abs(np.diff(power))
            # P90 of absolute changes as uncertainty estimate
            self.residual_p90 = float(np.percentile(diffs, 90)) * 2
            logger.info("Calculated persistence residual", residual_p90=self.residual_p90)

        return self

    def predict(
        self,
        last_power: float,
        horizon_steps: int,
        base_time: datetime,
        step_minutes: int = 10,
    ) -> list[ForecastPoint]:
        """Generate forecast points.

        Args:
            last_power: Last observed power in kW.
            horizon_steps: Number of steps to forecast.
            base_time: Starting timestamp for forecast.
            step_minutes: Time between forecast points.

        Returns:
            List of forecast points.
        """
        points: list[ForecastPoint] = []

        for i in range(1, horizon_steps + 1):
            point_time = base_time + timedelta(minutes=step_minutes * i)
            points.append(
                ForecastPoint(
                    timestamp=point_time,
                    p50=round(last_power, 3),
                    p10=round(max(0.0, last_power - self.residual_p90), 3),
                    p90=round(last_power + self.residual_p90, 3),
                )
            )

        return points

    def predict_from_observations(
        self,
        observations: list[Any],
        horizon_hours: int = 6,
    ) -> PowerForecast:
        """Generate forecast from observations.

        Args:
            observations: List of TurbineObservation objects.
            horizon_hours: Forecast horizon in hours.

        Returns:
            Power forecast.
        """
        if not observations:
            raise ValueError("At least one observation required")

        # Sort by timestamp and get last
        sorted_obs = sorted(observations, key=lambda x: x.timestamp)
        last_obs = sorted_obs[-1]

        # Calculate horizon steps (10-minute data)
        horizon_steps = horizon_hours * 6

        points = self.predict(
            last_power=last_obs.active_power_kw,
            horizon_steps=horizon_steps,
            base_time=last_obs.timestamp,
        )

        return PowerForecast(
            asset_id="T1",
            model_version=self._model_version,
            horizon_hours=ForecastHorizon(horizon_hours),
            unit=PowerUnit.KILOWATT,
            created_at=datetime.now(UTC),
            forecast=points,
            warnings=[],
        )


@dataclass
class PowerCurveBaseline:
    """Power curve baseline using manufacturer theoretical curve.

    Looks up expected power from wind speed using the
    theoretical power curve provided in the dataset.
    """

    wind_speed_col: str = "wind_speed_mps"
    power_curve_col: str = "theoretical_power_kwh"
    _model_version: str = "power-curve-baseline-0.1.0"

    def predict(
        self,
        wind_speed: float,
        theoretical_power: float,
    ) -> float:
        """Predict power from wind speed and theoretical curve.

        Args:
            wind_speed: Current wind speed in m/s.
            theoretical_power: Theoretical power from curve.

        Returns:
            Predicted power in kW.
        """
        # For baseline, use theoretical power directly
        # In practice, this could be adjusted by observed vs theoretical ratio
        return theoretical_power

    def evaluate(self, df: pl.DataFrame) -> dict[str, float]:
        """Evaluate power curve against actual observations.

        Args:
            df: Dataframe with actual and theoretical power.

        Returns:
            Dictionary with evaluation metrics.
        """
        if self.wind_speed_col not in df.columns or "active_power_kw" not in df.columns:
            return {"error": "Required columns not found"}

        actual = df["active_power_kw"].to_numpy()
        theoretical = df[self.power_curve_col].to_numpy()

        # Calculate metrics
        mae = float(np.mean(np.abs(actual - theoretical)))
        rmse = float(np.sqrt(np.mean((actual - theoretical) ** 2)))

        # Capacity factor comparison
        actual_mean = float(np.mean(actual))
        theoretical_mean = float(np.mean(theoretical))

        return {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "actual_mean": round(actual_mean, 2),
            "theoretical_mean": round(theoretical_mean, 2),
            "capacity_ratio": round(actual_mean / theoretical_mean, 3) if theoretical_mean > 0 else 0,
        }


@dataclass
class RollingMeanBaseline:
    """Rolling mean baseline.

    Forecast is the average of recent observations.
    """

    window_hours: int = 1  # Default 1-hour rolling window
    _model_version: str = "rolling-mean-baseline-0.1.0"

    def predict(
        self,
        observations: list[Any],
        horizon_steps: int,
        base_time: datetime,
        step_minutes: int = 10,
    ) -> list[ForecastPoint]:
        """Generate forecast from rolling mean.

        Args:
            observations: Recent observations.
            horizon_steps: Number of steps to forecast.
            base_time: Starting timestamp.
            step_minutes: Time between points.

        Returns:
            List of forecast points.
        """
        if not observations:
            raise ValueError("At least one observation required")

        # Calculate rolling mean from recent observations
        window_size = self.window_hours * 6  # 10-minute steps
        recent_powers = [obs.active_power_kw for obs in observations[-window_size:]]
        mean_power = float(np.mean(recent_powers)) if recent_powers else 0.0

        # Estimate residual based on variance
        if len(recent_powers) > 1:
            std = float(np.std(recent_powers))
            residual_p90 = std * 1.645  # ~90% for normal distribution
        else:
            residual_p90 = 200.0

        points: list[ForecastPoint] = []
        for i in range(1, horizon_steps + 1):
            point_time = base_time + timedelta(minutes=step_minutes * i)
            points.append(
                ForecastPoint(
                    timestamp=point_time,
                    p50=round(mean_power, 3),
                    p10=round(max(0.0, mean_power - residual_p90), 3),
                    p90=round(mean_power + residual_p90, 3),
                )
            )

        return points
