"""Forecasting application services."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from statistics import pstdev

from windenegy.domain.models import (
    ForecastHorizon,
    ForecastPoint,
    PowerForecast,
    PowerUnit,
    TurbineObservation,
)


@dataclass(frozen=True)
class ForecastService:
    """Baseline forecasting service shared by API, dashboard, and tests."""

    model_version: str = "persistence-baseline-0.1.0"
    rated_power_kw: float | None = None
    default_step_minutes: int = 10

    def forecast(
        self,
        asset_id: str,
        observations: list[TurbineObservation],
        horizon_hours: int | ForecastHorizon,
        created_at: datetime | None = None,
    ) -> PowerForecast:
        """Create a persistence forecast with uncertainty bands."""
        if not observations:
            msg = "At least one observation is required"
            raise ValueError(msg)

        horizon = self._coerce_horizon(horizon_hours)
        sorted_observations = sorted(observations, key=lambda item: item.timestamp)
        last_observation = sorted_observations[-1]
        step = self._infer_step(sorted_observations)
        forecast_count = max(1, int((horizon.value * 60) / (step.total_seconds() / 60.0)))
        spread = self._estimate_spread(sorted_observations)

        points = []
        for index in range(1, forecast_count + 1):
            timestamp = last_observation.timestamp + step * index
            p50 = self._clip_power(last_observation.active_power_kw)
            points.append(
                ForecastPoint(
                    timestamp=timestamp,
                    p50=p50,
                    p10=self._clip_power(p50 - spread),
                    p90=self._clip_power(p50 + spread),
                )
            )

        warnings = []
        if len(sorted_observations) < 3:
            warnings.append("Very short observation history; uncertainty band is approximate.")

        return PowerForecast(
            asset_id=asset_id,
            model_version=self.model_version,
            horizon_hours=horizon,
            unit=PowerUnit.KILOWATT,
            created_at=created_at or datetime.now(UTC),
            forecast=points,
            warnings=warnings,
        )

    def _coerce_horizon(self, horizon_hours: int | ForecastHorizon) -> ForecastHorizon:
        """Convert a supported integer horizon into a domain enum."""
        if isinstance(horizon_hours, ForecastHorizon):
            return horizon_hours
        try:
            return ForecastHorizon(horizon_hours)
        except ValueError as exc:
            msg = (
                f"Unsupported horizon: {horizon_hours}. Supported horizons are 1, 6, and 24 hours."
            )
            raise ValueError(msg) from exc

    def _infer_step(self, observations: list[TurbineObservation]) -> timedelta:
        """Infer sampling cadence from observations."""
        if len(observations) < 2:
            return timedelta(minutes=self.default_step_minutes)

        deltas = [
            (right.timestamp - left.timestamp).total_seconds()
            for left, right in pairwise(observations)
            if right.timestamp > left.timestamp
        ]
        if not deltas:
            return timedelta(minutes=self.default_step_minutes)

        median_seconds = sorted(deltas)[len(deltas) // 2]
        return timedelta(seconds=median_seconds)

    def _estimate_spread(self, observations: list[TurbineObservation]) -> float:
        """Estimate an operational uncertainty spread from recent variability."""
        powers = [item.active_power_kw for item in observations[-18:]]
        variability = pstdev(powers) if len(powers) >= 2 else 0.0
        last_power = observations[-1].active_power_kw
        conservative_floor = max(50.0, last_power * 0.1)
        return max(conservative_floor, variability * 1.28)

    def _clip_power(self, value: float) -> float:
        """Clip power to physical output bounds when rated power is known."""
        clipped = max(0.0, value)
        if self.rated_power_kw is not None:
            clipped = min(clipped, self.rated_power_kw)
        return round(clipped, 3)


def detect_ramp_events(
    observations: list[TurbineObservation],
    min_change_kw: float = 500.0,
    max_window_minutes: float = 60.0,
) -> list[tuple[datetime, datetime, float]]:
    """Detect large power changes over adjacent observations."""
    sorted_observations = sorted(observations, key=lambda item: item.timestamp)
    events: list[tuple[datetime, datetime, float]] = []
    for left, right in pairwise(sorted_observations):
        duration = (right.timestamp - left.timestamp).total_seconds() / 60.0
        change = right.active_power_kw - left.active_power_kw
        if 0 < duration <= max_window_minutes and abs(change) >= min_change_kw:
            events.append((left.timestamp, right.timestamp, change))
    return events
