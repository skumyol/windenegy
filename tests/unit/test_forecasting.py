"""Unit tests for forecasting services."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from windenegy.application.forecasting import ForecastService, detect_ramp_events
from windenegy.domain.models import ForecastHorizon, TurbineObservation


def _observations(count: int = 7) -> list[TurbineObservation]:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    return [
        TurbineObservation(
            timestamp=base + timedelta(minutes=10 * index),
            active_power_kw=500.0 + index * 10.0,
            wind_speed_mps=6.0,
            wind_direction_deg=120.0,
            theoretical_power_kwh=550.0,
        )
        for index in range(count)
    ]


def test_persistence_forecast_uses_sampling_cadence() -> None:
    """A 1-hour forecast on 10-minute data returns six points."""
    forecast = ForecastService().forecast("T1", _observations(), ForecastHorizon.HOUR_1)

    assert len(forecast.forecast) == 6
    assert forecast.forecast[0].timestamp == _observations()[-1].timestamp + timedelta(minutes=10)
    assert forecast.forecast[0].p10 <= forecast.forecast[0].p50 <= forecast.forecast[0].p90


def test_forecast_rejects_unsupported_horizon() -> None:
    """Unsupported horizons are rejected before response serialization."""
    with pytest.raises(ValueError, match="Unsupported horizon"):
        ForecastService().forecast("T1", _observations(), 2)


def test_detect_ramp_events() -> None:
    """Large adjacent power changes are detected."""
    observations = _observations(2)
    observations.append(
        TurbineObservation(
            timestamp=observations[-1].timestamp + timedelta(minutes=10),
            active_power_kw=1400.0,
            wind_speed_mps=10.0,
            wind_direction_deg=120.0,
            theoretical_power_kwh=1450.0,
        )
    )

    events = detect_ramp_events(observations, min_change_kw=500.0)

    assert len(events) == 1
    assert events[0][2] > 0.0
