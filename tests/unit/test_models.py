"""Unit tests for domain models."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from windenegy.domain.models import (
    ForecastHorizon,
    ForecastPoint,
    PowerForecast,
    PowerUnit,
    TurbineObservation,
)


class TestTurbineObservation:
    """Tests for TurbineObservation model."""

    def test_valid_observation(self) -> None:
        """Test creating a valid observation."""
        obs = TurbineObservation(
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            active_power_kw=1000.0,
            wind_speed_mps=7.5,
            wind_direction_deg=180.0,
            theoretical_power_kwh=950.0,
        )
        assert obs.active_power_kw == 1000.0
        assert obs.wind_speed_mps == 7.5

    def test_timestamp_string_parsing(self) -> None:
        """Test parsing timestamp from ISO string."""
        obs = TurbineObservation(
            timestamp="2024-01-01T12:00:00Z",
            active_power_kw=1000.0,
            wind_speed_mps=7.5,
            wind_direction_deg=180.0,
            theoretical_power_kwh=950.0,
        )
        assert obs.timestamp.year == 2024
        assert obs.timestamp.tzinfo is not None

    def test_negative_power_rejected(self) -> None:
        """Test that negative power is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TurbineObservation(
                timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
                active_power_kw=-100.0,
                wind_speed_mps=7.5,
                wind_direction_deg=180.0,
                theoretical_power_kwh=950.0,
            )
        assert "active_power_kw" in str(exc_info.value)

    def test_wind_direction_range(self) -> None:
        """Test wind direction must be in [0, 360)."""
        # Valid at boundaries
        TurbineObservation(
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            active_power_kw=1000.0,
            wind_speed_mps=7.5,
            wind_direction_deg=0.0,
            theoretical_power_kwh=950.0,
        )

        # Invalid - too high
        with pytest.raises(ValidationError):
            TurbineObservation(
                timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
                active_power_kw=1000.0,
                wind_speed_mps=7.5,
                wind_direction_deg=360.0,
                theoretical_power_kwh=950.0,
            )

    def test_model_is_immutable(self) -> None:
        """Test that models are frozen (immutable)."""
        obs = TurbineObservation(
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            active_power_kw=1000.0,
            wind_speed_mps=7.5,
            wind_direction_deg=180.0,
            theoretical_power_kwh=950.0,
        )
        with pytest.raises(ValidationError):
            obs.active_power_kw = 500.0  # type: ignore[misc]


class TestForecastPoint:
    """Tests for ForecastPoint model."""

    def test_valid_forecast_point(self) -> None:
        """Test creating a valid forecast point."""
        point = ForecastPoint(
            timestamp=datetime(2024, 1, 1, 13, 0, 0, tzinfo=UTC),
            p50=1000.0,
            p10=800.0,
            p90=1200.0,
        )
        assert point.p50 == 1000.0
        assert point.p10 == 800.0
        assert point.p90 == 1200.0

    def test_forecast_point_rejects_nan(self) -> None:
        """Test that NaN values are rejected."""
        with pytest.raises(ValidationError):
            ForecastPoint(
                timestamp=datetime(2024, 1, 1, 13, 0, 0, tzinfo=UTC),
                p50=float("nan"),
                p10=800.0,
                p90=1200.0,
            )

    def test_forecast_point_rejects_non_numeric(self) -> None:
        """Test that non-numeric values are rejected."""
        with pytest.raises(ValidationError):
            ForecastPoint(
                timestamp=datetime(2024, 1, 1, 13, 0, 0, tzinfo=UTC),
                p50="invalid",  # type: ignore[arg-type]
                p10=800.0,
                p90=1200.0,
            )


class TestPowerForecast:
    """Tests for PowerForecast model."""

    def test_valid_forecast(self) -> None:
        """Test creating a valid power forecast."""
        points = [
            ForecastPoint(
                timestamp=datetime(2024, 1, 1, 13, 0, 0, tzinfo=UTC),
                p50=1000.0,
                p10=800.0,
                p90=1200.0,
            ),
            ForecastPoint(
                timestamp=datetime(2024, 1, 1, 14, 0, 0, tzinfo=UTC),
                p50=1050.0,
                p10=850.0,
                p90=1250.0,
            ),
        ]

        forecast = PowerForecast(
            asset_id="T1",
            model_version="gbm-2024-01-01",
            horizon_hours=ForecastHorizon.HOUR_1,
            forecast=points,
        )

        assert forecast.asset_id == "T1"
        assert len(forecast.forecast) == 2
        assert forecast.unit == PowerUnit.KILOWATT

    def test_empty_forecast_rejected(self) -> None:
        """Test that empty forecast list is rejected."""
        with pytest.raises(ValidationError):
            PowerForecast(
                asset_id="T1",
                model_version="gbm-2024-01-01",
                horizon_hours=ForecastHorizon.HOUR_1,
                forecast=[],
            )

    def test_to_dict_serializes(self) -> None:
        """Test serialization to dictionary."""
        points = [
            ForecastPoint(
                timestamp=datetime(2024, 1, 1, 13, 0, 0, tzinfo=UTC),
                p50=1000.0,
                p10=800.0,
                p90=1200.0,
            ),
        ]

        forecast = PowerForecast(
            asset_id="T1",
            model_version="gbm-2024-01-01",
            horizon_hours=ForecastHorizon.HOUR_1,
            forecast=points,
        )

        data = forecast.to_dict()
        assert data["asset_id"] == "T1"
        assert data["unit"] == "kW"
        assert len(data["forecast"]) == 1


class TestForecastHorizon:
    """Tests for ForecastHorizon enum."""

    def test_horizon_values(self) -> None:
        """Test horizon enum values."""
        assert ForecastHorizon.HOUR_1.value == 1
        assert ForecastHorizon.HOUR_6.value == 6
        assert ForecastHorizon.HOUR_24.value == 24
