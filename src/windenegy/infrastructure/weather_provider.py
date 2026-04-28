"""Open-Meteo weather data provider.

Provides historical reanalysis and forecast data from Open-Meteo's free API.
No API key required. Provides wind speed at multiple heights including hub height (100m).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from structlog import get_logger

from windenegy.domain.models import WeatherObservation
from windenegy.application.weather import WeatherProvider

logger = get_logger(__name__)

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class OpenMeteoProvider(WeatherProvider):
    """Open-Meteo weather data provider.

    Fetches historical and forecast weather data including wind at multiple heights.
    Free tier: 10,000 requests/day, no API key required.
    """

    def __init__(
        self,
        latitude: float,
        longitude: float,
        altitude: float | None = None,
    ) -> None:
        """Initialize provider for a location.

        Args:
            latitude: Location latitude.
            longitude: Location longitude.
            altitude: Turbine hub height in meters (for wind extrapolation).
        """
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude

    def fetch(
        self,
        start: datetime,
        end: datetime,
        latitude: float,
        longitude: float,
    ) -> list[WeatherObservation]:
        """Fetch weather data for a time range.

        Args:
            start: Start time (UTC).
            end: End time (UTC).
            latitude: Location latitude (ignored, uses instance value).
            longitude: Location longitude (ignored, uses instance value).

        Returns:
            List of weather observations.
        """
        return self._fetch_historical(start, end)

    def _fetch_historical(
        self,
        start: datetime,
        end: datetime,
    ) -> list[WeatherObservation]:
        """Fetch historical weather data from Open-Meteo archive."""
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "pressure_msl",
                "wind_speed_10m",
                "wind_direction_10m",
                "wind_speed_100m",
                "wind_direction_100m",
                "wind_speed_120m",
                "wind_direction_120m",
            ],
            "timezone": "UTC",
        }

        logger.info("Fetching historical weather", params=params)

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(OPEN_METEO_ARCHIVE_URL, params=params)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as e:
            logger.error("Open-Meteo request failed", error=str(e))
            return []

        return self._parse_hourly_data(data)

    def fetch_forecast(
        self,
        hours_ahead: int = 48,
    ) -> list[WeatherObservation]:
        """Fetch weather forecast from Open-Meteo.

        Args:
            hours_ahead: Number of hours to forecast (max 48 for free tier).

        Returns:
            List of weather forecast observations.
        """
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "pressure_msl",
                "wind_speed_10m",
                "wind_direction_10m",
                "wind_speed_100m",
                "wind_direction_100m",
                "wind_speed_120m",
                "wind_direction_120m",
            ],
            "forecast_hours": hours_ahead,
            "timezone": "UTC",
        }

        logger.info("Fetching forecast", hours=hours_ahead)

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(OPEN_METEO_FORECAST_URL, params=params)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as e:
            logger.error("Open-Meteo forecast request failed", error=str(e))
            return []

        return self._parse_hourly_data(data)

    def _parse_hourly_data(self, data: dict[str, Any]) -> list[WeatherObservation]:
        """Parse Open-Meteo hourly response into WeatherObservations."""
        observations = []

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])

        for i, timestamp_str in enumerate(times):
            try:
                ts = timestamp_str.replace("Z", "+00:00")
                if "+" not in ts and "-" in ts and not ts.endswith("+00:00"):
                    ts = ts + "+00:00"
                timestamp = datetime.fromisoformat(ts)
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=UTC)
            except (ValueError, TypeError):
                continue

            obs = WeatherObservation(
                timestamp=timestamp,
                wind_speed_10m_mps=self._safe_get(hourly, "wind_speed_10m", i),
                wind_speed_100m_mps=self._safe_get(hourly, "wind_speed_100m", i),
                wind_direction_deg=self._safe_get(hourly, "wind_direction_10m", i),
                temperature_c=self._safe_get(hourly, "temperature_2m", i),
                pressure_hpa=self._safe_get(hourly, "pressure_msl", i),
            )
            observations.append(obs)

        logger.info("Parsed weather observations", count=len(observations))
        return observations

    def _safe_get(self, hourly: dict[str, list], key: str, index: int) -> float | None:
        """Safely get a value from hourly data array."""
        values = hourly.get(key)
        if values is None or index >= len(values):
            return None
        val = values[index]
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None


def create_provider(
    latitude: float,
    longitude: float,
    provider_type: str = "openmeteo",
    **kwargs: Any,
) -> WeatherProvider:
    """Factory function to create a weather provider.

    Args:
        latitude: Location latitude.
        longitude: Location longitude.
        provider_type: Type of provider ("openmeteo", "null").
        **kwargs: Additional provider configuration.

    Returns:
        Configured weather provider.
    """
    if provider_type == "openmeteo":
        return OpenMeteoProvider(
            latitude=latitude,
            longitude=longitude,
            altitude=kwargs.get("altitude"),
        )
    from windenegy.application.weather import NullWeatherProvider

    return NullWeatherProvider()
