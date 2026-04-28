"""Weather provider interfaces and deterministic fallback provider."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from windenegy.domain.models import WeatherObservation


class WeatherProvider(ABC):
    """Interface for weather observations or forecasts."""

    @abstractmethod
    def fetch(
        self,
        start: datetime,
        end: datetime,
        latitude: float,
        longitude: float,
    ) -> list[WeatherObservation]:
        """Fetch weather points for a location and time range."""
        ...


class NullWeatherProvider(WeatherProvider):
    """Deterministic provider used when weather enrichment is disabled."""

    def fetch(
        self,
        start: datetime,
        end: datetime,
        latitude: float,
        longitude: float,
    ) -> list[WeatherObservation]:
        """Return no weather points without failing the pipeline."""
        _ = (start, end, latitude, longitude)
        return []
