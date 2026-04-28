"""Repository interfaces for data access.

Following the Repository pattern from Domain-Driven Design,
these interfaces define contracts for data access without
depending on specific storage implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from windenegy.domain.models import (
    DataQualityReport,
    DataSplit,
    ModelArtifactMetadata,
    TurbineObservation,
    WeatherObservation,
)


class TurbineDataRepository(ABC):
    """Abstract repository for turbine SCADA data.

    Implementations handle the specifics of reading from CSV,
    Parquet, databases, or other storage systems.
    """

    @abstractmethod
    def load_observations(
        self,
        asset_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[TurbineObservation]:
        """Load turbine observations for a time range.

        Args:
            asset_id: Identifier for the turbine.
            start: Inclusive start time. If None, load from beginning.
            end: Exclusive end time. If None, load until end.

        Returns:
            List of observations in chronological order.
        """
        ...

    @abstractmethod
    def save_observations(
        self,
        asset_id: str,
        observations: list[TurbineObservation],
        path: Path | None = None,
    ) -> Path:
        """Save observations to storage.

        Args:
            asset_id: Identifier for the turbine.
            observations: Observations to save.
            path: Optional explicit path. If None, use default location.

        Returns:
            Path where data was saved.
        """
        ...

    @abstractmethod
    def get_validation_report(self, asset_id: str) -> DataQualityReport:
        """Get quality report for the data.

        Args:
            asset_id: Identifier for the turbine.

        Returns:
            Data quality report.
        """
        ...

    @abstractmethod
    def get_available_assets(self) -> list[str]:
        """List all available asset IDs in the repository.

        Returns:
            List of asset identifiers.
        """
        ...

    @abstractmethod
    def get_time_range(self, asset_id: str) -> tuple[datetime, datetime] | None:
        """Get the time range of available data for an asset.

        Args:
            asset_id: Identifier for the turbine.

        Returns:
            Tuple of (earliest, latest) timestamps, or None if no data.
        """
        ...


class WeatherDataRepository(ABC):
    """Abstract repository for weather data."""

    @abstractmethod
    def load_observations(
        self,
        latitude: float,
        longitude: float,
        start: datetime,
        end: datetime,
    ) -> list[WeatherObservation]:
        """Load weather observations/forecasts for a location and time range.

        Args:
            latitude: Location latitude.
            longitude: Location longitude.
            start: Inclusive start time.
            end: Exclusive end time.

        Returns:
            List of weather observations in chronological order.
        """
        ...

    @abstractmethod
    def save_observations(
        self,
        observations: list[WeatherObservation],
        latitude: float,
        longitude: float,
    ) -> Path:
        """Save weather observations to cache.

        Args:
            observations: Weather observations to save.
            latitude: Location latitude.
            longitude: Location longitude.

        Returns:
            Path where data was saved.
        """
        ...


class ModelRepository(ABC):
    """Abstract repository for model artifacts.

    Handles persistence of trained models and their metadata.
    """

    @abstractmethod
    def save_model(
        self,
        model_id: str,
        model: Any,
        metadata: ModelArtifactMetadata,
    ) -> Path:
        """Save a trained model with its metadata.

        Args:
            model_id: Unique identifier for this model instance.
            model: The model object (implementation-specific).
            metadata: Model metadata and schema.

        Returns:
            Path where model was saved.
        """
        ...

    @abstractmethod
    def load_model(self, model_id: str) -> tuple[Any, ModelArtifactMetadata]:
        """Load a model and its metadata.

        Args:
            model_id: Model identifier.

        Returns:
            Tuple of (model object, metadata).

        Raises:
            FileNotFoundError: If model not found.
        """
        ...

    @abstractmethod
    def list_models(self) -> list[str]:
        """List all available model IDs.

        Returns:
            List of model identifiers.
        """
        ...

    @abstractmethod
    def get_latest_model(self, model_type: str | None = None) -> str | None:
        """Get the ID of the most recently trained model.

        Args:
            model_type: Optional filter by model type.

        Returns:
            Model ID or None if no models exist.
        """
        ...


class SplitStrategy(ABC):
    """Abstract strategy for creating data splits.

    Implementations define how data is partitioned into
    train/validation/test sets.
    """

    @abstractmethod
    def create_splits(
        self,
        start: datetime,
        end: datetime,
    ) -> DataSplit:
        """Create data splits for a time range.

        Args:
            start: Earliest timestamp in dataset.
            end: Latest timestamp in dataset.

        Returns:
            Data split configuration.
        """
        ...


class ChronologicalSplitStrategy(SplitStrategy):
    """Chronological split preventing data leakage.

    Splits are: [train][validation][test] in time order.
    """

    def __init__(
        self,
        train_ratio: float = 0.7,
        validation_ratio: float = 0.15,
    ) -> None:
        """Initialize with split ratios.

        Args:
            train_ratio: Proportion for training (default 0.7).
            validation_ratio: Proportion for validation (default 0.15).
                           Test gets the remainder (default 0.15).
        """
        if train_ratio + validation_ratio >= 1.0:
            msg = "Train + validation ratios must be < 1.0"
            raise ValueError(msg)
        self._train_ratio = train_ratio
        self._validation_ratio = validation_ratio

    def create_splits(
        self,
        start: datetime,
        end: datetime,
    ) -> DataSplit:
        """Create chronological splits."""
        total_duration = (end - start).total_seconds()
        train_end = start + self._duration_from_seconds(total_duration * self._train_ratio)
        val_end = train_end + self._duration_from_seconds(total_duration * self._validation_ratio)

        return DataSplit(
            train_start=start,
            train_end=train_end,
            validation_start=train_end,
            validation_end=val_end,
            test_start=val_end,
            test_end=end,
        )

    @staticmethod
    def _duration_from_seconds(seconds: float) -> timedelta:
        """Convert seconds to timedelta-compatible duration."""
        return timedelta(seconds=seconds)
