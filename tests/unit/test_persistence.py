"""Unit tests for persistence layer."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from windenegy.domain.models import ModelArtifactMetadata, TurbineObservation
from windenegy.domain.repository import ChronologicalSplitStrategy
from windenegy.infrastructure.persistence import (
    CSVTurbineRepository,
    DataValidator,
    FileSystemModelRepository,
)


class TestCSVTurbineRepository:
    """Tests for CSV turbine repository."""

    @pytest.fixture
    def repo(self, tmp_path: Path) -> CSVTurbineRepository:
        """Create a temporary repository."""
        return CSVTurbineRepository(tmp_path)

    @pytest.fixture
    def sample_observations(self) -> list[TurbineObservation]:
        """Create sample observations."""
        return [
            TurbineObservation(
                timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
                active_power_kw=1000.0,
                wind_speed_mps=7.5,
                wind_direction_deg=180.0,
                theoretical_power_kwh=950.0,
            ),
            TurbineObservation(
                timestamp=datetime(2024, 1, 1, 12, 10, 0, tzinfo=UTC),
                active_power_kw=1050.0,
                wind_speed_mps=8.0,
                wind_direction_deg=185.0,
                theoretical_power_kwh=1000.0,
            ),
        ]

    def test_save_and_load_observations(
        self,
        repo: CSVTurbineRepository,
        sample_observations: list[TurbineObservation],
    ) -> None:
        """Test saving and loading observations."""
        # Save
        path = repo.save_observations("T1", sample_observations)
        assert path.exists()

        # Load
        loaded = repo.load_observations("T1")
        assert len(loaded) == 2
        assert loaded[0].active_power_kw == 1000.0

    def test_load_with_time_filter(
        self,
        repo: CSVTurbineRepository,
        sample_observations: list[TurbineObservation],
    ) -> None:
        """Test loading with time range filter."""
        repo.save_observations("T1", sample_observations)

        # Load only first observation
        start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        end = datetime(2024, 1, 1, 12, 5, 0, tzinfo=UTC)
        loaded = repo.load_observations("T1", start=start, end=end)

        assert len(loaded) == 1
        assert loaded[0].timestamp.minute == 0

    def test_get_available_assets(self, repo: CSVTurbineRepository) -> None:
        """Test listing available assets."""
        # Initially empty
        assert repo.get_available_assets() == []

        # Add some data
        repo.save_observations("T1", [])
        repo.save_observations("T2", [])

        assets = repo.get_available_assets()
        assert "T1" in assets
        assert "T2" in assets

    def test_get_time_range(
        self,
        repo: CSVTurbineRepository,
        sample_observations: list[TurbineObservation],
    ) -> None:
        """Test getting time range for asset."""
        # Non-existent asset
        assert repo.get_time_range("T1") is None

        # Save and check
        repo.save_observations("T1", sample_observations)
        time_range = repo.get_time_range("T1")
        assert time_range is not None
        assert time_range[1] > time_range[0]


class TestFileSystemModelRepository:
    """Tests for filesystem model repository."""

    @pytest.fixture
    def repo(self, tmp_path: Path) -> FileSystemModelRepository:
        """Create a temporary repository."""
        return FileSystemModelRepository(tmp_path)

    @pytest.fixture
    def sample_metadata(self) -> ModelArtifactMetadata:
        """Create sample metadata."""
        return ModelArtifactMetadata(
            model_type="gradient_boosting",
            model_version="test-1.0",
            feature_schema={"wind_speed": "float64"},
            target="active_power_kw",
            horizon_hours=6,
        )

    def test_save_and_load_model(
        self,
        repo: FileSystemModelRepository,
        sample_metadata: ModelArtifactMetadata,
    ) -> None:
        """Test saving and loading model."""
        dummy_model = {"coef": [1.0, 2.0], "intercept": 0.5}

        # Save
        path = repo.save_model("test-model", dummy_model, sample_metadata)
        assert path.exists()

        # Load
        loaded_model, loaded_metadata = repo.load_model("test-model")
        assert loaded_model == dummy_model
        assert loaded_metadata.model_type == "gradient_boosting"

    def test_list_models(
        self,
        repo: FileSystemModelRepository,
        sample_metadata: ModelArtifactMetadata,
    ) -> None:
        """Test listing models."""
        assert repo.list_models() == []

        repo.save_model("model-a", {}, sample_metadata)
        repo.save_model("model-b", {}, sample_metadata)

        models = repo.list_models()
        assert "model-a" in models
        assert "model-b" in models

    def test_get_latest_model(
        self,
        repo: FileSystemModelRepository,
        sample_metadata: ModelArtifactMetadata,
    ) -> None:
        """Test getting latest model."""
        # No models
        assert repo.get_latest_model() is None

        # Add models
        repo.save_model("model-old", {}, sample_metadata)

        # Get latest (only one exists)
        latest = repo.get_latest_model()
        assert latest == "model-old"


class TestChronologicalSplitStrategy:
    """Tests for chronological split strategy."""

    def test_split_ratios(self) -> None:
        """Test split proportions."""
        strategy = ChronologicalSplitStrategy(
            train_ratio=0.7,
            validation_ratio=0.15,
        )

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 2, tzinfo=UTC)

        split = strategy.create_splits(start, end)

        # Check chronological order
        assert split.train_start == start
        assert split.train_end <= split.validation_start
        assert split.validation_end <= split.test_start
        assert split.test_end == end

    def test_invalid_ratios(self) -> None:
        """Test that invalid ratios are rejected."""
        with pytest.raises(ValueError):
            ChronologicalSplitStrategy(
                train_ratio=0.6,
                validation_ratio=0.5,  # Total > 1.0
            )


class TestDataValidator:
    """Tests for data validation."""

    def test_validate_required_columns(self) -> None:
        """Test required column validation."""
        df = pd.DataFrame(
            {
                "timestamp": ["2024-01-01"],
                "power": [100.0],
            }
        )

        validator = DataValidator()
        validator.validate_required_columns(df, ["timestamp", "power"])
        summary = validator.get_summary()

        assert summary.passed
        assert len(summary.errors) == 0

    def test_validate_missing_columns(self) -> None:
        """Test detection of missing columns."""
        df = pd.DataFrame({"timestamp": ["2024-01-01"]})

        validator = DataValidator()
        validator.validate_required_columns(df, ["timestamp", "power"])
        summary = validator.get_summary()

        assert not summary.passed
        assert len(summary.errors) == 1
        assert summary.errors[0].field == "columns"
