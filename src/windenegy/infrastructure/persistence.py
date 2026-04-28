"""Concrete repository implementations.

These classes implement the repository interfaces using specific
storage technologies (CSV, Parquet, filesystem).
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import polars as pl

from windenegy.domain.models import (
    DataQualityReport,
    ModelArtifactMetadata,
    TurbineObservation,
    ValidationError,
    ValidationSummary,
)
from windenegy.domain.repository import (
    ModelRepository,
    TurbineDataRepository,
)


class CSVTurbineRepository(TurbineDataRepository):
    """File-based CSV repository for turbine SCADA data."""

    COLUMN_MAP: ClassVar[dict[str, str]] = {
        "Date/Time": "timestamp",
        "LV ActivePower (kW)": "active_power_kw",
        "Wind Speed (m/s)": "wind_speed_mps",
        "Theoretical_Power_Curve (KWh)": "theoretical_power_kwh",
        "Wind Direction (°)": "wind_direction_deg",
    }

    def __init__(self, data_dir: Path) -> None:
        """Initialize with data directory.

        Args:
            data_dir: Root directory for raw data storage.
        """
        self._data_dir = Path(data_dir)
        self._metadata: dict[str, DataQualityReport] = {}

    def load_observations(
        self,
        asset_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[TurbineObservation]:
        """Load observations from CSV file."""
        file_path = self._get_file_path(asset_id)

        if not file_path.exists():
            return []

        # Read with polars for performance, convert to pandas for validation
        df = pl.read_csv(file_path, try_parse_dates=True).to_pandas()

        # Normalize column names
        df = self._normalize_columns(df)

        # Parse timestamps
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Apply time filter
        if start is not None:
            df = df[df["timestamp"] >= start]
        if end is not None:
            df = df[df["timestamp"] < end]

        # Convert to domain models
        return [TurbineObservation(**row) for row in df.to_dict("records")]

    def save_observations(
        self,
        asset_id: str,
        observations: list[TurbineObservation],
        path: Path | None = None,
    ) -> Path:
        """Save observations to CSV."""
        if path is None:
            path = self._data_dir / f"{asset_id}.csv"

        path.parent.mkdir(parents=True, exist_ok=True)

        records = [obs.model_dump() for obs in observations]
        df = pd.DataFrame(records)

        # Rename columns back to original format for compatibility
        reverse_map = {v: k for k, v in self.COLUMN_MAP.items()}
        df = df.rename(columns=reverse_map)

        df.to_csv(path, index=False)
        return path

    def get_validation_report(self, asset_id: str) -> DataQualityReport:
        """Generate data quality report."""
        if asset_id in self._metadata:
            return self._metadata[asset_id]

        file_path = self._get_file_path(asset_id)
        if not file_path.exists():
            msg = f"No data found for asset {asset_id}"
            raise FileNotFoundError(msg)

        df = pl.read_csv(file_path, try_parse_dates=True).to_pandas()
        df = self._normalize_columns(df)

        report = self._validate_dataframe(df)
        self._metadata[asset_id] = report
        return report

    def get_available_assets(self) -> list[str]:
        """List CSV files in data directory."""
        if not self._data_dir.exists():
            return []
        return [f.stem for f in self._data_dir.glob("*.csv")]

    def get_time_range(self, asset_id: str) -> tuple[datetime, datetime] | None:
        """Get time range from file."""
        file_path = self._get_file_path(asset_id)
        if not file_path.exists():
            return None

        df = pl.read_csv(file_path, try_parse_dates=True).to_pandas()
        df = self._normalize_columns(df)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        return df["timestamp"].min(), df["timestamp"].max()

    def _get_file_path(self, asset_id: str) -> Path:
        """Resolve file path for asset."""
        return self._data_dir / f"{asset_id}.csv"

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to internal format."""
        return df.rename(columns=self.COLUMN_MAP)

    def _validate_dataframe(self, df: pd.DataFrame) -> DataQualityReport:
        """Validate dataframe and return quality report."""
        missing_counts = {col: int(df[col].isna().sum()) for col in df.columns}

        range_violations: dict[str, int] = {}

        # Check wind speed is non-negative
        if "wind_speed_mps" in df.columns:
            range_violations["wind_speed_mps_negative"] = int((df["wind_speed_mps"] < 0).sum())

        # Check wind direction is in [0, 360)
        if "wind_direction_deg" in df.columns:
            range_violations["wind_direction_out_of_range"] = int(
                ((df["wind_direction_deg"] < 0) | (df["wind_direction_deg"] >= 360)).sum()
            )

        # Check power is non-negative
        if "active_power_kw" in df.columns:
            range_violations["active_power_negative"] = int((df["active_power_kw"] < 0).sum())

        # Check for monotonic timestamps
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp")
        monotonic = df["timestamp"].is_monotonic_increasing

        # Count duplicates
        duplicate_timestamps = int(df["timestamp"].duplicated().sum())

        return DataQualityReport(
            total_records=len(df),
            missing_counts=missing_counts,
            range_violations=range_violations,
            duplicate_timestamps=duplicate_timestamps,
            monotonic=bool(monotonic),
            start_time=df["timestamp"].min().to_pydatetime() if monotonic else None,
            end_time=df["timestamp"].max().to_pydatetime() if monotonic else None,
        )


class FileSystemModelRepository(ModelRepository):
    """Filesystem-based model artifact repository."""

    METADATA_FILENAME = "metadata.json"
    MODEL_FILENAME = "model.pkl"

    def __init__(self, base_dir: Path) -> None:
        """Initialize with base directory.

        Args:
            base_dir: Directory for model storage.
        """
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def save_model(
        self,
        model_id: str,
        model: Any,
        metadata: ModelArtifactMetadata,
    ) -> Path:
        """Save model to filesystem."""
        model_dir = self._base_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata_path = model_dir / self.METADATA_FILENAME
        with metadata_path.open("w") as f:
            json.dump(metadata.to_dict(), f, indent=2, default=str)

        # Save model
        model_path = model_dir / self.MODEL_FILENAME
        with model_path.open("wb") as f:
            pickle.dump(model, f)

        return model_dir

    def load_model(self, model_id: str) -> tuple[Any, ModelArtifactMetadata]:
        """Load model from filesystem."""
        model_dir = self._base_dir / model_id

        if not model_dir.exists():
            msg = f"Model {model_id} not found"
            raise FileNotFoundError(msg)

        # Load metadata
        metadata_path = model_dir / self.METADATA_FILENAME
        with metadata_path.open() as f:
            metadata_dict = json.load(f)
        metadata = ModelArtifactMetadata(**metadata_dict)

        # Load model
        model_path = model_dir / self.MODEL_FILENAME
        with model_path.open("rb") as f:
            model = pickle.load(f)

        return model, metadata

    def list_models(self) -> list[str]:
        """List all model directories."""
        return [d.name for d in self._base_dir.iterdir() if d.is_dir()]

    def get_latest_model(self, model_type: str | None = None) -> str | None:
        """Get most recent model by training timestamp."""
        models = self.list_models()
        if not models:
            return None

        candidates: list[tuple[str, datetime]] = []
        for model_id in models:
            try:
                _, metadata = self.load_model(model_id)
                if model_type is None or metadata.model_type == model_type:
                    candidates.append((model_id, metadata.training_timestamp))
            except (FileNotFoundError, json.JSONDecodeError):
                continue

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]


class DataValidator:
    """Comprehensive data validation utility."""

    def __init__(self, strict: bool = False) -> None:
        """Initialize validator.

        Args:
            strict: If True, treat warnings as errors.
        """
        self._strict = strict
        self._errors: list[ValidationError] = []
        self._warnings: list[ValidationError] = []

    def validate_required_columns(
        self,
        df: pd.DataFrame,
        required: list[str],
    ) -> DataValidator:
        """Validate required columns exist."""
        missing = set(required) - set(df.columns)
        if missing:
            self._errors.append(
                ValidationError(
                    field="columns",
                    check="required",
                    message=f"Missing required columns: {missing}",
                    count=len(missing),
                    severity="error",
                )
            )
        return self

    def validate_timestamp_column(
        self,
        df: pd.DataFrame,
        column: str = "timestamp",
    ) -> DataValidator:
        """Validate timestamp column."""
        if column not in df.columns:
            self._errors.append(
                ValidationError(
                    field=column,
                    check="exists",
                    message=f"Timestamp column '{column}' not found",
                    count=1,
                    severity="error",
                )
            )
            return self

        # Try to parse timestamps
        try:
            parsed = pd.to_datetime(df[column], utc=True)
        except (ValueError, TypeError) as e:
            self._errors.append(
                ValidationError(
                    field=column,
                    check="parseable",
                    message=f"Failed to parse timestamps: {e}",
                    count=1,
                    severity="error",
                )
            )
            return self

        # Check for monotonicity
        if not parsed.is_monotonic_increasing:
            non_monotonic = (~parsed.sort_values().diff().gt(0)).sum()
            self._warnings.append(
                ValidationError(
                    field=column,
                    check="monotonic",
                    message="Timestamps are not strictly monotonic",
                    count=int(non_monotonic),
                    severity="error" if self._strict else "warning",
                )
            )

        # Check for duplicates
        duplicates = parsed.duplicated().sum()
        if duplicates > 0:
            self._warnings.append(
                ValidationError(
                    field=column,
                    check="duplicates",
                    message=f"Found {duplicates} duplicate timestamps",
                    count=int(duplicates),
                    severity="error" if self._strict else "warning",
                )
            )

        return self

    def validate_numeric_range(
        self,
        df: pd.DataFrame,
        column: str,
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> DataValidator:
        """Validate numeric column is within range."""
        if column not in df.columns:
            return self

        violations = 0
        if min_val is not None:
            violations += (df[column] < min_val).sum()
        if max_val is not None:
            violations += (df[column] > max_val).sum()

        if violations > 0:
            self._warnings.append(
                ValidationError(
                    field=column,
                    check="range",
                    message=f"Values outside valid range [{min_val}, {max_val}]",
                    count=int(violations),
                    severity="error" if self._strict else "warning",
                )
            )

        return self

    def get_summary(self) -> ValidationSummary:
        """Get validation summary."""
        return ValidationSummary(
            errors=self._errors,
            warnings=self._warnings,
            passed=len(self._errors) == 0 and (not self._strict or len(self._warnings) == 0),
        )
