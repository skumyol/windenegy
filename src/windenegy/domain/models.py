"""Core domain models for wind power forecasting.

These models represent the immutable business entities of the system.
They have zero external dependencies beyond the Python standard library and Pydantic.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum, StrEnum
from math import isfinite
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PowerUnit(StrEnum):
    """Valid units for power measurements."""

    KILOWATT = "kW"
    MEGAWATT = "MW"


class WindDirectionUnit(StrEnum):
    """Valid units for wind direction measurements."""

    DEGREES = "deg"


class SpeedUnit(StrEnum):
    """Valid units for speed measurements."""

    METERS_PER_SECOND = "m/s"


class TurbineObservation(BaseModel):
    """A single SCADA observation from a wind turbine.

    Attributes:
        timestamp: UTC timestamp of the observation.
        active_power_kw: Active power output in kilowatts. Must be non-negative.
        wind_speed_mps: Wind speed at hub height in meters per second. Must be non-negative.
        wind_direction_deg: Wind direction in degrees [0, 360).
        theoretical_power_kwh: Theoretical power from manufacturer curve in kWh.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    timestamp: datetime = Field(..., description="UTC timestamp of observation")
    active_power_kw: float = Field(..., ge=0.0, description="Active power output in kilowatts")
    wind_speed_mps: float = Field(..., ge=0.0, description="Wind speed at hub height in m/s")
    wind_direction_deg: float = Field(
        ..., ge=0.0, lt=360.0, description="Wind direction in degrees"
    )
    theoretical_power_kwh: float = Field(
        ..., ge=0.0, description="Theoretical power from manufacturer curve in kWh"
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def _ensure_utc(cls, value: datetime | str) -> datetime:
        """Ensure timestamp is timezone-aware UTC."""
        if isinstance(value, str):
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if value.tzinfo is None:
            msg = "Timestamp must be timezone-aware"
            raise ValueError(msg)
        return value.astimezone(UTC)


class WeatherObservation(BaseModel):
    """A single weather observation or forecast point.

    Attributes:
        timestamp: UTC timestamp of the observation/forecast.
        wind_speed_10m_mps: Wind speed at 10m height.
        wind_speed_100m_mps: Wind speed at 100m height (if available).
        wind_direction_deg: Wind direction in degrees [0, 360).
        temperature_c: Air temperature in Celsius.
        pressure_hpa: Atmospheric pressure in hPa.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    timestamp: datetime = Field(..., description="UTC timestamp")
    wind_speed_10m_mps: float | None = Field(None, ge=0.0, description="Wind speed at 10m in m/s")
    wind_speed_100m_mps: float | None = Field(None, ge=0.0, description="Wind speed at 100m in m/s")
    wind_direction_deg: float | None = Field(
        None, ge=0.0, lt=360.0, description="Wind direction in degrees"
    )
    temperature_c: float | None = Field(None, description="Air temperature in Celsius")
    pressure_hpa: float | None = Field(None, ge=0.0, description="Pressure in hPa")

    @field_validator("timestamp", mode="before")
    @classmethod
    def _ensure_utc(cls, value: datetime | str) -> datetime:
        """Ensure timestamp is timezone-aware UTC."""
        if isinstance(value, str):
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if value.tzinfo is None:
            msg = "Timestamp must be timezone-aware"
            raise ValueError(msg)
        return value.astimezone(UTC)


class ForecastHorizon(Enum):
    """Supported forecast horizons."""

    HOUR_1 = 1
    HOUR_6 = 6
    HOUR_24 = 24


class ForecastPoint(BaseModel):
    """A single point in a power forecast.

    Attributes:
        timestamp: UTC timestamp for this forecast point.
        p50: Median (P50) forecast value.
        p10: Lower bound (P10) forecast value.
        p90: Upper bound (P90) forecast value.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    timestamp: datetime
    p50: float = Field(..., description="Median forecast value")
    p10: float = Field(..., description="10th percentile forecast")
    p90: float = Field(..., description="90th percentile forecast")

    @field_validator("p10", "p50", "p90")
    @classmethod
    def _ensure_finite(cls, v: float) -> float:
        """Ensure forecast values are finite numbers."""
        if not isfinite(v):
            msg = "Forecast value cannot be NaN"
            raise ValueError(msg)
        return float(v)


class PowerForecast(BaseModel):
    """A complete power forecast for an asset.

    Attributes:
        asset_id: Identifier for the wind turbine or farm.
        model_version: Version identifier of the model that produced this forecast.
        horizon_hours: Forecast horizon in hours.
        unit: Unit of the forecast values.
        created_at: UTC timestamp when the forecast was generated.
        forecast: Sequence of forecast points.
        warnings: Any warnings about forecast quality or data issues.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    asset_id: str = Field(..., min_length=1, description="Asset identifier")
    model_version: str = Field(..., min_length=1, description="Model version string")
    horizon_hours: ForecastHorizon
    unit: PowerUnit = PowerUnit.KILOWATT
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Forecast generation timestamp",
    )
    forecast: list[ForecastPoint] = Field(..., min_length=1)
    warnings: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize forecast to plain dictionary."""
        return self.model_dump(mode="json")


class AssetMetadata(BaseModel):
    """Metadata about a wind asset.

    Attributes:
        asset_id: Unique identifier.
        latitude: Asset latitude.
        longitude: Asset longitude.
        rated_power_kw: Maximum rated power in kW.
        hub_height_m: Turbine hub height in meters.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    asset_id: str = Field(..., min_length=1)
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    rated_power_kw: float = Field(..., gt=0.0)
    hub_height_m: float = Field(..., gt=0.0)


class DataSplit(BaseModel):
    """Time-based data split configuration.

    All splits are chronological to prevent leakage.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    train_start: datetime
    train_end: datetime
    validation_start: datetime
    validation_end: datetime
    test_start: datetime
    test_end: datetime

    @field_validator("train_end", "validation_end")
    @classmethod
    def _ensure_chronological(cls, v: datetime, info: Any) -> datetime:
        """Ensure splits are in chronological order."""
        data = info.data
        if "train_start" in data and v < data["train_start"]:
            msg = "End must be after start"
            raise ValueError(msg)
        return v


class ModelMetrics(BaseModel):
    """Performance metrics for a forecasting model.

    Attributes:
        model_id: Identifier for the model.
        horizon_hours: Forecast horizon evaluated.
        mae: Mean Absolute Error.
        rmse: Root Mean Squared Error.
        mape: Mean Absolute Percentage Error (in percent).
        skill_score: Skill score versus persistence baseline.
        coverage_p90: Actual coverage of P90 prediction intervals.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    model_id: str
    horizon_hours: int
    mae: float = Field(..., ge=0.0)
    rmse: float = Field(..., ge=0.0)
    mape: float = Field(..., ge=0.0)
    skill_score: float | None = None
    coverage_p90: float | None = Field(None, ge=0.0, le=1.0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics to plain dictionary."""
        return self.model_dump(mode="json")


class DataQualityReport(BaseModel):
    """Report on data quality after ingestion and validation.

    Attributes:
        total_records: Total number of records processed.
        missing_counts: Count of missing values by column.
        range_violations: Count of values outside physical ranges.
        duplicate_timestamps: Number of duplicate timestamps found.
        monotonic: Whether timestamps are strictly monotonic.
        start_time: Earliest timestamp.
        end_time: Latest timestamp.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    total_records: int = Field(..., ge=0)
    missing_counts: dict[str, int] = Field(default_factory=dict)
    range_violations: dict[str, int] = Field(default_factory=dict)
    duplicate_timestamps: int = Field(..., ge=0)
    monotonic: bool
    start_time: datetime | None = None
    end_time: datetime | None = None

    @property
    def is_valid(self) -> bool:
        """Quick check if data passes basic quality thresholds."""
        return self.monotonic and self.duplicate_timestamps == 0


class ModelArtifactMetadata(BaseModel):
    """Metadata stored alongside a serialized model artifact.

    This is the contract that links a trained model to its
    feature schema, target, and evaluation context.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    model_type: str = Field(..., min_length=1)
    model_version: str = Field(..., min_length=1)
    feature_schema: dict[str, str] = Field(..., description="Feature name -> type mapping")
    target: str = Field(..., description="Target variable name")
    horizon_hours: int = Field(..., gt=0)
    training_window_hours: int | None = None
    metrics: ModelMetrics | None = None
    git_sha: str | None = None
    training_timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    config_snapshot: dict[str, Any] = Field(default_factory=dict)

    @field_validator("training_timestamp", mode="before")
    @classmethod
    def _parse_training_timestamp(cls, value: datetime | str) -> datetime:
        """Parse JSON-serialized metadata timestamps."""
        if isinstance(value, str):
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to plain dictionary."""
        return self.model_dump(mode="json")


class RampEvent(BaseModel):
    """A detected ramp event in power output.

    Ramp events are significant changes in power output that
    pose operational challenges.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    start_time: datetime
    end_time: datetime
    power_before_kw: float
    power_after_kw: float
    ramp_magnitude_kw: float = Field(..., description="Absolute change in kW")
    ramp_rate_kw_per_min: float = Field(..., description="Rate of change")
    direction: str = Field(..., pattern="^(up|down)$")

    @property
    def duration_minutes(self) -> float:
        """Duration of the ramp event in minutes."""
        return (self.end_time - self.start_time).total_seconds() / 60.0


class ValidationError(BaseModel):
    """A single validation error found during data quality checks."""

    model_config = ConfigDict(frozen=True, strict=True)

    field: str
    check: str
    message: str
    count: int = Field(..., ge=0)
    severity: str = Field(default="warning", pattern="^(warning|error)$")


class ValidationSummary(BaseModel):
    """Summary of all validation checks performed on a dataset."""

    model_config = ConfigDict(frozen=True, strict=True)

    errors: list[ValidationError] = Field(default_factory=list)
    warnings: list[ValidationError] = Field(default_factory=list)
    passed: bool = False
    checked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def has_errors(self) -> bool:
        """Return True if any errors were found."""
        return any(e.severity == "error" for e in self.errors)

    @property
    def total_issues(self) -> int:
        """Total number of issues (errors + warnings)."""
        return sum(e.count for e in self.errors + self.warnings)
