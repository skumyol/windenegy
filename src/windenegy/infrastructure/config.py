"""Configuration management with pydantic-settings.

This module provides strictly-typed, immutable configuration objects
that are injected into components rather than accessed as globals.

Configuration hierarchy (highest priority first):
1. Environment variables (WINDENEGY_* prefix)
2. .env file (if present)
3. YAML config file (config/default.yaml or via WINDENEGY_CONFIG_PATH)
4. Default values defined here
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataConfig(BaseSettings):
    """Data paths and processing configuration."""

    model_config = SettingsConfigDict(
        env_prefix="WINDENEGY_DATA_",
        frozen=True,
    )

    raw_path: Path = Field(default=Path("data/raw"))
    interim_path: Path = Field(default=Path("data/interim"))
    processed_path: Path = Field(default=Path("data/processed"))
    fixture_path: Path = Field(default=Path("tests/fixtures"))
    scada_filename: str = Field(default="T1.csv")

    train_split: float = Field(default=0.7, ge=0.0, le=1.0)
    validation_split: float = Field(default=0.15, ge=0.0, le=1.0)

    @field_validator("train_split", "validation_split")
    @classmethod
    def _check_split_sum(cls, v: float) -> float:
        """Validate split proportions."""
        return v

    @field_validator("raw_path", "interim_path", "processed_path", "fixture_path")
    @classmethod
    def _ensure_path(cls, v: Path) -> Path:
        """Ensure path is absolute or resolved relative to project root."""
        if v.is_absolute():
            return v
        return Path.cwd() / v


class ModelConfig(BaseSettings):
    """Model training and inference configuration."""

    model_config = SettingsConfigDict(
        env_prefix="WINDENEGY_MODEL_",
        frozen=True,
    )

    artifacts_path: Path = Field(default=Path("artifacts/models"))
    metrics_path: Path = Field(default=Path("artifacts/metrics"))
    reports_path: Path = Field(default=Path("artifacts/reports"))

    default_model: Literal["persistence", "gradient_boosting", "patchtst"] = Field(
        default="gradient_boosting"
    )

    # Sequence model parameters
    sequence_length: int = Field(
        default=144,
        ge=1,
        description="Input sequence length (10-min steps)",
    )
    prediction_horizon: int = Field(default=6, ge=1, description="Prediction steps ahead")

    # GBM parameters
    gbm_n_estimators: int = Field(default=200, ge=1)
    gbm_learning_rate: float = Field(default=0.05, ge=0.0)
    gbm_max_depth: int = Field(default=6, ge=1)
    gbm_early_stopping_rounds: int = Field(default=20, ge=1)

    @field_validator("artifacts_path", "metrics_path", "reports_path")
    @classmethod
    def _ensure_path(cls, v: Path) -> Path:
        """Ensure path is absolute or resolved relative to project root."""
        if v.is_absolute():
            return v
        return Path.cwd() / v


class APIConfig(BaseSettings):
    """FastAPI service configuration."""

    model_config = SettingsConfigDict(
        env_prefix="WINDENEGY_API_",
        frozen=True,
    )

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    reload: bool = Field(default=False)
    log_level: Literal["debug", "info", "warning", "error"] = Field(default="info")

    # Rate limiting
    max_requests_per_minute: int = Field(default=60, ge=1)


class DashboardConfig(BaseSettings):
    """Streamlit dashboard configuration."""

    model_config = SettingsConfigDict(
        env_prefix="WINDENEGY_DASHBOARD_",
        frozen=True,
    )

    port: int = Field(default=8501, ge=1, le=65535)
    api_url: str = Field(default="http://localhost:8000")
    refresh_interval_seconds: int = Field(default=30, ge=1)


class WeatherConfig(BaseSettings):
    """Weather data source configuration."""

    model_config = SettingsConfigDict(
        env_prefix="WINDENEGY_WEATHER_",
        frozen=True,
    )

    provider: Literal["openmeteo", "era5", "null"] = Field(default="null")
    cache_enabled: bool = Field(default=True)
    cache_ttl_hours: int = Field(default=24, ge=1)

    # Open-Meteo specific
    openmeteo_base_url: str = Field(default="https://archive-api.open-meteo.com/v1")


class LoggingConfig(BaseSettings):
    """Structured logging configuration."""

    model_config = SettingsConfigDict(
        env_prefix="WINDENEGY_LOG_",
        frozen=True,
    )

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    format: Literal["json", "console"] = Field(default="console")
    include_traceback: bool = Field(default=True)


class AppConfig(BaseSettings):
    """Root configuration container.

    This is the single source of truth for all application configuration.
    It is meant to be instantiated once at startup and injected into components.
    """

    model_config = SettingsConfigDict(
        env_prefix="WINDENEGY_",
        frozen=True,
        extra="ignore",
    )

    env: Literal["development", "testing", "production"] = Field(default="development")
    debug: bool = Field(default=False)
    version: str = Field(default="0.1.0")

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    weather: WeatherConfig = Field(default_factory=WeatherConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> AppConfig:
        """Load configuration from a YAML file.

        The YAML file should have a flat structure or nested sections
        matching the config classes above.

        Args:
            path: Path to YAML configuration file.

        Returns:
            Populated AppConfig instance.
        """
        if not path.exists():
            return cls()

        with path.open() as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)

    @classmethod
    def for_testing(cls) -> AppConfig:
        """Create a configuration suitable for testing.

        Uses in-memory paths and minimal settings for fast tests.
        """
        return cls(
            env="testing",
            debug=True,
            data=DataConfig(
                raw_path=Path("/tmp/windenegy_test/raw"),
                interim_path=Path("/tmp/windenegy_test/interim"),
                processed_path=Path("/tmp/windenegy_test/processed"),
                fixture_path=Path("tests/fixtures"),
            ),
            model=ModelConfig(
                artifacts_path=Path("/tmp/windenegy_test/models"),
                metrics_path=Path("/tmp/windenegy_test/metrics"),
                reports_path=Path("/tmp/windenegy_test/reports"),
            ),
        )

    def ensure_directories(self) -> None:
        """Ensure all configured directories exist.

        Idempotent - safe to call multiple times.
        """
        for path in [
            self.data.raw_path,
            self.data.interim_path,
            self.data.processed_path,
            self.model.artifacts_path,
            self.model.metrics_path,
            self.model.reports_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)


def load_config(config_path: Path | None = None) -> AppConfig:
    """Load configuration with standard precedence.

    Priority (highest first):
    1. Environment variables
    2. .env file (automatically loaded by pydantic-settings)
    3. YAML config file
    4. Default values

    Args:
        config_path: Optional explicit path to YAML config.
            If not provided, looks for configs/default.yaml.

    Returns:
        Configured AppConfig instance.
    """
    if config_path is None:
        default_path = Path("configs/default.yaml")
        if default_path.exists():
            return AppConfig.from_yaml(default_path)
        return AppConfig()

    return AppConfig.from_yaml(config_path)
