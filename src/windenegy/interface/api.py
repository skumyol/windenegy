"""FastAPI interface for wind power forecasting.

This module provides REST API endpoints for the forecasting service.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from windenegy.application.training import forecast_points_from_model
from windenegy.application.forecasting import ForecastService
from windenegy.application.risk import RampDetector, UnderproductionAnalyzer
from windenegy.application.training import load_latest_gradient_boosting, metric_summary
from windenegy.infrastructure.config import AppConfig
from windenegy.infrastructure.weather_provider import create_provider
from windenegy.domain.models import (
    ForecastHorizon,
    ForecastPoint,
    PowerForecast,
    PowerUnit,
    TurbineObservation,
)

app = FastAPI(
    title="Windenegy API",
    description="Wind power forecasting API for renewable energy operations",
    version="0.1.0",
)

forecast_service = ForecastService()
model_dir = Path(os.getenv("WINDENEGY_MODEL_ARTIFACTS_PATH", "artifacts/models"))

config = AppConfig()
weather_provider = None
if config.weather.provider == "openmeteo":
    weather_provider = create_provider(
        latitude=config.weather.latitude,
        longitude=config.weather.longitude,
        provider_type="openmeteo",
    )


class ForecastRequest(BaseModel):
    """Request model for forecast endpoint."""

    asset_id: str = Field(..., min_length=1, description="Turbine identifier")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Forecast creation timestamp",
    )
    horizon_hours: int = Field(
        ...,
        ge=1,
        le=72,
        description="Forecast horizon in hours",
    )
    observations: list[TurbineObservation] = Field(
        ...,
        min_length=1,
        description="Recent SCADA observations",
    )
    weather_forecast: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Weather forecast data",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: datetime
    version: str = "0.1.0"


class MetadataResponse(BaseModel):
    """API metadata response."""

    name: str = "Windenegy API"
    version: str = "0.1.0"
    description: str = "Wind power forecasting for renewable energy"
    available_models: list[str] = []
    supported_horizons: list[int] = [1, 6, 24]
    active_model: dict[str, Any] = Field(default_factory=dict)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(UTC),
    )


@app.get("/metadata", response_model=MetadataResponse)
async def get_metadata() -> MetadataResponse:
    """Get API metadata and capabilities."""
    trained = load_latest_gradient_boosting(model_dir)
    return MetadataResponse(
        available_models=["persistence", "gradient_boosting"] if trained else ["persistence"],
        active_model=metric_summary(trained[1]) if trained else {},
        supported_horizons=[1, 6, 24],
    )


@app.post("/forecast", response_model=PowerForecast)
async def create_forecast(request: ForecastRequest) -> PowerForecast:
    """Generate a power forecast for an asset.

    This endpoint accepts recent SCADA observations and returns
    a multi-horizon power forecast with uncertainty bands.
    """
    if not request.observations:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one observation required",
        )

    weather_forecast = request.weather_forecast
    if not weather_forecast and weather_provider is not None:
        try:
            weather_obs = weather_provider.fetch_forecast(hours_ahead=request.horizon_hours)
            weather_forecast = [
                {
                    "timestamp": w.timestamp.isoformat(),
                    "wind_speed_100m": w.wind_speed_100m_mps,
                    "wind_speed_10m": w.wind_speed_10m_mps,
                    "temperature": w.temperature_c,
                }
                for w in weather_obs
            ]
        except Exception:
            pass

    trained = None
    ml_fallback_reason = None
    horizon_mismatch = False
    try:
        trained = load_latest_gradient_boosting(model_dir)
        if trained is not None:
            model, metadata = trained
            if metadata.horizon_hours == request.horizon_hours:
                try:
                    points = [
                        ForecastPoint(**point)
                        for point in forecast_points_from_model(
                            model,
                            observations=request.observations,
                            created_at=request.created_at,
                        )
                    ]
                    return PowerForecast(
                        asset_id=request.asset_id,
                        model_version=metadata.model_version,
                        horizon_hours=ForecastHorizon(request.horizon_hours),
                        unit=PowerUnit.KILOWATT,
                        created_at=request.created_at,
                        forecast=points,
                        warnings=[],
                    )
                except (ValueError, KeyError) as exc:
                    ml_fallback_reason = str(exc)
            else:
                horizon_mismatch = True

        forecast = forecast_service.forecast(
            asset_id=request.asset_id,
            observations=request.observations,
            horizon_hours=request.horizon_hours,
            created_at=request.created_at,
        )
        warnings = list(forecast.warnings)
        # Only warn about ML fallback if we had enough data but ML wasn't used
        if trained is not None and len(request.observations) >= 10:
            if ml_fallback_reason:
                warnings.append(f"ML model error ({ml_fallback_reason}); using persistence baseline.")
            elif horizon_mismatch:
                warnings.append(f"ML model trained for {metadata.horizon_hours}h horizon; using persistence for {request.horizon_hours}h.")
        return forecast.model_copy(update={"warnings": warnings})
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


class RiskAssessmentRequest(BaseModel):
    """Request for risk assessment."""

    asset_id: str = Field(..., min_length=1)
    timestamps: list[datetime] = Field(..., min_length=1)
    p50_forecast: list[float] = Field(..., min_length=1)
    p10_forecast: list[float] = Field(..., min_length=1)
    p90_forecast: list[float] = Field(..., min_length=1)
    capacity_kw: float = Field(default=2000.0, gt=0)


class RampDetectionRequest(BaseModel):
    """Request for ramp event detection."""

    asset_id: str = Field(..., min_length=1)
    timestamps: list[datetime] = Field(..., min_length=2)
    power_values: list[float] = Field(..., min_length=2)
    interval_minutes: int = Field(default=10, ge=1)
    capacity_kw: float = Field(default=2000.0, gt=0)


class RampEventResponse(BaseModel):
    """Ramp event in response."""

    start_time: str
    end_time: str
    power_change_kw: float
    power_change_pct: float
    duration_minutes: int
    direction: str
    risk_level: str


class UnderproductionRiskResponse(BaseModel):
    """Underproduction risk in response."""

    timestamp: str
    expected_power_kw: float
    shortfall_threshold_kw: float
    shortfall_probability: float
    expected_shortfall_kw: float
    risk_level: str


class RiskAssessmentResponse(BaseModel):
    """Risk assessment response."""

    asset_id: str
    generated_at: datetime
    ramp_events: list[RampEventResponse]
    underproduction_risks: list[UnderproductionRiskResponse]


@app.post("/risk/assess", response_model=RiskAssessmentResponse)
async def assess_risks(request: RiskAssessmentRequest) -> RiskAssessmentResponse:
    """Assess operational risks from forecast.

    Identifies ramp events and underproduction risks based on
    forecast intervals and capacity thresholds.
    """
    if len(request.timestamps) != len(request.p50_forecast):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Timestamps and forecasts must have same length",
        )

    # Ramp detection on forecast
    ramp_detector = RampDetector(capacity_kw=request.capacity_kw)
    ramp_events = ramp_detector.detect_from_series(
        timestamps=[ts.isoformat() for ts in request.timestamps],
        power_values=request.p50_forecast,
    )

    # Underproduction analysis
    analyzer = UnderproductionAnalyzer(capacity_kw=request.capacity_kw)
    risks = analyzer.analyze_forecast(
        timestamps=[ts.isoformat() for ts in request.timestamps],
        p50_forecast=request.p50_forecast,
        p10_forecast=request.p10_forecast,
    )

    return RiskAssessmentResponse(
        asset_id=request.asset_id,
        generated_at=datetime.now(UTC),
        ramp_events=[
            RampEventResponse(
                start_time=e.start_time,
                end_time=e.end_time,
                power_change_kw=e.power_change_kw,
                power_change_pct=e.power_change_pct,
                duration_minutes=e.duration_minutes,
                direction=e.direction,
                risk_level=e.risk_level.value,
            )
            for e in ramp_events
        ],
        underproduction_risks=[
            UnderproductionRiskResponse(
                timestamp=r.timestamp,
                expected_power_kw=r.expected_power_kw,
                shortfall_threshold_kw=r.shortfall_threshold_kw,
                shortfall_probability=r.shortfall_probability,
                expected_shortfall_kw=r.expected_shortfall_kw,
                risk_level=r.risk_level.value,
            )
            for r in risks
        ],
    )


@app.post("/risk/ramps", response_model=list[RampEventResponse])
async def detect_ramps(request: RampDetectionRequest) -> list[RampEventResponse]:
    """Detect power ramp events in historical or forecast data.

    Identifies rapid power changes that could stress grid operations
    or trigger protective controls.
    """
    if len(request.timestamps) != len(request.power_values):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Timestamps and power values must have same length",
        )

    detector = RampDetector(capacity_kw=request.capacity_kw)
    events = detector.detect_from_series(
        timestamps=[ts.isoformat() for ts in request.timestamps],
        power_values=request.power_values,
        interval_minutes=request.interval_minutes,
    )

    return [
        RampEventResponse(
            start_time=e.start_time,
            end_time=e.end_time,
            power_change_kw=e.power_change_kw,
            power_change_pct=e.power_change_pct,
            duration_minutes=e.duration_minutes,
            direction=e.direction,
            risk_level=e.risk_level.value,
        )
        for e in events
    ]


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "message": "Windenegy API - Wind power forecasting service",
        "docs": "/docs",
        "health": "/health",
        "risk_endpoints": ["/risk/assess", "/risk/ramps"],
    }
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
