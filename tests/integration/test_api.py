"""Integration tests for the FastAPI interface."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from windenegy.interface import api

pytestmark = pytest.mark.integration


def test_forecast_endpoint_returns_sequence_forecast(tmp_path: Path) -> None:
    """The forecast endpoint returns a schema-compatible multi-point forecast."""
    api.model_dir = tmp_path / "models"
    client = TestClient(api.app)
    payload = {
        "asset_id": "T1",
        "horizon_hours": 1,
        "observations": [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "active_power_kw": 500.0,
                "wind_speed_mps": 6.0,
                "wind_direction_deg": 120.0,
                "theoretical_power_kwh": 550.0,
            },
            {
                "timestamp": "2024-01-01T00:10:00Z",
                "active_power_kw": 520.0,
                "wind_speed_mps": 6.2,
                "wind_direction_deg": 121.0,
                "theoretical_power_kwh": 570.0,
            },
        ],
    }

    response = client.post("/forecast", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["asset_id"] == "T1"
    assert len(body["forecast"]) == 6


def test_forecast_endpoint_rejects_unsupported_horizon(tmp_path: Path) -> None:
    """Unsupported horizons produce a clear 400 response."""
    api.model_dir = tmp_path / "models"
    client = TestClient(api.app)
    payload = {
        "asset_id": "T1",
        "horizon_hours": 2,
        "observations": [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "active_power_kw": 500.0,
                "wind_speed_mps": 6.0,
                "wind_direction_deg": 120.0,
                "theoretical_power_kwh": 550.0,
            }
        ],
    }

    response = client.post("/forecast", json=payload)

    assert response.status_code == 400
    assert "Unsupported horizon" in response.json()["detail"]
