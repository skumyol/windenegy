"""Integration tests for the training and serving pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from windenegy.application.training import (
    GradientBoostingPowerModel,
    load_latest_gradient_boosting,
    train_gradient_boosting_from_csv,
)
from windenegy.interface import api

pytestmark = pytest.mark.integration


def test_train_persist_reload_and_forecast(tmp_path: Path) -> None:
    """A raw CSV can produce a persisted model that can forecast from observations."""
    csv_path = _write_synthetic_scada(tmp_path)

    result = train_gradient_boosting_from_csv(
        csv_path=csv_path,
        model_dir=tmp_path / "models",
        metrics_dir=tmp_path / "metrics",
        horizon_hours=1,
    )
    loaded = load_latest_gradient_boosting(tmp_path / "models")

    assert result.metrics_path.exists()
    assert loaded is not None
    model, metadata = loaded
    assert isinstance(model, GradientBoostingPowerModel)
    assert metadata.metrics is not None
    assert metadata.metrics.mae >= 0.0


def test_api_uses_trained_model_when_available(tmp_path: Path) -> None:
    """The API serves the trained artifact when its horizon matches the request."""
    csv_path = _write_synthetic_scada(tmp_path)
    train_gradient_boosting_from_csv(
        csv_path=csv_path,
        model_dir=tmp_path / "models",
        metrics_dir=tmp_path / "metrics",
        horizon_hours=1,
    )
    api.model_dir = tmp_path / "models"
    client = TestClient(api.app)

    response = client.post(
        "/forecast",
        json={
            "asset_id": "T1",
            "horizon_hours": 1,
            "observations": _observation_payload(csv_path),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"].startswith("gradient-boosting")
    assert len(body["forecast"]) == 6


def _write_synthetic_scada(tmp_path: Path, rows: int = 96) -> Path:
    """Write a deterministic SCADA-shaped CSV."""
    timestamps = pd.date_range("2018-01-01T00:00:00Z", periods=rows, freq="10min")
    frame = pd.DataFrame(
        {
            "Date/Time": timestamps,
            "LV ActivePower (kW)": [300.0 + index * 4.0 for index in range(rows)],
            "Wind Speed (m/s)": [5.0 + index * 0.02 for index in range(rows)],
            "Theoretical_Power_Curve (KWh)": [340.0 + index * 4.2 for index in range(rows)],
            "Wind Direction (°)": [110.0 + (index % 10) for index in range(rows)],
        }
    )
    csv_path = tmp_path / "synthetic_scada.csv"
    frame.to_csv(csv_path, index=False)
    return csv_path


def _observation_payload(csv_path: Path) -> list[dict[str, object]]:
    """Return enough recent observations for model feature generation."""
    frame = pd.read_csv(csv_path).tail(32)
    return [
        {
            "timestamp": row["Date/Time"],
            "active_power_kw": float(row["LV ActivePower (kW)"]),
            "wind_speed_mps": float(row["Wind Speed (m/s)"]),
            "theoretical_power_kwh": float(row["Theoretical_Power_Curve (KWh)"]),
            "wind_direction_deg": float(row["Wind Direction (°)"]),
        }
        for _, row in frame.iterrows()
    ]
