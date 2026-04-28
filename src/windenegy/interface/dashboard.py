"""Streamlit dashboard for wind power forecasting."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any, cast

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("WINDENEGY_DASHBOARD_API_URL", "http://localhost:8765")


def _sample_observations() -> list[dict[str, Any]]:
    """Build a small deterministic observation sequence for the demo view."""
    timestamps = pd.date_range("2018-01-01T00:00:00Z", periods=18, freq="10min")
    observations = []
    for index, timestamp in enumerate(timestamps):
        observations.append(
            {
                "timestamp": timestamp.isoformat(),
                "active_power_kw": 420.0 + index * 8.0,
                "wind_speed_mps": 5.5 + index * 0.04,
                "wind_direction_deg": 110.0,
                "theoretical_power_kwh": 470.0 + index * 9.0,
            }
        )
    return observations


def _request_forecast(horizon_hours: int) -> dict[str, Any] | None:
    """Call the API forecast endpoint."""
    payload = {
        "asset_id": "T1",
        "created_at": datetime.now(UTC).isoformat(),
        "horizon_hours": horizon_hours,
        "observations": _sample_observations(),
        "weather_forecast": [],
    }
    try:
        response = requests.post(f"{API_URL}/forecast", json=payload, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Forecast request failed: {exc}")
        return None
    return cast("dict[str, Any]", response.json())


st.set_page_config(
    page_title="Windenegy Dashboard",
    page_icon="W",
    layout="wide",
)

st.title("Windenegy Wind Power Forecasting")
st.caption("Operational forecast view for weather-aware renewable energy decisions.")

status_col, version_col = st.columns(2)
try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    with status_col:
        st.success("API connected")
        st.caption(f"Status: {health['status']}")
except requests.RequestException:
    with status_col:
        st.error("API unavailable")
        st.caption(f"Could not connect to {API_URL}")

with version_col:
    st.info("Dashboard v0.1.0")

def _request_risk_assessment(forecast: dict[str, Any]) -> dict[str, Any] | None:
    """Call the API risk assessment endpoint."""
    points = forecast.get("forecast", [])
    if not points:
        return None

    payload = {
        "asset_id": forecast.get("asset_id", "T1"),
        "timestamps": [p["timestamp"] for p in points],
        "p50_forecast": [p["p50"] for p in points],
        "p10_forecast": [p["p10"] for p in points],
        "p90_forecast": [p["p90"] for p in points],
        "capacity_kw": 2000.0,
    }
    try:
        response = requests.post(f"{API_URL}/risk/assess", json=payload, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Risk assessment failed: {exc}")
        return None
    return cast("dict[str, Any]", response.json())


forecast_tab, risk_tab, quality_tab, config_tab = st.tabs(["Forecast", "Risk", "Data Quality", "Configuration"])

with forecast_tab:
    horizon = st.radio("Forecast horizon", options=[1, 6, 24], index=1, horizontal=True)
    forecast = _request_forecast(horizon)

    if forecast is not None:
        points = pd.DataFrame(forecast["forecast"])
        points["timestamp"] = pd.to_datetime(points["timestamp"])
        st.subheader("Power forecast")
        st.line_chart(points.set_index("timestamp")[["p10", "p50", "p90"]])

        metric_col, warning_col = st.columns(2)
        with metric_col:
            st.metric("Forecast points", len(points))
            st.metric("Model", forecast["model_version"])
        with warning_col:
            for warning in forecast.get("warnings", []):
                st.warning(warning)

with risk_tab:
    st.subheader("Operational Risk Assessment")
    if forecast is not None:
        risk_data = _request_risk_assessment(forecast)
        if risk_data is not None:
            ramp_col, under_col = st.columns(2)

            with ramp_col:
                st.markdown("#### Ramp Events")
                ramps = risk_data.get("ramp_events", [])
                if ramps:
                    for ramp in ramps:
                        level = ramp.get("risk_level", "low")
                        icon = "🔴" if level in ["critical", "high"] else "🟡" if level == "medium" else "🟢"
                        st.write(
                            f"{icon} **{ramp['direction'].upper()} ramp** ({ramp['duration_minutes']}min)"
                        )
                        st.caption(
                            f"Change: {ramp['power_change_kw']:.0f}kW ({ramp['power_change_pct']:.1f}%)"
                        )
                else:
                    st.info("No significant ramp events detected")

            with under_col:
                st.markdown("#### Underproduction Risk")
                risks = risk_data.get("underproduction_risks", [])
                if risks:
                    for risk in risks[:5]:  # Show top 5
                        level = risk.get("risk_level", "low")
                        icon = (
                            "🔴" if level in ["critical", "high"]
                            else "🟡" if level == "medium"
                            else "🟢"
                        )
                        st.write(
                            f"{icon} **{level.upper()}** at {risk['timestamp'][:16]}"
                        )
                        st.caption(
                            f"Expected: {risk['expected_power_kw']:.0f}kW "
                            f"(shortfall: {risk['expected_shortfall_kw']:.0f}kW, "
                            f"p={risk['shortfall_probability']:.0%})"
                        )
                else:
                    st.success("No underproduction risks identified")
    else:
        st.info("Generate a forecast first to see risk assessment")

with quality_tab:
    st.subheader("Data quality checks")
    checks = pd.DataFrame(
        {
            "Check": [
                "Required schema",
                "Chronological timestamps",
                "Physical power range",
                "Wind direction range",
            ],
            "Status": ["Expected", "Expected", "Expected", "Expected"],
            "Action": [
                "Validate raw CSV before training",
                "Sort and deduplicate on ingestion",
                "Reject or flag negative power",
                "Modulo-normalize direction",
            ],
        }
    )
    st.dataframe(checks, width="stretch")

with config_tab:
    st.subheader("Runtime configuration")
    st.json(
        {
            "api_url": API_URL,
            "supported_horizons": [1, 6, 24],
            "weather_provider": "null",
            "current_model": "persistence-baseline-0.1.0",
        }
    )

st.caption("Windenegy: climate intelligence for renewable energy operations.")
