"""Streamlit dashboard for wind power forecasting.

Designed as a data scientist's exploratory tool with STAR layout:
- Situation: What data we have
- Task: What we want to predict
- Action: Which model & inputs
- Result: Predictions vs ground truth
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("WINDENEGY_DASHBOARD_API_URL", "http://localhost:8765")
DATA_PATH = Path(os.getenv("WINDENEGY_DATA_RAW_PATH", "data/raw")) / "T1.csv"
MODELS_DIR = Path(os.getenv("WINDENEGY_MODEL_ARTIFACTS_PATH", "artifacts/models"))
CAPACITY_KW = 3600.0  # Turbine rated capacity (T1 is ~3.6 MW)


st.set_page_config(
    page_title="Windenegy | Wind Power Forecasting",
    page_icon="W",
    layout="wide",
)


# ============================================================================
# DATA LOADING (cached)
# ============================================================================

@st.cache_data(show_spinner="Loading SCADA dataset...")
def load_scada_data() -> pd.DataFrame:
    """Load the real SCADA training data."""
    if not DATA_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={
        "Date/Time": "timestamp",
        "LV ActivePower (kW)": "active_power_kw",
        "Wind Speed (m/s)": "wind_speed_mps",
        "Theoretical_Power_Curve (KWh)": "theoretical_power_kwh",
        "Wind Direction (\u00b0)": "wind_direction_deg",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d %m %Y %H:%M", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


@st.cache_data
def load_model_registry() -> list[dict[str, Any]]:
    """Discover all trained models and their metadata."""
    models = []
    if not MODELS_DIR.exists():
        return models
    for model_dir in sorted(MODELS_DIR.iterdir()):
        meta_path = model_dir / "metadata.json"
        if meta_path.exists():
            with meta_path.open() as f:
                meta = json.load(f)
                meta["_path"] = str(model_dir)
                models.append(meta)
    return models


def get_api_metadata() -> dict[str, Any]:
    """Fetch live API metadata."""
    try:
        r = requests.get(f"{API_URL}/metadata", timeout=3)
        r.raise_for_status()
        return cast("dict[str, Any]", r.json())
    except requests.RequestException:
        return {}


def get_api_health() -> dict[str, Any] | None:
    """Check API connectivity."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        r.raise_for_status()
        return cast("dict[str, Any]", r.json())
    except requests.RequestException:
        return None


def request_forecast(observations: list[dict[str, Any]], horizon: int) -> dict[str, Any] | None:
    """Call /forecast with a window of real observations."""
    payload = {
        "asset_id": "T1",
        "created_at": datetime.now(UTC).isoformat(),
        "horizon_hours": horizon,
        "observations": observations,
        "weather_forecast": [],
    }
    try:
        r = requests.post(f"{API_URL}/forecast", json=payload, timeout=15)
        r.raise_for_status()
        return cast("dict[str, Any]", r.json())
    except requests.RequestException as exc:
        st.error(f"Forecast API error: {exc}")
        return None


def request_risk(forecast: dict[str, Any]) -> dict[str, Any] | None:
    """Call /risk/assess with a forecast result."""
    points = forecast.get("forecast", [])
    if not points:
        return None
    payload = {
        "asset_id": forecast.get("asset_id", "T1"),
        "timestamps": [p["timestamp"] for p in points],
        "p50_forecast": [p["p50"] for p in points],
        "p10_forecast": [p["p10"] for p in points],
        "p90_forecast": [p["p90"] for p in points],
        "capacity_kw": CAPACITY_KW,
    }
    try:
        r = requests.post(f"{API_URL}/risk/assess", json=payload, timeout=10)
        r.raise_for_status()
        return cast("dict[str, Any]", r.json())
    except requests.RequestException as exc:
        st.error(f"Risk API error: {exc}")
        return None


# ============================================================================
# HEADER
# ============================================================================

st.title("\U0001F32C\ufe0f  Windenegy: Wind Power Forecasting")
st.caption("End-to-end prototype: SCADA \u2192 features \u2192 ML \u2192 P10/P50/P90 forecasts \u2192 risk")

health = get_api_health()
api_meta = get_api_metadata()
df = load_scada_data()
models = load_model_registry()

# Status bar
c1, c2, c3, c4 = st.columns(4)
c1.metric("API", "\u2705 Online" if health else "\u274C Offline", help=API_URL)
c2.metric("Dataset rows", f"{len(df):,}" if not df.empty else "Missing")
c3.metric("Trained models", len(models))
active = api_meta.get("active_model", {})
c4.metric("Active model", active.get("model_type", "persistence"))

st.divider()


# ============================================================================
# TABS
# ============================================================================

tab_data, tab_models, tab_predict, tab_risk, tab_eval = st.tabs([
    "\U0001F4CA Data (EDA)",
    "\U0001F9E0 Models",
    "\U0001F52E Predict",
    "\u26A0\ufe0f Risk",
    "\U0001F4C8 Evaluation",
])


# ----------------------------------------------------------------------------
# TAB 1: DATA (Exploratory Data Analysis)
# ----------------------------------------------------------------------------

with tab_data:
    if df.empty:
        st.error(f"SCADA file not found at {DATA_PATH}. Run scripts/download_scada.py.")
    else:
        # Situation block
        with st.container(border=True):
            st.markdown("### \U0001F4CD Situation")
            cols = st.columns(4)
            cols[0].metric("Records", f"{len(df):,}")
            cols[1].metric("Date range", f"{(df.timestamp.max() - df.timestamp.min()).days} days")
            cols[2].metric("Sample rate", "10 min")
            cols[3].metric("Capacity factor", f"{df.active_power_kw.mean()/CAPACITY_KW*100:.1f}%")
            st.caption(
                f"Source: Kaggle Turkey Wind Turbine SCADA \u00b7 "
                f"Period: {df.timestamp.min():%Y-%m-%d} to {df.timestamp.max():%Y-%m-%d} \u00b7 "
                f"Asset: T1"
            )

        # Task block
        with st.container(border=True):
            st.markdown("### \U0001F3AF Task")
            st.markdown(
                """
                **Predict turbine active power output (kW)** N steps ahead from recent SCADA observations.

                | Input | \u2192 | Model | \u2192 | Output |
                |-------|---|-------|---|--------|
                | Wind, direction, theoretical, lags | | Persistence / GBM / PatchTST | | P10/P50/P90 (1\u201324h) |
                """
            )

        # Action: explore data
        st.markdown("### \U0001F50D Action: Explore the data")

        # Sample window selector
        col_a, col_b = st.columns([1, 3])
        with col_a:
            sample_size = st.selectbox("Sample window", [144, 432, 1008, 4320], index=1,
                                       format_func=lambda x: f"{x} obs ({x*10/60:.0f}h)")
            start_idx = st.slider("Start index", 0, max(0, len(df) - sample_size), len(df)//2, step=144)
        with col_b:
            sample = df.iloc[start_idx:start_idx + sample_size].copy()
            ts_min = f"{sample.timestamp.min():%Y-%m-%d %H:%M}"
            ts_max = f"{sample.timestamp.max():%Y-%m-%d %H:%M}"
            st.caption(f"Showing {ts_min} \u2192 {ts_max}")

        # Time series chart
        st.markdown("**Time series: power & wind speed**")
        ts_chart = sample.set_index("timestamp")[["active_power_kw", "theoretical_power_kwh"]]
        st.line_chart(ts_chart, height=240)

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Power curve (observed vs theoretical)**")
            scatter = sample[["wind_speed_mps", "active_power_kw", "theoretical_power_kwh"]].copy()
            scatter.columns = ["wind_speed", "observed", "theoretical"]
            st.scatter_chart(
                scatter.melt("wind_speed", var_name="series", value_name="power"),
                x="wind_speed", y="power", color="series", height=300,
            )
            st.caption("Cut-in ~3 m/s, rated power plateau, cut-out ~25 m/s")

        with col_r:
            st.markdown("**Wind speed distribution**")
            hist_df = pd.DataFrame({
                "wind_speed_bin": pd.cut(sample.wind_speed_mps, bins=20).astype(str),
                "count": 1,
            }).groupby("wind_speed_bin").count().reset_index()
            st.bar_chart(hist_df.set_index("wind_speed_bin"), height=300)
            st.caption(f"Mean: {sample.wind_speed_mps.mean():.2f} m/s \u00b7 Std: {sample.wind_speed_mps.std():.2f}")

        # Stats and raw data
        col_stats, col_raw = st.columns([1, 1])
        with col_stats:
            st.markdown("**Descriptive statistics**")
            st.dataframe(
                sample[["active_power_kw", "wind_speed_mps", "theoretical_power_kwh", "wind_direction_deg"]]
                .describe().round(2),
                use_container_width=True,
            )
        with col_raw:
            st.markdown("**Sample observations (head)**")
            display_sample = sample.head(10).copy()
            num_cols = display_sample.select_dtypes(include=["number"]).columns
            display_sample[num_cols] = display_sample[num_cols].round(2)
            st.dataframe(display_sample, use_container_width=True, height=300)


# ----------------------------------------------------------------------------
# TAB 2: MODELS
# ----------------------------------------------------------------------------

with tab_models:
    st.markdown("### \U0001F9E0 Available Models")
    st.caption("Trained model registry with architecture, features, and test metrics.")

    if not models:
        st.warning("No trained models found. Run scripts/train_*.py first.")
    else:
        # Comparison table
        st.markdown("#### Performance comparison (test set)")
        comp_rows = []
        for m in models:
            metrics = m.get("metrics", {})
            comp_rows.append({
                "Model": m.get("model_type", "?"),
                "Version": m.get("model_version", "?"),
                "Horizon (h)": m.get("horizon_hours", "?"),
                "MAE (kW)": round(metrics.get("mae", 0), 1),
                "RMSE (kW)": round(metrics.get("rmse", 0), 1),
                "Skill score": round(metrics.get("skill_score", 0), 3),
                "P90 coverage": f"{metrics.get('coverage_p90', 0)*100:.1f}%",
            })
        comp_df = pd.DataFrame(comp_rows)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        st.caption(
            "**Skill score**: improvement over persistence baseline (>0 is better). "
            "**P90 coverage**: fraction of actuals inside the P10\u2013P90 interval (target: 80%)."
        )

        st.divider()

        # Per-model detail
        st.markdown("#### Architecture detail")
        model_names = [m.get("model_version", f"model_{i}") for i, m in enumerate(models)]
        selected = st.radio("Select model", model_names, horizontal=True)
        chosen = next((m for m in models if m.get("model_version") == selected), models[0])

        col_arch, col_feat = st.columns([1, 1])
        with col_arch:
            st.markdown("**Schematic**")
            mtype = chosen.get("model_type", "")
            if mtype == "gradient_boosting":
                st.code(
                    """
SCADA observations (N \u2265 25 obs)
        \u2193
[ Feature engineering ]
  \u2022 Time: hour_sin/cos, day_sin/cos
  \u2022 Direction: dir_sin/cos
  \u2022 Lags: power & wind @ {1,3,6,12,24}
  \u2022 Rolling means: {3,6,12} window
        \u2193
   25 features  \u2192  GradientBoostingRegressor
                     (sklearn, n_estimators=200)
        \u2193
   Point forecast (kW)
        \u2193
   + residual P90 \u2192 P10/P50/P90 intervals
                    """,
                    language="text",
                )
            elif mtype == "patchtst":
                st.code(
                    """
24h sequence window (144 timesteps \u00d7 5 features)
        \u2193
[ Patching: split into 8-step patches ]
        \u2193
[ MLP encoder per patch ]  (channel-independent)
        \u2193
[ Aggregation \u2192 dense head ]
        \u2193
   Multi-step forecast (6 \u00d7 10min = 1h)
        \u2193
   + Conformal residuals \u2192 P10/P90 bands
                    """,
                    language="text",
                )
            else:
                st.code(f"Model type: {mtype}\n(no schematic available)", language="text")

        with col_feat:
            st.markdown("**Configuration**")
            st.json({
                "type": chosen.get("model_type"),
                "horizon_hours": chosen.get("horizon_hours"),
                "target": chosen.get("target"),
                "trained_at": chosen.get("training_timestamp"),
                "config": chosen.get("config_snapshot", {}),
            })

        st.markdown("**Feature schema**")
        feature_schema = chosen.get("feature_schema", {})
        if feature_schema:
            feat_df = pd.DataFrame(
                [{"feature": k, "dtype": v} for k, v in feature_schema.items()]
            )
            st.dataframe(feat_df, use_container_width=True, hide_index=True, height=240)

        st.markdown("**Test metrics**")
        st.json(chosen.get("metrics", {}))


# ----------------------------------------------------------------------------
# TAB 3: PREDICT (real data \u2192 forecast \u2192 compare to actual)
# ----------------------------------------------------------------------------

with tab_predict:
    if df.empty:
        st.error("Dataset required to run predictions.")
    else:
        # Situation block
        with st.container(border=True):
            st.markdown("### \U0001F4CD Situation")
            st.markdown(
                "Pick any window from the SCADA history. We feed it to the API, "
                "and compare the forecast against the **actual future values** (already in the data)."
            )

        # Task block
        with st.container(border=True):
            st.markdown("### \U0001F3AF Task: Forecast N hours ahead from input window")
            cc = st.columns(3)
            with cc[0]:
                horizon = st.selectbox(
                    "Horizon (hours)", [1, 6, 12, 24], index=1,
                    help="Longer horizons show more interesting forecast trajectories.",
                )
            with cc[1]:
                input_window_hours = st.selectbox(
                    "Input window (hours)", [3, 6, 12, 24], index=2,
                    help="GBM model needs ≥ 25 observations (4h+) to build lag features.",
                )
            with cc[2]:
                window_obs = input_window_hours * 6  # 10-min cadence
                horizon_obs = horizon * 6
                # Need room for input + horizon + buffer
                max_start = max(0, len(df) - window_obs - horizon_obs - 1)
                start = st.slider("Window start (row)", 0, max_start, max_start // 2, step=144)

        # Action: prepare inputs & call API
        with st.container(border=True):
            st.markdown("### \u2699\ufe0f Action: Send observations to API")

            input_slice = df.iloc[start:start + window_obs].copy()
            actual_future = df.iloc[start + window_obs:start + window_obs + horizon_obs].copy()

            observations = [
                {
                    "timestamp": row.timestamp.isoformat(),
                    "active_power_kw": float(row.active_power_kw),
                    "wind_speed_mps": float(row.wind_speed_mps),
                    "wind_direction_deg": float(row.wind_direction_deg),
                    "theoretical_power_kwh": float(row.theoretical_power_kwh),
                }
                for row in input_slice.itertuples()
            ]

            ac = st.columns(3)
            ac[0].metric("Input observations", len(observations))
            in_range = f"{input_slice.timestamp.min():%H:%M} \u2192 {input_slice.timestamp.max():%H:%M}"
            ac[1].metric("Input range", in_range)
            ac[2].metric("Forecast horizon", f"{horizon}h ({horizon_obs} pts)")

            with st.expander("View input payload (first 3 obs)"):
                st.json(observations[:3])

            forecast = request_forecast(observations, horizon)

        # Result block
        with st.container(border=True):
            st.markdown("### \U0001F4C8 Result: Forecast vs actual")

            if forecast is None:
                st.error("No forecast returned.")
            else:
                fc_df = pd.DataFrame(forecast["forecast"])
                fc_df["timestamp"] = pd.to_datetime(fc_df["timestamp"])
                forecast_start = fc_df["timestamp"].min()

                # Show only last 12 obs (2h) of history so forecast is visible
                hist_display = input_slice.tail(12)[["timestamp", "active_power_kw"]].copy()
                hist_display["series"] = "History"

                # Prepare actuals (ground truth) for the forecast window
                fut_display = (
                    actual_future[["timestamp", "active_power_kw"]].copy()
                    if not actual_future.empty
                    else pd.DataFrame()
                )
                if not fut_display.empty:
                    fut_display["series"] = "Actual (held back)"

                # Build Altair chart: uncertainty band + lines + now marker
                try:
                    import altair as alt

                    # Uncertainty band (P10-P90)
                    band = (
                        alt.Chart(fc_df)
                        .mark_area(opacity=0.3, color="#e74c3c")
                        .encode(
                            x=alt.X("timestamp:T", title="Time"),
                            y=alt.Y("p10:Q", title="Power (kW)", scale=alt.Scale(zero=False)),
                            y2="p90:Q",
                            tooltip=["timestamp:T", "p10:Q", "p50:Q", "p90:Q"],
                        )
                    )

                    # Forecast P50 line
                    forecast_line = (
                        alt.Chart(fc_df)
                        .mark_line(color="#c0392b", strokeWidth=3)
                        .encode(x="timestamp:T", y="p50:Q", tooltip=["timestamp:T", "p50:Q"])
                    )

                    # History line (last 2h)
                    hist_line = (
                        alt.Chart(hist_display)
                        .mark_line(color="#3498db", strokeWidth=2)
                        .encode(x="timestamp:T", y="active_power_kw:Q")
                    )

                    # Actual future line (ground truth)
                    actual_line = None
                    if not fut_display.empty:
                        actual_line = (
                            alt.Chart(fut_display)
                            .mark_line(color="#27ae60", strokeWidth=2, strokeDash=[5, 5])
                            .encode(x="timestamp:T", y="active_power_kw:Q")
                        )

                    # "Now" vertical rule at forecast start
                    now_rule = (
                        alt.Chart(pd.DataFrame({"now": [forecast_start]}))
                        .mark_rule(color="#7f8c8d", strokeDash=[3, 3])
                        .encode(x="now:T")
                    )

                    # Combine layers
                    layers = [band, forecast_line, hist_line]
                    if actual_line:
                        layers.append(actual_line)
                    layers.append(now_rule)

                    chart = alt.layer(*layers).properties(
                        width="container",
                        height=320,
                    ).interactive()

                    st.altair_chart(chart, use_container_width=True)
                    st.caption(
                        "**History** (blue, last 2h) \u2192 **Now** (dashed line) \u2192 "
                        "**Forecast P50** (red) with **P10\u2013P90 band** (shaded) \u00b7 "
                        "**Actual** (green dashed) = ground truth"
                    )
                except Exception as e:
                    # Fallback to simple line chart if altair fails
                    st.error(f"Chart error: {e}")
                    st.line_chart(fc_df.set_index("timestamp")[["p10", "p50", "p90"]], height=320)

                # Quality metrics
                if not actual_future.empty:
                    aligned = pd.merge(
                        fc_df[["timestamp", "p10", "p50", "p90"]],
                        actual_future[["timestamp", "active_power_kw"]],
                        on="timestamp", how="inner",
                    )
                    if not aligned.empty:
                        residuals = aligned["active_power_kw"] - aligned["p50"]
                        mae = float(np.abs(residuals).mean())
                        rmse = float(np.sqrt((residuals**2).mean()))
                        coverage = float(
                            ((aligned["active_power_kw"] >= aligned["p10"])
                             & (aligned["active_power_kw"] <= aligned["p90"])).mean()
                        )
                        rc = st.columns(4)
                        rc[0].metric("MAE", f"{mae:.0f} kW")
                        rc[1].metric("RMSE", f"{rmse:.0f} kW")
                        rc[2].metric("P10\u2013P90 coverage", f"{coverage*100:.0f}%")
                        rc[3].metric("Model used", forecast.get("model_version", "?"))

                        st.markdown("**Residuals (actual \u2212 P50)**")
                        resid_df = aligned[["timestamp"]].copy()
                        resid_df["residual_kw"] = residuals.values
                        st.line_chart(resid_df.set_index("timestamp"), height=180)

                # Warnings
                warns = forecast.get("warnings", [])
                if warns:
                    for w in warns:
                        st.warning(w)

                # Save state for risk tab
                st.session_state["last_forecast"] = forecast


# ----------------------------------------------------------------------------
# TAB 4: RISK
# ----------------------------------------------------------------------------

with tab_risk:
    forecast = st.session_state.get("last_forecast")

    if forecast is None:
        st.info("Generate a forecast in the **Predict** tab first.")
    else:
        with st.container(border=True):
            st.markdown("### \U0001F4CD Situation")
            st.caption(
                f"Risk assessment based on the latest forecast: model={forecast.get('model_version')}, "
                f"asset={forecast.get('asset_id')}, points={len(forecast.get('forecast', []))}"
            )

        with st.container(border=True):
            st.markdown("### \U0001F3AF Task")
            st.markdown(
                "Detect operational risks from forecast trajectory:\n"
                "1. **Ramp events** \u2014 rapid power swings that stress the grid\n"
                "2. **Underproduction risk** \u2014 high probability of falling below contracted output"
            )

        risk_data = request_risk(forecast)
        if risk_data is None:
            st.error("Could not get risk assessment.")
        else:
            with st.container(border=True):
                st.markdown("### \U0001F4C8 Result")

                ramp_col, under_col = st.columns(2)

                with ramp_col:
                    st.markdown("#### \U0001F30A Ramp events")
                    ramps = risk_data.get("ramp_events", [])
                    if ramps:
                        ramp_rows = [
                            {
                                "Time": r["timestamp"][:16],
                                "Direction": "\u2191 up" if r["direction"] == "up" else "\u2193 down",
                                "\u0394 Power (kW)": round(r["power_change_kw"], 0),
                                "\u0394 (%)": round(r["power_change_pct"], 1),
                                "Duration (min)": r["duration_minutes"],
                                "Risk": r.get("risk_level", "low"),
                            }
                            for r in ramps
                        ]
                        st.dataframe(pd.DataFrame(ramp_rows), hide_index=True,
                                     use_container_width=True)
                        st.caption(
                            "A *ramp* is a power change \u2265 10% of capacity within 1 hour. "
                            "Severity scales with magnitude: medium \u2265 10%, high \u2265 25%, critical \u2265 50%."
                        )
                    else:
                        st.success("No significant ramp events detected.")

                with under_col:
                    st.markdown("#### \u26A1 Underproduction risk")
                    risks = risk_data.get("underproduction_risks", [])
                    if risks:
                        rows = [
                            {
                                "Time": r["timestamp"][:16],
                                "Expected (kW)": round(r["expected_power_kw"], 0),
                                "Shortfall (kW)": round(r["expected_shortfall_kw"], 0),
                                "P(shortfall)": f"{r['shortfall_probability']:.0%}",
                                "Risk": r.get("risk_level", "low"),
                            }
                            for r in risks[:10]
                        ]
                        st.dataframe(pd.DataFrame(rows), hide_index=True,
                                     use_container_width=True)
                        st.caption(
                            "Probability the P10 forecast falls below 70% of capacity, "
                            "weighted by interval width (uncertainty)."
                        )
                    else:
                        st.success("No underproduction risks identified.")


# ----------------------------------------------------------------------------
# TAB 5: EVALUATION (Backtest: predictions vs actuals, error over time)
# ----------------------------------------------------------------------------

with tab_eval:
    st.markdown("### \U0001F4CA Model Evaluation: Walk-forward Backtest")
    st.caption(
        "Run multiple models on the same test window to compare predictions vs actuals "
        "and track error evolution over time."
    )

    if df.empty:
        st.error("Dataset required for evaluation.")
    else:
        # Controls
        with st.container(border=True):
            st.markdown("#### \u2699\ufe0f Configure backtest")
            ec = st.columns(4)
            with ec[0]:
                eval_horizon = st.selectbox(
                    "Forecast horizon", [1, 6], index=0, key="eval_h",
                    help="1h uses GBM; 6h uses persistence (GBM only trained for 1h).",
                )
            with ec[1]:
                eval_window_hours = st.selectbox(
                    "Test window (hours)", [6, 12, 24, 48], index=1, key="eval_w",
                    help="Longer windows show more error accumulation patterns.",
                )
            with ec[2]:
                max_eval_start = max(0, len(df) - eval_window_hours * 6 - eval_horizon * 6 - 1)
                # Find an active (non-zero) window for default
                power_series = df["active_power_kw"].rolling(72).mean()
                active_idx = power_series[power_series > 500].index.tolist()
                default_start = (
                    int(active_idx[len(active_idx) // 2])
                    if active_idx
                    else max_eval_start // 2
                )
                default_start = min(default_start, max_eval_start)
                eval_start = st.slider(
                    "Test start (row)", 0, max_eval_start, default_start,
                    step=144, key="eval_s",
                    help="Defaults to an active (windy) period for meaningful results.",
                )
            with ec[3]:
                st.markdown("**Models to compare**")
                compare_persistence = st.checkbox("Persistence baseline", value=True)
                compare_gbm = st.checkbox("Gradient Boosting (if 1h)",
                                           value=(eval_horizon == 1))

        # Run backtest button
        if st.button("\u25b6\ufe0f Run backtest", type="primary"):
            with st.spinner("Running walk-forward backtest..."):
                test_slice = df.iloc[
                    eval_start:eval_start + eval_window_hours * 6
                ].copy()

                results = []
                step_obs = 36  # 6h input window → GBM has enough lags (needs ≥25)

                for i in range(0, len(test_slice) - step_obs - eval_horizon * 6, eval_horizon * 6):
                    input_win = test_slice.iloc[i:i + step_obs]
                    actual_win = test_slice.iloc[
                        i + step_obs:i + step_obs + eval_horizon * 6
                    ]
                    if len(actual_win) < eval_horizon * 6:
                        continue

                    observations = [
                        {
                            "timestamp": row.timestamp.isoformat(),
                            "active_power_kw": float(row.active_power_kw),
                            "wind_speed_mps": float(row.wind_speed_mps),
                            "wind_direction_deg": float(row.wind_direction_deg),
                            "theoretical_power_kwh": float(row.theoretical_power_kwh),
                        }
                        for row in input_win.itertuples()
                    ]

                    # API call (returns GBM if available, else persistence)
                    if compare_gbm:
                        fc = request_forecast(observations, eval_horizon)
                        if fc:
                            model_ver = fc.get("model_version", "unknown")
                            for p, actual in zip(
                                fc.get("forecast", []),
                                actual_win.itertuples(),
                                strict=False,
                            ):
                                results.append({
                                    "timestamp": pd.to_datetime(p["timestamp"]),
                                    "actual_kw": float(actual.active_power_kw),
                                    "predicted_kw": float(p["p50"]),
                                    "p10": float(p["p10"]),
                                    "p90": float(p["p90"]),
                                    "model": model_ver,
                                    "error_kw": float(actual.active_power_kw) - float(p["p50"]),
                                })

                    # Local persistence baseline (last value carried forward)
                    if compare_persistence:
                        last_power = float(input_win["active_power_kw"].iloc[-1])
                        # Estimate residual from input window variance
                        residual = float(
                            np.percentile(
                                np.abs(np.diff(input_win["active_power_kw"].to_numpy())), 90
                            )
                        ) * 2 if len(input_win) > 1 else 200.0
                        for actual in actual_win.itertuples():
                            results.append({
                                "timestamp": actual.timestamp,
                                "actual_kw": float(actual.active_power_kw),
                                "predicted_kw": last_power,
                                "p10": max(0.0, last_power - residual),
                                "p90": last_power + residual,
                                "model": "persistence-local",
                                "error_kw": float(actual.active_power_kw) - last_power,
                            })

            if not results:
                st.warning("No predictions generated. Check model availability.")
            else:
                results_df = pd.DataFrame(results)

                # Chart 1: Predictions vs Actuals
                with st.container(border=True):
                    st.markdown("#### \U0001F4C8 Chart 1: Predictions vs Actuals")

                    try:
                        import altair as alt

                        actual_line = (
                            alt.Chart(results_df.drop_duplicates("timestamp"))
                            .mark_line(color="#27ae60", strokeWidth=3)
                            .encode(x="timestamp:T", y="actual_kw:Q", tooltip=["timestamp:T", "actual_kw:Q"])
                        )

                        pred_lines = alt.Chart(results_df).mark_line(strokeWidth=2).encode(
                            x="timestamp:T",
                            y="predicted_kw:Q",
                            color=alt.Color(
                                "model:N",
                                scale=alt.Scale(
                                    domain=[
                                        "persistence-local",
                                        "persistence-baseline-0.1.0",
                                        "gradient-boosting-T1-1h",
                                    ],
                                    range=["#e74c3c", "#f39c12", "#3498db"],
                                ),
                                legend=alt.Legend(title="Model"),
                            ),
                            tooltip=["timestamp:T", "model:N", "predicted_kw:Q"],
                        )

                        pred_vs_actual = alt.layer(actual_line, pred_lines).properties(
                            width="container", height=350,
                        ).interactive()

                        st.altair_chart(pred_vs_actual, use_container_width=True)
                        st.caption(
                            "**Green** = actual (ground truth) \u00b7 "
                            "**Red/Blue** = model predictions (P50)"
                        )
                    except Exception as e:
                        st.error(f"Chart error: {e}")
                        pivoted = results_df.pivot_table(
                            index="timestamp", columns="model", values="predicted_kw"
                        )
                        pivoted["actual"] = results_df.drop_duplicates("timestamp").set_index(
                            "timestamp"
                        )["actual_kw"]
                        st.line_chart(pivoted, height=350)

                # Chart 2: Error over time
                with st.container(border=True):
                    st.markdown("#### \U0001F4C9 Chart 2: Error Over Time (Actual \u2212 Predicted)")

                    try:
                        error_chart = alt.Chart(results_df).mark_line(strokeWidth=2).encode(
                            x="timestamp:T",
                            y="error_kw:Q",
                            color=alt.Color(
                                "model:N",
                                scale=alt.Scale(
                                    domain=[
                                        "persistence-local",
                                        "persistence-baseline-0.1.0",
                                        "gradient-boosting-T1-1h",
                                    ],
                                    range=["#e74c3c", "#f39c12", "#3498db"],
                                ),
                                legend=alt.Legend(title="Model"),
                            ),
                            tooltip=["timestamp:T", "model:N", "error_kw:Q"],
                        ).properties(width="container", height=280).interactive()

                        zero_line = (
                            alt.Chart(pd.DataFrame({"y": [0]}))
                            .mark_rule(color="#7f8c8d", strokeDash=[3, 3])
                            .encode(y="y:Q")
                        )

                        st.altair_chart(alt.layer(error_chart, zero_line), use_container_width=True)
                        st.caption("Error = actual \u2212 predicted. Closer to zero line = better.")
                    except Exception as e:
                        st.error(f"Chart error: {e}")
                        error_pivot = results_df.pivot_table(
                            index="timestamp", columns="model", values="error_kw"
                        )
                        st.line_chart(error_pivot, height=280)

                # Metrics summary
                with st.container(border=True):
                    st.markdown("#### \U0001F4CB Backtest Metrics Summary")
                    metrics = []
                    for model_name, group in results_df.groupby("model"):
                        metrics.append({
                            "Model": model_name,
                            "MAE (kW)": round(group["error_kw"].abs().mean(), 1),
                            "RMSE (kW)": round(np.sqrt((group["error_kw"] ** 2).mean()), 1),
                            "Max error (kW)": round(group["error_kw"].abs().max(), 1),
                            "Bias (kW)": round(group["error_kw"].mean(), 1),
                            "Coverage": round(
                                ((group["actual_kw"] >= group["p10"])
                                 & (group["actual_kw"] <= group["p90"])).mean() * 100, 1
                            ),
                        })
                    st.dataframe(pd.DataFrame(metrics), hide_index=True, use_container_width=True)


st.divider()
st.caption(f"Windenegy v0.1.0 \u00b7 API: {API_URL} \u00b7 [Docs](/docs)")
