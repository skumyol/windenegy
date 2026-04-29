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

from windenegy.application.training import forecast_points_from_model
from windenegy.domain.models import TurbineObservation
from windenegy.infrastructure.persistence import FileSystemModelRepository

API_URL = os.getenv("WINDENEGY_DASHBOARD_API_URL", "http://localhost:8765")
DATA_PATH = Path(os.getenv("WINDENEGY_DATA_RAW_PATH", "data/raw")) / "T1.csv"
MODELS_DIR = Path(os.getenv("WINDENEGY_MODEL_ARTIFACTS_PATH", "artifacts/models"))
METRICS_DIR = Path(os.getenv("WINDENEGY_MODEL_METRICS_PATH", "artifacts/metrics"))
CAPACITY_KW = 3600.0  # Turbine rated capacity (T1 is ~3.6 MW)
MODEL_COLORS = ["#c0392b", "#f39c12", "#8e44ad", "#16a085", "#2c3e50"]


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


@st.cache_data
def load_test_outputs(model_version: str | None) -> pd.DataFrame:
    """Load persisted one-to-one test predictions for a trained model."""
    if not model_version:
        return pd.DataFrame()
    path = METRICS_DIR / f"{model_version}_test_outputs.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


@st.cache_data
def load_model_comparison() -> dict[str, Any]:
    """Load the unified model-comparison artifact written by train_all.py.

    Returns a dict with keys:
        - ``generated_at``: ISO timestamp.
        - ``split``: split metadata.
        - ``horizons``: list of horizons evaluated.
        - ``results``: list of per-(model, horizon) rows.
    Returns an empty dict if no comparison has been run.
    """
    path = METRICS_DIR / "model_comparison.json"
    if not path.exists():
        return {}
    try:
        with path.open() as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    # Back-compat: old format was a bare list. Wrap it.
    if isinstance(data, list):
        return {"results": data, "split": {}, "horizons": []}
    return cast("dict[str, Any]", data)


@st.cache_data
def load_test_outputs_by_id(test_outputs_path: str | None) -> pd.DataFrame:
    """Load a per-model test-output CSV by its absolute or repo-relative path."""
    if not test_outputs_path:
        return pd.DataFrame()
    path = Path(test_outputs_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


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


def _iso(ts: Any) -> str:
    """Convert datetime / Timestamp to ISO string for JSON."""
    if isinstance(ts, str):
        return ts
    # pandas Timestamp or datetime
    return ts.isoformat()


def request_risk(forecast: dict[str, Any]) -> dict[str, Any] | None:
    """Call /risk/assess with a forecast result."""
    points = forecast.get("forecast", [])
    if not points:
        return None
    payload = {
        "asset_id": forecast.get("asset_id", "T1"),
        "timestamps": [_iso(p["timestamp"]) for p in points],
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


def render_forecast_legend() -> None:
    """Render a stable legend for the forecast chart.

    Altair's automatic legend handling gets brittle once we layer fixed-color
    marks, so we render the legend explicitly to keep the label semantics and
    line styles stable across environments.
    """
    st.markdown(
        """
        <div style="display:flex; flex-wrap:wrap; gap:14px; align-items:center; margin: 0.25rem 0 0.75rem 0;">
          <span style="display:inline-flex; align-items:center; gap:8px;">
            <span style="width:18px; height:0; border-top:3px solid #3498db; display:inline-block;"></span>
            <span>History (blue, last 2h)</span>
          </span>
          <span style="display:inline-flex; align-items:center; gap:8px;">
            <span style="width:18px; height:0; border-top:3px dashed #7f8c8d; display:inline-block;"></span>
            <span>Now (dashed line)</span>
          </span>
          <span style="display:inline-flex; align-items:center; gap:8px;">
            <span style="display:inline-flex; align-items:center; gap:4px;">
              <span style="width:18px; height:0; border-top:3px solid #c0392b; display:inline-block;"></span>
              <span style="width:12px; height:10px; background:rgba(231, 76, 60, 0.22); display:inline-block; border:1px solid rgba(192, 57, 43, 0.25);"></span>
            </span>
            <span>Forecast P50 (red) with P10–P90 band (shaded)</span>
          </span>
          <span style="display:inline-flex; align-items:center; gap:8px;">
            <span style="width:18px; height:0; border-top:3px dashed #27ae60; display:inline-block;"></span>
            <span>Actual (green dashed) = ground truth</span>
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_multi_model_legend(model_series: list[dict[str, str]]) -> None:
    """Render a stable legend for the multi-model overlay."""
    items = [
        """
          <span style="display:inline-flex; align-items:center; gap:8px;">
            <span style="width:18px; height:0; border-top:3px solid #3498db; display:inline-block;"></span>
            <span>History (blue, last 2h)</span>
          </span>
        """,
        """
          <span style="display:inline-flex; align-items:center; gap:8px;">
            <span style="width:18px; height:0; border-top:3px dashed #7f8c8d; display:inline-block;"></span>
            <span>Now (dashed line)</span>
          </span>
        """,
        """
          <span style="display:inline-flex; align-items:center; gap:8px;">
            <span style="width:18px; height:0; border-top:3px dashed #27ae60; display:inline-block;"></span>
            <span>Actual (green dashed) = ground truth</span>
          </span>
        """,
    ]
    for series in model_series:
        items.append(
            f"""
              <span style="display:inline-flex; align-items:center; gap:8px;">
                <span style="width:18px; height:0; border-top:3px solid {series['color']}; display:inline-block;"></span>
                <span>{series['label']}</span>
              </span>
            """
        )
    st.markdown(
        f"""
        <div style="display:flex; flex-wrap:wrap; gap:14px; align-items:center; margin: 0.25rem 0 0.75rem 0;">
          {''.join(items)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_prediction_overlay_frame(
    history_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    model_overlays: list[dict[str, Any]],
) -> pd.DataFrame:
    """Convert history, gold data, and model forecasts into one plotting frame."""
    frames: list[pd.DataFrame] = []

    if not history_df.empty:
        frames.append(
            history_df.loc[:, ["timestamp", "active_power_kw"]]
            .rename(columns={"active_power_kw": "value"})
            .assign(series="Observed history", kind="history")
        )

    if not actual_df.empty:
        frames.append(
            actual_df.loc[:, ["timestamp", "active_power_kw"]]
            .rename(columns={"active_power_kw": "value"})
            .assign(series="Gold data", kind="actual")
        )

    for overlay in model_overlays:
        overlay_df = pd.DataFrame(overlay.get("points", []))
        if overlay_df.empty:
            continue
        overlay_df = overlay_df.loc[:, ["timestamp", "p50"]].copy()
        overlay_df["timestamp"] = pd.to_datetime(overlay_df["timestamp"], utc=True)
        overlay_df = overlay_df.rename(columns={"p50": "value"})
        overlay_df["series"] = overlay["label"]
        overlay_df["kind"] = "model"
        frames.append(overlay_df)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "value", "series", "kind"])

    overlay_df = pd.concat(frames, ignore_index=True)
    overlay_df["timestamp"] = pd.to_datetime(overlay_df["timestamp"], utc=True)
    overlay_df["value"] = pd.to_numeric(overlay_df["value"], errors="coerce")
    return overlay_df.dropna(subset=["timestamp", "value", "series"])


def _build_patchtst_sequence(input_slice: pd.DataFrame, seq_len: int) -> np.ndarray:
    """Build the sequence tensor used by the PatchTST stand-in."""
    feature_cols = [
        "active_power_kw",
        "wind_speed_mps",
        "theoretical_power_kwh",
        "wind_direction_deg",
    ]
    frame = input_slice.tail(seq_len)
    if len(frame) < seq_len:
        msg = f"Need at least {seq_len} observations for PatchTST prediction"
        raise ValueError(msg)
    return frame[feature_cols].to_numpy(dtype=float)


def _baseline_forecast_points(
    input_slice: pd.DataFrame,
    horizon_hours: int,
    step_minutes: float = 10.0,
) -> dict[str, list[dict[str, Any]]]:
    """Generate persistence, power-curve, and rolling-mean forecasts from a window."""
    if input_slice.empty or "timestamp" not in input_slice.columns:
        return {}
    timestamps = pd.to_datetime(input_slice["timestamp"], utc=True)
    step = pd.Timedelta(minutes=step_minutes)
    last_ts = timestamps.iloc[-1]
    horizon_steps = max(1, int(round(horizon_hours * 60 / step_minutes)))
    forecast_ts = [last_ts + step * (i + 1) for i in range(horizon_steps)]

    last_power = float(input_slice["active_power_kw"].iloc[-1])
    last_theoretical = (
        float(input_slice["theoretical_power_kwh"].iloc[-1])
        if "theoretical_power_kwh" in input_slice.columns
        else last_power
    )
    rolling_mean = float(input_slice["active_power_kw"].tail(36).mean())
    # simple residual from recent std
    recent_std = float(input_slice["active_power_kw"].tail(36).std()) if len(input_slice) >= 6 else 200.0
    resid = max(50.0, recent_std * 1.5)

    def _points(value: float) -> list[dict[str, Any]]:
        return [
            {
                "timestamp": ts.isoformat(),
                "p50": round(float(value), 3),
                "p10": round(max(0.0, float(value) - resid), 3),
                "p90": round(float(value) + resid, 3),
            }
            for ts in forecast_ts
        ]

    return {
        "persistence": _points(last_power),
        "power_curve": _points(last_theoretical),
        "rolling_mean": _points(rolling_mean),
    }


def _forecast_from_model_artifact(
    model_meta: dict[str, Any],
    input_slice: pd.DataFrame,
) -> list[dict[str, Any]] | None:
    """Load one saved model artifact and build its forecast points."""
    model_dir = model_meta.get("_path")
    model_version = model_meta.get("model_version")
    model_type = model_meta.get("model_type")
    if not model_dir or not model_version or not model_type:
        return None

    repository = FileSystemModelRepository(Path(model_dir).parent)
    try:
        model, metadata = repository.load_model(model_version)
    except FileNotFoundError:
        return None

    observations = [
        TurbineObservation(
            timestamp=row.timestamp,
            active_power_kw=float(row.active_power_kw),
            wind_speed_mps=float(row.wind_speed_mps),
            wind_direction_deg=float(row.wind_direction_deg),
            theoretical_power_kwh=float(row.theoretical_power_kwh),
        )
        for row in input_slice.itertuples(index=False)
    ]

    if metadata.model_type == "gradient_boosting":
        points = forecast_points_from_model(
            model,
            observations=observations,
            created_at=datetime.now(UTC),
        )
        return [
            {
                "timestamp": point["timestamp"],
                "p50": float(point["p50"]),
                "p10": float(point["p10"]),
                "p90": float(point["p90"]),
            }
            for point in points
        ]

    if metadata.model_type == "patchtst":
        try:
            config_snapshot = metadata.config_snapshot or {}
            seq_len = int(config_snapshot.get("seq_len", 144))
            residual_p90 = float(config_snapshot.get("residual_p90", 200.0))
            pred_len = int(config_snapshot.get("pred_len", 6))

            if len(input_slice) < seq_len:
                return None

            # Handle both dict format and object format
            if isinstance(model, dict):
                regressor = model.get("regressor")
                scaler = model.get("scaler")
                if regressor is None:
                    return None
                # Build sequence and scale it
                sequence = _build_patchtst_sequence(input_slice, seq_len)
                # Flatten to (1, 576) for the regressor
                sequence_flat = sequence.flatten().reshape(1, -1)
                if scaler is not None:
                    try:
                        sequence_flat = scaler.transform(sequence_flat)
                    except Exception:
                        pass
                prediction = np.asarray(regressor.predict(sequence_flat), dtype=float).flatten()
            else:
                # Object format
                sequence = _build_patchtst_sequence(input_slice, seq_len)
                prediction = np.asarray(model.predict(sequence), dtype=float).flatten()

            if prediction.size == 0:
                return None

            if prediction.size == 1 and pred_len > 1:
                prediction = np.repeat(prediction, pred_len)

            timestamps = pd.to_datetime(input_slice["timestamp"], utc=True)
            step = timestamps.iloc[-1] - timestamps.iloc[-2] if len(timestamps) >= 2 else pd.Timedelta(minutes=10)
            start = pd.to_datetime(input_slice["timestamp"].iloc[-1], utc=True)
            return [
                {
                    "timestamp": start + step * (index + 1),
                    "p50": round(float(value), 3),
                    "p10": round(max(0.0, float(value) - residual_p90), 3),
                    "p90": round(float(value) + residual_p90, 3),
                }
                for index, value in enumerate(prediction[:pred_len])
            ]
        except Exception as exc:
            return None

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
                width="stretch",
            )
        with col_raw:
            st.markdown("**Sample observations (head)**")
            display_sample = sample.head(10).copy()
            num_cols = display_sample.select_dtypes(include=["number"]).columns
            display_sample[num_cols] = display_sample[num_cols].round(2)
            st.dataframe(display_sample, width="stretch", height=300)


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
        st.dataframe(comp_df, width="stretch", hide_index=True)
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
            st.dataframe(feat_df, width="stretch", hide_index=True, height=240)

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
                "Pick any window from the SCADA history. We project every saved model on the same test window "
                "and compare the forecasts against the **actual future values** (already in the data)."
            )

        # Task block
        with st.container(border=True):
            st.markdown("### \U0001F3AF Task: Forecast N hours ahead from input window")
            cc = st.columns(3)
            has_patchtst = any(m.get("model_type") == "patchtst" for m in models)
            with cc[0]:
                horizon = st.selectbox(
                    "Horizon (hours)", [1, 6, 12, 24], index=1,
                    help="Longer horizons show more interesting forecast trajectories.",
                )
            with cc[1]:
                input_window_hours = st.selectbox(
                    "Input window (hours)", [6, 12, 24, 36, 48], index=2,
                    help="PatchTST needs 144 obs (24h). Select 36h+ for 1h horizon.",
                )
            with cc[2]:
                window_obs = input_window_hours * 6  # 10-min cadence
                horizon_obs = horizon * 6
                # Need room for input + horizon + buffer
                max_start = max(0, len(df) - window_obs - horizon_obs - 1)
                start = st.slider("Window start (row)", 0, max_start, max_start // 2, step=144)

        # Action: prepare inputs and build overlays
        with st.container(border=True):
            st.markdown("### \u2699\ufe0f Action: Build model overlays")

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

            if has_patchtst and window_obs < 144:
                st.info("PatchTST is hidden for this run because it needs 24h of history. Choose the 24h input window to enable it.")

            with st.expander("View input payload (first 3 obs)"):
                st.json(observations[:3])

            forecast = request_forecast(observations, horizon)
            model_overlays: list[dict[str, Any]] = []

            # API forecast (primary)
            if forecast is not None:
                model_overlays.append(
                    {
                        "label": forecast.get("model_version", "api-forecast"),
                        "color": MODEL_COLORS[0],
                        "points": forecast.get("forecast", []),
                    }
                )

            # Artifact-based models
            for index, model_meta in enumerate(models):
                points = _forecast_from_model_artifact(model_meta, input_slice)
                if points:
                    model_overlays.append(
                        {
                            "label": f"{model_meta.get('model_version', 'model')}",
                            "color": MODEL_COLORS[(index + 1) % len(MODEL_COLORS)],
                            "points": points,
                        }
                    )

            # Baseline forecasts (computed on-the-fly)
            baseline_points = _baseline_forecast_points(input_slice, horizon)
            for idx, (name, points) in enumerate(baseline_points.items()):
                model_overlays.append(
                    {
                        "label": name.replace("_", " ").title(),
                        "color": MODEL_COLORS[(len(models) + idx + 1) % len(MODEL_COLORS)],
                        "points": points,
                    }
                )
            primary_model = model_overlays[0] if model_overlays else None
            if forecast is None and primary_model is not None:
                forecast = {
                    "asset_id": "T1",
                    "model_version": primary_model["label"],
                    "forecast": primary_model["points"],
                    "warnings": [],
                }

        # Result block
        with st.container(border=True):
            st.markdown("### \U0001F4C8 Result: Forecast vs actual")

            if forecast is None and not model_overlays:
                st.error("No forecast returned.")
            else:
                # Show only last 12 obs (2h) of history so forecast is visible
                hist_display = input_slice.tail(12)[["timestamp", "active_power_kw"]].copy()

                # Prepare actuals (ground truth) for the forecast window
                fut_display = (
                    actual_future[["timestamp", "active_power_kw"]].copy()
                    if not actual_future.empty
                    else pd.DataFrame()
                )

                primary_forecast_df = pd.DataFrame(forecast["forecast"]) if forecast is not None else pd.DataFrame()
                if not primary_forecast_df.empty:
                    primary_forecast_df["timestamp"] = pd.to_datetime(primary_forecast_df["timestamp"], utc=True)

                overlay_frame = _build_prediction_overlay_frame(hist_display, fut_display, model_overlays)
                forecast_start = pd.to_datetime(input_slice["timestamp"].iloc[-1], utc=True)

                # Build Altair chart: one paper-style overlay with shared legend.
                if overlay_frame.empty:
                    st.info("No data available for chart.")
                else:
                    try:
                        import altair as alt

                        series_order = ["Observed history", "Gold data"] + [overlay["label"] for overlay in model_overlays]
                        series_colors = ["#9aa0a6", "#111111"] + [overlay["color"] for overlay in model_overlays]

                        forecast_chart = (
                            alt.Chart(overlay_frame)
                            .mark_line(strokeWidth=2.75)
                            .encode(
                                x=alt.X("timestamp:T", title="Time"),
                                y=alt.Y("value:Q", title="Power (kW)", scale=alt.Scale(zero=False)),
                                color=alt.Color(
                                    "series:N",
                                    scale=alt.Scale(domain=series_order, range=series_colors),
                                    legend=alt.Legend(title="Series", orient="top", symbolType="stroke"),
                                ),
                                tooltip=[
                                    alt.Tooltip("series:N", title="Series"),
                                    alt.Tooltip("timestamp:T", title="Time"),
                                    alt.Tooltip("value:Q", title="Power (kW)", format=".1f"),
                                ],
                            )
                        )

                        now_rule = (
                            alt.Chart(pd.DataFrame({"now": [forecast_start]}))
                            .mark_rule(color="#7f8c8d", strokeDash=[3, 3])
                            .encode(x="now:T")
                        )

                        chart = alt.layer(forecast_chart, now_rule).properties(
                            width="container",
                            height=360,
                            title="Forecast overlay: gold data vs model predictions",
                        ).interactive()

                        st.altair_chart(chart, width="stretch")
                        st.caption(
                            "**Observed history** = recent context \u00b7 **Gold data** = actual future values \u00b7 "
                            "**Colored lines** = model P50 forecasts \u00b7 **Dashed marker** = forecast origin"
                        )
                    except Exception as e:
                        # Fallback to simple line chart if altair fails
                        st.error(f"Chart error: {e}")
                        if not overlay_frame.empty:
                            fallback = overlay_frame.pivot_table(
                                index="timestamp",
                                columns="series",
                                values="value",
                                aggfunc="last",
                            )
                            st.line_chart(fallback, height=360)

                # Quality metrics
                if not actual_future.empty and forecast is not None and not primary_forecast_df.empty:
                    aligned = pd.merge(
                        primary_forecast_df[["timestamp", "p10", "p50", "p90"]],
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
                warns = forecast.get("warnings", []) if forecast is not None else []
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
                                     width="stretch")
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
                                     width="stretch")
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
    st.markdown("### \U0001F4CA Methodological Model Comparison")
    st.caption(
        "Every model is trained on the SAME train/val partition and "
        "evaluated on the SAME chronological test set. Test predictions "
        "are saved per-row, so the chart below overlays each model on "
        "top of the actuals on the exact same timeline."
    )

    comparison = load_model_comparison()
    results: list[dict[str, Any]] = comparison.get("results", []) if comparison else []
    split_meta: dict[str, Any] = comparison.get("split", {}) if comparison else {}

    if not results:
        st.warning(
            "No comparison artifact found. Run the unified training pipeline:\n\n"
            "```bash\npython scripts/train_all.py\n```"
        )
    else:
        # ---- Split summary ---------------------------------------------------
        with st.container(border=True):
            st.markdown("#### \U0001F50D Test split")
            sc = st.columns(4)
            sc[0].metric("Train rows", f"{split_meta.get('rows_train', 0):,}")
            sc[1].metric("Val rows", f"{split_meta.get('rows_val', 0):,}")
            sc[2].metric("Test rows", f"{split_meta.get('rows_test', 0):,}")
            sc[3].metric(
                "Step",
                f"{split_meta.get('step_minutes', 10):.0f} min",
            )
            st.caption(
                f"Train ends: {split_meta.get('train_end_ts', '?')}  ·  "
                f"Val ends: {split_meta.get('val_end_ts', '?')}  ·  "
                f"Test ends: {split_meta.get('test_end_ts', '?')}  ·  "
                f"Generated: {comparison.get('generated_at', '?')}"
            )

        results_df = pd.DataFrame(results)

        # ---- Comparison table -----------------------------------------------
        with st.container(border=True):
            st.markdown("#### \U0001F4CB Comparison table (held-out test set)")
            display = results_df.copy()
            display["skill_score"] = display["skill_score"].apply(
                lambda v: f"{v:+.3f}" if pd.notna(v) else "n/a"
            )
            display["coverage_p90"] = display["coverage_p90"].apply(
                lambda v: f"{v * 100:.1f}%" if pd.notna(v) else "n/a"
            )
            display = display.rename(
                columns={
                    "model_type": "Model",
                    "horizon_hours": "Horizon (h)",
                    "mae": "MAE (kW)",
                    "rmse": "RMSE (kW)",
                    "mape": "sMAPE (%)",
                    "skill_score": "Skill vs persistence",
                    "coverage_p90": "P10–P90 coverage",
                    "test_rows": "Test rows",
                }
            )
            st.dataframe(
                display[
                    [
                        "Model",
                        "Horizon (h)",
                        "MAE (kW)",
                        "RMSE (kW)",
                        "sMAPE (%)",
                        "Skill vs persistence",
                        "P10–P90 coverage",
                        "Test rows",
                    ]
                ].round(2),
                width="stretch",
                hide_index=True,
            )

            # Highlight best per horizon
            best_lines: list[str] = []
            for h in sorted(results_df["horizon_hours"].unique()):
                sub = results_df[results_df["horizon_hours"] == h]
                best = sub.loc[sub["mae"].idxmin()]
                best_lines.append(
                    f"**{int(h)}h:** {best['model_type']} "
                    f"(MAE = {best['mae']:.1f} kW, "
                    f"skill = {best['skill_score']:+.3f})"
                )
            if best_lines:
                st.success("Best by MAE → " + " · ".join(best_lines))

        # ---- Per-horizon overlay --------------------------------------------
        horizons_available = sorted(int(h) for h in results_df["horizon_hours"].unique())
        with st.container(border=True):
            st.markdown("#### \U0001F4C8 Predictions vs actual (overlay)")
            ec = st.columns([1, 1, 2])
            with ec[0]:
                eval_horizon = st.selectbox(
                    "Horizon", horizons_available,
                    index=0, key="eval_h_overlay",
                )
            horizon_rows = results_df[results_df["horizon_hours"] == eval_horizon]
            with ec[1]:
                model_options = horizon_rows["model_type"].unique().tolist()
                selected_models = st.multiselect(
                    "Models",
                    options=model_options,
                    default=model_options,
                    key="eval_models",
                )

            # Load all test-output CSVs for the selected horizon + models.
            per_model: dict[str, pd.DataFrame] = {}
            for _, row in horizon_rows.iterrows():
                if row["model_type"] not in selected_models:
                    continue
                test_df = load_test_outputs_by_id(row.get("test_outputs_path"))
                if not test_df.empty:
                    per_model[row["model_type"]] = test_df

            if not per_model:
                st.info("No test outputs found for the selected horizon.")
            else:
                # Determine common timeline (intersection by timestamp)
                first_df = next(iter(per_model.values()))
                ts_min = pd.to_datetime(first_df["timestamp"].min(), utc=True)
                ts_max = pd.to_datetime(first_df["timestamp"].max(), utc=True)
                with ec[2]:
                    span_hours = max(
                        1,
                        int((ts_max - ts_min).total_seconds() // 3600),
                    )
                    default_window = min(72, span_hours)
                    window_hours = st.slider(
                        "Zoom window (hours)",
                        6, span_hours, default_window,
                        key="eval_window_hours",
                    )
                    max_offset_hours = max(0, span_hours - window_hours)
                    offset_hours = st.slider(
                        "Window offset (hours from start)",
                        0, max_offset_hours,
                        max_offset_hours // 2 if max_offset_hours else 0,
                        key="eval_offset_hours",
                    )
                window_start = ts_min + pd.Timedelta(hours=offset_hours)
                window_end = window_start + pd.Timedelta(hours=window_hours)

                # Build long-form frame for Altair: one row per (timestamp, series).
                # "Actual" series is shared (taken from the first model's CSV).
                actual_long = (
                    first_df.loc[
                        (first_df["timestamp"] >= window_start)
                        & (first_df["timestamp"] < window_end),
                        ["timestamp", "actual_kw"],
                    ]
                    .rename(columns={"actual_kw": "value"})
                    .assign(series="Actual")
                )
                model_frames = [actual_long]
                for label, dfm in per_model.items():
                    sub = dfm.loc[
                        (dfm["timestamp"] >= window_start)
                        & (dfm["timestamp"] < window_end),
                        ["timestamp", "predicted_kw"],
                    ].rename(columns={"predicted_kw": "value"})
                    sub = sub.assign(series=label)
                    model_frames.append(sub)
                overlay_df = pd.concat(model_frames, ignore_index=True)
                if overlay_df.empty:
                    st.info("No data in selected window.")
                else:
                    try:
                        import altair as alt

                        domain = ["Actual", *list(per_model.keys())]
                        palette = ["#111111"] + MODEL_COLORS[: len(per_model)]
                        overlay_chart = (
                            alt.Chart(overlay_df)
                            .mark_line(strokeWidth=2)
                            .encode(
                                x=alt.X("timestamp:T", title="Time"),
                                y=alt.Y(
                                    "value:Q",
                                    title="Power (kW)",
                                    scale=alt.Scale(zero=False),
                                ),
                                color=alt.Color(
                                    "series:N",
                                    scale=alt.Scale(domain=domain, range=palette),
                                    legend=alt.Legend(title="Series", orient="top"),
                                ),
                                tooltip=[
                                    alt.Tooltip("series:N"),
                                    alt.Tooltip("timestamp:T"),
                                    alt.Tooltip("value:Q", format=".1f"),
                                ],
                            )
                            .properties(width="container", height=380)
                            .interactive()
                        )
                        st.altair_chart(overlay_chart, width="stretch")
                    except Exception as exc:  # pragma: no cover - chart fallback
                        st.error(f"Chart error: {exc}")
                        pivot = overlay_df.pivot_table(
                            index="timestamp", columns="series", values="value"
                        )
                        st.line_chart(pivot, height=380)

                # ---- Error over time --------------------------------------
                st.markdown("##### Error over time (actual − predicted)")
                err_frames = []
                for label, dfm in per_model.items():
                    sub = dfm.loc[
                        (dfm["timestamp"] >= window_start)
                        & (dfm["timestamp"] < window_end),
                        ["timestamp", "actual_kw", "predicted_kw"],
                    ].copy()
                    sub["error_kw"] = sub["actual_kw"] - sub["predicted_kw"]
                    sub["series"] = label
                    err_frames.append(sub.loc[:, ["timestamp", "error_kw", "series"]])
                err_long = pd.concat(err_frames, ignore_index=True) if err_frames else pd.DataFrame()
                if not err_long.empty and not err_long["error_kw"].isna().all():
                    try:
                        import altair as alt

                        err_chart = (
                            alt.Chart(err_long)
                            .mark_line(strokeWidth=1.5, opacity=0.85)
                            .encode(
                                x=alt.X("timestamp:T", title="Time"),
                                y=alt.Y("error_kw:Q", title="Error (kW)"),
                                color=alt.Color(
                                    "series:N",
                                    scale=alt.Scale(
                                        domain=list(per_model.keys()),
                                        range=MODEL_COLORS[: len(per_model)],
                                    ),
                                    legend=alt.Legend(title="Model", orient="top"),
                                ),
                                tooltip=[
                                    alt.Tooltip("series:N"),
                                    alt.Tooltip("timestamp:T"),
                                    alt.Tooltip("error_kw:Q", format=".1f"),
                                ],
                            )
                            .properties(width="container", height=240)
                            .interactive()
                        )
                        zero = (
                            alt.Chart(pd.DataFrame({"y": [0]}))
                            .mark_rule(color="#7f8c8d", strokeDash=[3, 3])
                            .encode(y="y:Q")
                        )
                        st.altair_chart(alt.layer(err_chart, zero), width="stretch")
                    except Exception:
                        pivot = err_long.pivot_table(
                            index="timestamp", columns="series", values="error_kw"
                        )
                        st.line_chart(pivot, height=240)

                # ---- Per-model windowed metrics ---------------------------
                st.markdown("##### Per-model metrics on the visible window")
                rows = []
                for label, dfm in per_model.items():
                    sub = dfm.loc[
                        (dfm["timestamp"] >= window_start)
                        & (dfm["timestamp"] < window_end)
                    ]
                    if sub.empty:
                        continue
                    err = (sub["actual_kw"] - sub["predicted_kw"]).to_numpy()
                    cov = float(
                        ((sub["actual_kw"] >= sub["p10"])
                         & (sub["actual_kw"] <= sub["p90"])).mean()
                    )
                    rows.append(
                        {
                            "Model": label,
                            "Rows": len(sub),
                            "MAE (kW)": round(float(np.abs(err).mean()), 1),
                            "RMSE (kW)": round(float(np.sqrt((err ** 2).mean())), 1),
                            "Bias (kW)": round(float(err.mean()), 1),
                            "P10–P90 coverage": f"{cov * 100:.1f}%",
                        }
                    )
                if rows:
                    st.dataframe(
                        pd.DataFrame(rows),
                        width="stretch",
                        hide_index=True,
                    )


st.divider()
st.caption(f"Windenegy v0.1.0 \u00b7 API: {API_URL} \u00b7 [Docs](/docs)")
