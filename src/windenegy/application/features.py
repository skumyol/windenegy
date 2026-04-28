"""Feature engineering utilities shared by training and inference."""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import pandas as pd

RAW_COLUMN_MAP: Final[dict[str, str]] = {
    "Date/Time": "timestamp",
    "LV ActivePower (kW)": "active_power_kw",
    "Wind Speed (m/s)": "wind_speed_mps",
    "Theoretical_Power_Curve (KWh)": "theoretical_power_kwh",
    "Wind Direction (°)": "wind_direction_deg",
}

REQUIRED_INTERNAL_COLUMNS: Final[tuple[str, ...]] = (
    "timestamp",
    "active_power_kw",
    "wind_speed_mps",
    "theoretical_power_kwh",
    "wind_direction_deg",
)


def normalize_scada_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Kaggle SCADA columns to the internal schema."""
    normalized = df.rename(columns=RAW_COLUMN_MAP).copy()
    missing = sorted(set(REQUIRED_INTERNAL_COLUMNS) - set(normalized.columns))
    if missing:
        msg = f"Missing required SCADA columns: {missing}"
        raise ValueError(msg)

    normalized = normalized.loc[:, list(REQUIRED_INTERNAL_COLUMNS)]
    # Handle both Kaggle format ("01 01 2018 00:00") and ISO format ("2018-01-01 00:00:00")
    ts_data = normalized["timestamp"].astype(str)
    if ts_data.str.match(r"^\d{4}-").any():
        # ISO format detected
        normalized["timestamp"] = pd.to_datetime(ts_data, utc=True)
    else:
        # Kaggle format "DD MM YYYY HH:MM"
        normalized["timestamp"] = pd.to_datetime(ts_data, format="%d %m %Y %H:%M", utc=True)
    normalized["wind_direction_deg"] = normalized["wind_direction_deg"].mod(360.0)
    normalized = normalized.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return normalized.reset_index(drop=True)


def add_direction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add circular wind-direction features."""
    featured = df.copy()
    radians = np.deg2rad(featured["wind_direction_deg"].astype(float))
    featured["wind_direction_sin"] = np.sin(radians)
    featured["wind_direction_cos"] = np.cos(radians)
    return featured


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic time-of-day and day-of-year features."""
    featured = df.copy()
    timestamp = pd.to_datetime(featured["timestamp"], utc=True)
    hour = timestamp.dt.hour + timestamp.dt.minute / 60.0
    day_of_year = timestamp.dt.dayofyear.astype(float)

    featured["hour_sin"] = np.sin(2 * math.pi * hour / 24.0)
    featured["hour_cos"] = np.cos(2 * math.pi * hour / 24.0)
    featured["day_sin"] = np.sin(2 * math.pi * day_of_year / 366.0)
    featured["day_cos"] = np.cos(2 * math.pi * day_of_year / 366.0)
    return featured


def add_lag_features(
    df: pd.DataFrame,
    lags: tuple[int, ...] = (1, 3, 6, 12, 24),
    windows: tuple[int, ...] = (3, 6, 12),
) -> pd.DataFrame:
    """Add lag and rolling features for tabular models."""
    featured = df.copy()
    for lag in lags:
        featured[f"active_power_lag_{lag}"] = featured["active_power_kw"].shift(lag)
        featured[f"wind_speed_lag_{lag}"] = featured["wind_speed_mps"].shift(lag)

    for window in windows:
        featured[f"active_power_roll_mean_{window}"] = (
            featured["active_power_kw"].rolling(window).mean()
        )
        featured[f"wind_speed_roll_mean_{window}"] = (
            featured["wind_speed_mps"].rolling(window).mean()
        )
        featured[f"active_power_roll_std_{window}"] = (
            featured["active_power_kw"].rolling(window).std()
        )

    featured["wind_speed_sq"] = featured["wind_speed_mps"] ** 2
    featured["wind_speed_cb"] = featured["wind_speed_mps"] ** 3
    featured["power_curve_deviation"] = (
        featured["active_power_kw"] - featured["theoretical_power_kwh"]
    )
    featured["power_efficiency"] = np.where(
        featured["theoretical_power_kwh"] > 50,
        featured["active_power_kw"] / featured["theoretical_power_kwh"],
        1.0
    )
    featured["power_rate_of_change"] = featured["active_power_kw"].diff()
    featured["wind_speed_rate_of_change"] = featured["wind_speed_mps"].diff()

    featured["turbulence_intensity"] = np.where(
        featured["active_power_roll_mean_3"] > 50,
        featured["active_power_roll_std_3"] / featured["active_power_roll_mean_3"],
        0.0
    )

    featured["wind_power_density"] = (
        0.5 * 1.225 * featured["wind_speed_cb"]
    )

    featured["capacity_factor"] = np.where(
        featured["theoretical_power_kwh"] > 100,
        featured["active_power_kw"] / featured["theoretical_power_kwh"],
        0.0
    )

    featured["wind_shear_indicator"] = np.where(
        featured["wind_speed_mps"] > 3,
        featured["active_power_kw"] / (featured["wind_speed_sq"] + 1),
        0.0
    )

    return featured.dropna().reset_index(drop=True)


def build_tabular_feature_frame(
    df: pd.DataFrame,
    lags: tuple[int, ...] = (1, 3, 6, 12, 24),
    windows: tuple[int, ...] = (3, 6, 12),
) -> pd.DataFrame:
    """Build the canonical tabular feature frame."""
    normalized = normalize_scada_frame(df)
    featured = add_direction_features(normalized)
    featured = add_time_features(featured)
    featured = add_lag_features(featured, lags=lags, windows=windows)
    featured["persistence_kw"] = featured["active_power_kw"]
    return featured


def build_supervised_feature_frame(
    df: pd.DataFrame,
    horizon_steps: int,
    lags: tuple[int, ...] = (1, 3, 6, 12, 24),
    windows: tuple[int, ...] = (3, 6, 12),
) -> pd.DataFrame:
    """Build a feature frame with a future power target."""
    if horizon_steps < 1:
        msg = "horizon_steps must be >= 1"
        raise ValueError(msg)

    featured = build_tabular_feature_frame(df, lags=lags, windows=windows)
    featured["target_active_power_kw"] = featured["active_power_kw"].shift(-horizon_steps)
    return featured.dropna().reset_index(drop=True)


def observations_to_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    """Build an internal SCADA dataframe from observation dictionaries."""
    frame = pd.DataFrame.from_records(records)
    return normalize_scada_frame(frame)
