"""Unit tests for feature engineering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from windenegy.application.features import build_tabular_feature_frame, normalize_scada_frame


def test_normalize_scada_frame_maps_kaggle_columns() -> None:
    """Kaggle columns are normalized to the internal contract."""
    frame = pd.read_csv(Path("tests/fixtures/sample_scada.csv"))

    normalized = normalize_scada_frame(frame)

    assert list(normalized.columns) == [
        "timestamp",
        "active_power_kw",
        "wind_speed_mps",
        "theoretical_power_kwh",
        "wind_direction_deg",
    ]
    assert normalized["timestamp"].dt.tz is not None


def test_normalize_scada_frame_rejects_missing_columns() -> None:
    """Missing required columns fail loudly."""
    with pytest.raises(ValueError, match="Missing required SCADA columns"):
        normalize_scada_frame(pd.DataFrame({"Date/Time": ["2018-01-01"]}))


def test_build_tabular_feature_frame_adds_shared_features() -> None:
    """The canonical tabular feature frame includes lag and cyclic features."""
    frame = pd.read_csv(Path("tests/fixtures/sample_scada.csv"))

    featured = build_tabular_feature_frame(frame)

    assert "wind_direction_sin" in featured.columns
    assert "wind_direction_cos" in featured.columns
    assert "hour_sin" in featured.columns
    assert "active_power_lag_1" in featured.columns
    assert not featured.isna().any().any()
