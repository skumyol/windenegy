"""Unified train/test split + per-model evaluation pipeline.

This module is the single source of truth for *methodological* model
comparison. Every model is trained on the same train/val partition and
evaluated on the same chronological test partition, producing a
test-output CSV with an identical schema. The dashboard consumes those
CSVs to overlay every model on top of the actuals on the same timeline.

Key design rules:

- One canonical split: chronological train (default 70%), validation
  (default 15%), test (default 15%) over the normalized SCADA frame.
- One forecast semantics for *every* model: at test time `t`, predict
  `active_power_kw[t]` using only information available at or before
  `t - horizon_steps` (for h-step-ahead point forecasting).
- One row schema for every model's saved test outputs:
  `timestamp, actual_kw, predicted_kw, p10, p90, baseline_kw,
  residual_kw, model_id, model_type, horizon_hours`.
- Skill score is always computed against the persistence baseline on
  the same rows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from windenegy.application.evaluation import build_metrics
from windenegy.application.features import (
    build_supervised_feature_frame,
    normalize_scada_frame,
)
from windenegy.domain.models import ModelMetrics

try:  # pragma: no cover - optional dependency
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


TEST_OUTPUT_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "actual_kw",
    "predicted_kw",
    "p10",
    "p90",
    "baseline_kw",
    "residual_kw",
    "model_id",
    "model_type",
    "horizon_hours",
)


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitInfo:
    """Chronological train/val/test split over a normalized SCADA frame."""

    normalized: pd.DataFrame
    train_end_idx: int
    val_end_idx: int
    train_end_ts: datetime
    val_end_ts: datetime
    test_end_ts: datetime
    step_minutes: float
    train_ratio: float
    val_ratio: float

    @property
    def n_total(self) -> int:
        return len(self.normalized)

    @property
    def n_train(self) -> int:
        return self.train_end_idx

    @property
    def n_val(self) -> int:
        return self.val_end_idx - self.train_end_idx

    @property
    def n_test(self) -> int:
        return self.n_total - self.val_end_idx

    def to_dict(self) -> dict[str, Any]:
        ts = self.normalized["timestamp"]
        return {
            "rows_total": self.n_total,
            "rows_train": self.n_train,
            "rows_val": self.n_val,
            "rows_test": self.n_test,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "train_start_ts": ts.iloc[0].isoformat(),
            "train_end_ts": self.train_end_ts.isoformat(),
            "val_end_ts": self.val_end_ts.isoformat(),
            "test_end_ts": self.test_end_ts.isoformat(),
            "step_minutes": self.step_minutes,
        }


def prepare_split(
    csv_path: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> SplitInfo:
    """Load a SCADA CSV and produce a chronological split."""
    if not 0.0 < train_ratio < 1.0:
        msg = "train_ratio must be in (0, 1)"
        raise ValueError(msg)
    if not 0.0 < val_ratio < 1.0 or train_ratio + val_ratio >= 1.0:
        msg = "val_ratio must be in (0, 1) and train+val < 1"
        raise ValueError(msg)

    raw = pd.read_csv(csv_path)
    normalized = normalize_scada_frame(raw)
    n = len(normalized)
    if n < 100:
        msg = f"Need at least 100 rows for a meaningful split, got {n}"
        raise ValueError(msg)

    train_end = max(1, int(n * train_ratio))
    val_end = max(train_end + 1, int(n * (train_ratio + val_ratio)))
    val_end = min(val_end, n - 1)

    deltas = normalized["timestamp"].diff().dropna().dt.total_seconds() / 60.0
    step_minutes = float(deltas.median()) if not deltas.empty else 10.0

    return SplitInfo(
        normalized=normalized,
        train_end_idx=train_end,
        val_end_idx=val_end,
        train_end_ts=normalized["timestamp"].iloc[train_end - 1].to_pydatetime(),
        val_end_ts=normalized["timestamp"].iloc[val_end - 1].to_pydatetime(),
        test_end_ts=normalized["timestamp"].iloc[-1].to_pydatetime(),
        step_minutes=step_minutes,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )


# ---------------------------------------------------------------------------
# Anchors and helpers
# ---------------------------------------------------------------------------


def _build_anchors(
    split: SplitInfo,
    horizon_steps: int,
    region: str,
) -> pd.DataFrame:
    """Build per-row anchors for a forecast region.

    For each row at index `i` in the region, an h-step-ahead forecast
    targets `y[i]` using info available up to `i - horizon_steps`.
    """
    if region == "train":
        start = horizon_steps
        end = split.train_end_idx
    elif region == "val":
        start = max(horizon_steps, split.train_end_idx)
        end = split.val_end_idx
    elif region == "test":
        start = max(horizon_steps, split.val_end_idx)
        end = split.n_total
    else:
        msg = f"Unknown region: {region}"
        raise ValueError(msg)

    if start >= end:
        return pd.DataFrame(
            columns=["idx", "timestamp", "actual_kw", "baseline_kw", "theoretical_kw", "wind_speed"],
        )

    idx = np.arange(start, end)
    nf = split.normalized
    return pd.DataFrame(
        {
            "idx": idx,
            "timestamp": nf["timestamp"].iloc[idx].to_numpy(),
            "actual_kw": nf["active_power_kw"].iloc[idx].to_numpy(),
            "baseline_kw": nf["active_power_kw"].iloc[idx - horizon_steps].to_numpy(),
            "theoretical_kw": nf["theoretical_power_kwh"].iloc[idx].to_numpy(),
            "wind_speed": nf["wind_speed_mps"].iloc[idx].to_numpy(),
        }
    )


def _resid_p90(actual: np.ndarray, predicted: np.ndarray, floor: float = 50.0) -> float:
    """Robust P90 of absolute residuals with a floor."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mask = np.isfinite(a) & np.isfinite(p)
    if not mask.any():
        return floor
    r = np.abs(a[mask] - p[mask])
    if r.size == 0:
        return floor
    return float(max(floor, float(np.quantile(r, 0.9))))


def _format_outputs(
    model_id: str,
    model_type: str,
    horizon_hours: int,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Stamp model identity and compute residuals; enforce column order."""
    out = df.copy()
    out["model_id"] = model_id
    out["model_type"] = model_type
    out["horizon_hours"] = horizon_hours
    out["residual_kw"] = out["actual_kw"].astype(float) - out["predicted_kw"].astype(float)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out.loc[:, list(TEST_OUTPUT_COLUMNS)].sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-model evaluators
# ---------------------------------------------------------------------------


def evaluate_persistence(
    split: SplitInfo,
    horizon_steps: int,
    horizon_hours: int,
    model_id: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Persistence baseline: predict y[t] = y[t - h]."""
    val = _build_anchors(split, horizon_steps, "val")
    p90 = _resid_p90(val["actual_kw"].to_numpy(), val["baseline_kw"].to_numpy())

    test = _build_anchors(split, horizon_steps, "test").copy()
    test["predicted_kw"] = test["baseline_kw"]
    test["p10"] = np.maximum(0.0, test["predicted_kw"] - p90)
    test["p90"] = test["predicted_kw"] + p90

    return _format_outputs(model_id, "persistence", horizon_hours, test), {"residual_p90": p90}


def evaluate_power_curve(
    split: SplitInfo,
    horizon_steps: int,
    horizon_hours: int,
    model_id: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Manufacturer power curve baseline."""
    val = _build_anchors(split, horizon_steps, "val")
    p90 = _resid_p90(val["actual_kw"].to_numpy(), val["theoretical_kw"].to_numpy())

    test = _build_anchors(split, horizon_steps, "test").copy()
    test["predicted_kw"] = test["theoretical_kw"]
    test["p10"] = np.maximum(0.0, test["predicted_kw"] - p90)
    test["p90"] = test["predicted_kw"] + p90

    return _format_outputs(model_id, "power_curve", horizon_hours, test), {"residual_p90": p90}


def evaluate_rolling_mean(
    split: SplitInfo,
    horizon_steps: int,
    horizon_hours: int,
    model_id: str,
    window: int = 6,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rolling-mean baseline using last `window` obs available at t-h."""
    rolling = split.normalized["active_power_kw"].rolling(window=window).mean()

    def predict(anchors: pd.DataFrame) -> np.ndarray:
        if anchors.empty:
            return np.array([])
        end_idx = anchors["idx"].to_numpy() - horizon_steps
        return rolling.iloc[end_idx].to_numpy()

    val = _build_anchors(split, horizon_steps, "val").copy()
    val["predicted_kw"] = predict(val)
    val = val.dropna(subset=["predicted_kw"])
    p90 = _resid_p90(val["actual_kw"].to_numpy(), val["predicted_kw"].to_numpy())

    test = _build_anchors(split, horizon_steps, "test").copy()
    test["predicted_kw"] = predict(test)
    test = test.dropna(subset=["predicted_kw"]).reset_index(drop=True)
    test["p10"] = np.maximum(0.0, test["predicted_kw"] - p90)
    test["p90"] = test["predicted_kw"] + p90

    return (
        _format_outputs(model_id, "rolling_mean", horizon_hours, test),
        {"residual_p90": p90, "window": window},
    )


def evaluate_gradient_boosting(
    split: SplitInfo,
    horizon_steps: int,
    horizon_hours: int,
    model_id: str,
    lags: tuple[int, ...] = (1, 3, 6, 12, 24, 36, 48),
    windows: tuple[int, ...] = (3, 6, 12, 24, 48),
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any], Any, list[str]]:
    """Train LightGBM (or sklearn fallback) on train+val, evaluate on test.

    Predicts the *delta* over the persistence baseline, with a
    bias correction estimated on validation residuals.
    """
    supervised = build_supervised_feature_frame(
        split.normalized,
        horizon_steps=horizon_steps,
        lags=lags,
        windows=windows,
    )
    if len(supervised) < 50:
        msg = "Not enough rows after feature engineering for gradient boosting"
        raise ValueError(msg)

    step = pd.Timedelta(minutes=split.step_minutes)
    forecast_ts = pd.to_datetime(supervised["timestamp"], utc=True) + step * horizon_steps
    supervised = supervised.assign(forecast_ts=forecast_ts)

    train_end_ts = pd.Timestamp(split.train_end_ts)
    val_end_ts = pd.Timestamp(split.val_end_ts)

    train_mask = supervised["forecast_ts"] <= train_end_ts
    val_mask = (supervised["forecast_ts"] > train_end_ts) & (supervised["forecast_ts"] <= val_end_ts)
    test_mask = supervised["forecast_ts"] > val_end_ts

    non_feature = {"timestamp", "active_power_kw", "target_active_power_kw", "forecast_ts"}
    feature_columns = [
        c
        for c in supervised.columns
        if c not in non_feature and pd.api.types.is_numeric_dtype(supervised[c])
    ]

    train_df = supervised.loc[train_mask]
    val_df = supervised.loc[val_mask]
    test_df = supervised.loc[test_mask]

    if train_df.empty or test_df.empty:
        msg = "Empty train or test partition after feature engineering"
        raise ValueError(msg)

    train_active = train_df[train_df["active_power_kw"] > 100]
    if len(train_active) < 100:
        train_active = train_df
    val_active = val_df[val_df["active_power_kw"] > 100]
    if len(val_active) < 50:
        val_active = val_df

    weights = np.ones(len(train_active))
    weights[(train_active["active_power_kw"] > 500).to_numpy()] = 2.0

    train_target = (
        train_active["target_active_power_kw"] - train_active["active_power_kw"]
    ).to_numpy()
    val_target = (
        val_active["target_active_power_kw"] - val_active["active_power_kw"]
    ).to_numpy()

    if lgb is not None:
        regressor: Any = lgb.LGBMRegressor(
            boosting_type="gbdt",
            objective="regression_l1",
            num_leaves=64,
            learning_rate=0.05,
            n_estimators=2000,
            subsample=0.8,
            subsample_freq=5,
            colsample_bytree=0.9,
            reg_lambda=0.1,
            min_child_samples=25,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
        eval_set = (
            [(val_active[feature_columns], val_target)] if len(val_active) else None
        )
        regressor.fit(
            train_active[feature_columns],
            train_target,
            sample_weight=weights,
            eval_set=eval_set,
            eval_metric="l1",
            callbacks=[lgb.early_stopping(100, verbose=False)] if eval_set else None,
        )
        best_it = getattr(regressor, "best_iteration_", None)
        kw: dict[str, Any] = {"num_iteration": best_it} if best_it else {}
        val_pred_delta = regressor.predict(val_active[feature_columns], **kw)
        test_pred_delta = regressor.predict(test_df[feature_columns], **kw)
    else:
        from sklearn.ensemble import GradientBoostingRegressor

        regressor = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
        )
        regressor.fit(train_active[feature_columns], train_target, sample_weight=weights)
        val_pred_delta = regressor.predict(val_active[feature_columns])
        test_pred_delta = regressor.predict(test_df[feature_columns])

    bias = float(np.mean(val_target - val_pred_delta)) if len(val_target) else 0.0
    val_predicted = val_active["active_power_kw"].to_numpy() + val_pred_delta + bias
    p90 = _resid_p90(val_active["target_active_power_kw"].to_numpy(), val_predicted)

    test_predicted = test_df["active_power_kw"].to_numpy() + test_pred_delta + bias
    test_predicted = np.maximum(0.0, test_predicted)
    out = pd.DataFrame(
        {
            "timestamp": test_df["forecast_ts"].to_numpy(),
            "actual_kw": test_df["target_active_power_kw"].to_numpy(),
            "predicted_kw": test_predicted,
            "p10": np.maximum(0.0, test_predicted - p90),
            "p90": test_predicted + p90,
            "baseline_kw": test_df["active_power_kw"].to_numpy(),
        }
    )

    artifact = {
        "feature_columns": feature_columns,
        "lags": list(lags),
        "windows": list(windows),
        "bias_correction": bias,
        "residual_p90": p90,
        "predicts_delta": True,
        "horizon_steps": horizon_steps,
        "n_train": int(len(train_active)),
        "n_val": int(len(val_active)),
        "n_test": int(len(test_df)),
    }
    return (
        _format_outputs(model_id, "gradient_boosting", horizon_hours, out),
        artifact,
        regressor,
        feature_columns,
    )


def evaluate_patchtst(
    split: SplitInfo,
    horizon_steps: int,
    horizon_hours: int,
    model_id: str,
    seq_len: int = 144,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any], Any] | None:
    """Sequence-model evaluator (MLP stand-in for PatchTST).

    Returns ``None`` if there is not enough data to train.
    """
    feature_cols = [
        "active_power_kw",
        "wind_speed_mps",
        "theoretical_power_kwh",
        "wind_direction_deg",
    ]
    nf = split.normalized
    arr = nf[feature_cols].to_numpy(dtype=float)
    target = nf["active_power_kw"].to_numpy(dtype=float)
    ts = pd.to_datetime(nf["timestamp"], utc=True).to_numpy()
    n = len(nf)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for i in range(n - seq_len - horizon_steps + 1):
        forecast_idx = i + seq_len + horizon_steps - 1
        if forecast_idx < split.train_end_idx:
            train_idx.append(i)
        elif forecast_idx < split.val_end_idx:
            val_idx.append(i)
        else:
            test_idx.append(i)

    if len(train_idx) < 200 or not test_idx:
        return None

    def stack(indices: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not indices:
            return np.empty((0, seq_len * len(feature_cols))), np.empty(0), np.empty(0, dtype="datetime64[ns]")
        x = np.stack([arr[i : i + seq_len].flatten() for i in indices])
        y = np.array([target[i + seq_len + horizon_steps - 1] for i in indices])
        t = np.array([ts[i + seq_len + horizon_steps - 1] for i in indices])
        return x, y, t

    x_train, y_train, _ = stack(train_idx)
    x_val, y_val, _ = stack(val_idx)
    x_test, y_test, t_test = stack(test_idx)
    last_obs_test = np.array([arr[i + seq_len - 1, 0] for i in test_idx])

    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x_train)
    regressor = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=random_state,
    )
    regressor.fit(scaler.transform(x_train), y_train)

    if x_val.size:
        val_pred = regressor.predict(scaler.transform(x_val))
        p90 = _resid_p90(y_val, val_pred)
    else:
        p90 = 200.0

    test_pred = regressor.predict(scaler.transform(x_test))
    test_pred = np.maximum(0.0, test_pred)

    out = pd.DataFrame(
        {
            "timestamp": t_test,
            "actual_kw": y_test,
            "predicted_kw": test_pred,
            "p10": np.maximum(0.0, test_pred - p90),
            "p90": test_pred + p90,
            "baseline_kw": last_obs_test,
        }
    )

    artifact = {
        "seq_len": seq_len,
        "pred_len": horizon_steps,
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "residual_p90": p90,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
    }
    return (
        _format_outputs(model_id, "patchtst", horizon_hours, out),
        artifact,
        (regressor, scaler),
    )


# ---------------------------------------------------------------------------
# Metrics + persistence
# ---------------------------------------------------------------------------


def metrics_from_outputs(
    out_df: pd.DataFrame,
    model_id: str,
    horizon_hours: int,
) -> ModelMetrics:
    """Build domain metrics from a unified test-output frame."""
    return build_metrics(
        model_id=model_id,
        horizon_hours=horizon_hours,
        actual=out_df["actual_kw"].to_numpy(),
        predicted=out_df["predicted_kw"].to_numpy(),
        baseline_predicted=out_df["baseline_kw"].to_numpy(),
        lower=out_df["p10"].to_numpy(),
        upper=out_df["p90"].to_numpy(),
    )


def save_test_outputs(metrics_dir: Path, model_id: str, out_df: pd.DataFrame) -> Path:
    """Persist a test-output CSV with stable schema and ISO timestamps."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / f"{model_id}_test_outputs.csv"
    df = out_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    df.to_csv(csv_path, index=False)
    return csv_path


def save_metrics_json(metrics_dir: Path, model_id: str, metrics: ModelMetrics) -> Path:
    """Persist a metrics JSON file alongside the test-output CSV."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / f"{model_id}.json"
    with path.open("w") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    return path


def save_split_json(metrics_dir: Path, split: SplitInfo) -> Path:
    """Persist split metadata so the dashboard can show it."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / "test_split.json"
    with path.open("w") as f:
        json.dump(split.to_dict(), f, indent=2, default=str)
    return path
