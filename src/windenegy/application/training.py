"""Model training and artifact wiring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

try:  # pragma: no cover - optional dependency
    import lightgbm as lgb
except ImportError:  # pragma: no cover - runtime fallback
    lgb = None

from windenegy.application.evaluation import build_metrics
from windenegy.application.features import (
    build_supervised_feature_frame,
    build_tabular_feature_frame,
    normalize_scada_frame,
)
from windenegy.domain.models import ModelArtifactMetadata, ModelMetrics, TurbineObservation
from windenegy.infrastructure.persistence import FileSystemModelRepository

TARGET_COLUMN = "target_active_power_kw"
NON_FEATURE_COLUMNS = {
    "timestamp",
    "active_power_kw",
    "target_active_power_kw",
}


@dataclass(frozen=True)
class TrainingResult:
    """Result of a completed training run."""

    model_id: str
    metadata: ModelArtifactMetadata
    metrics_path: Path
    test_predictions_path: Path
    model_path: Path


@dataclass(frozen=True)
class GradientBoostingPowerModel:
    """Serializable model wrapper with its inference feature contract."""

    regressor: Any
    feature_columns: list[str]
    residual_p90: float
    bias_correction: float
    horizon_hours: int
    model_version: str
    lags: tuple[int, ...]
    windows: tuple[int, ...]
    predicts_delta: bool = False

    def __getstate__(self) -> dict[str, Any]:  # pragma: no cover - compatibility
        return {
            "regressor": self.regressor,
            "feature_columns": self.feature_columns,
            "residual_p90": self.residual_p90,
            "bias_correction": self.bias_correction,
            "horizon_hours": self.horizon_hours,
            "model_version": self.model_version,
            "lags": self.lags,
            "windows": self.windows,
            "predicts_delta": self.predicts_delta,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:  # pragma: no cover - compatibility
        state.setdefault("predicts_delta", False)
        for key, value in state.items():
            object.__setattr__(self, key, value)

    def predict_from_observations(self, observations: list[TurbineObservation]) -> float:
        """Predict future power from recent SCADA observations."""
        if not observations:
            msg = "At least one observation is required"
            raise ValueError(msg)
        records = [item.model_dump() for item in observations]
        frame = pd.DataFrame.from_records(records)
        feature_frame = build_tabular_feature_frame(
            frame,
            lags=self.lags,
            windows=self.windows,
        )
        if feature_frame.empty:
            msg = "Not enough observations to build model features"
            raise ValueError(msg)

        missing_features = set(self.feature_columns) - set(feature_frame.columns)
        if missing_features:
            msg = f"Missing features: {missing_features}"
            raise ValueError(msg)

        latest_features = feature_frame.iloc[[-1]][self.feature_columns]
        prediction = float(self.regressor.predict(latest_features)[0])
        prediction = prediction + self.bias_correction
        if self.predicts_delta:
            last_power = float(feature_frame["active_power_kw"].iloc[-1])
            prediction = last_power + prediction
        return max(0.0, prediction)


def train_gradient_boosting_from_csv(
    csv_path: Path,
    model_dir: Path,
    metrics_dir: Path,
    horizon_hours: int = 1,
    asset_id: str = "T1",
    random_state: int = 42,
    test_start_pct: float | None = None,
) -> TrainingResult:
    """Train and persist a gradient boosting forecaster from a SCADA CSV.

    Args:
        test_start_pct: Optional fraction (0.8-1.0) indicating where test set starts.
                        If None, uses chronological 80/10/10 split.
    """
    raw_frame = pd.read_csv(csv_path)
    normalized = normalize_scada_frame(raw_frame)
    step_minutes = _infer_step_minutes(normalized["timestamp"])
    horizon_steps = max(1, round((horizon_hours * 60) / step_minutes))
    supervised, lags, windows = _build_training_frame(normalized, horizon_steps)
    supervised["target_delta_kw"] = supervised[TARGET_COLUMN] - supervised["active_power_kw"]
    if len(supervised) < 10:
        msg = (
            "Not enough rows after feature engineering for training. "
            "Use a longer SCADA file or a shorter horizon."
        )
        raise ValueError(msg)

    if test_start_pct is not None:
        n = len(supervised)
        test_start = int(n * test_start_pct)
        test = supervised.iloc[test_start:]
        train_val = supervised.iloc[:test_start]
        val_size = max(1, len(train_val) // 10)
        train = train_val.iloc[:-val_size]
        validation = train_val.iloc[-val_size:]
    else:
        train, validation, test = _chronological_frame_split(supervised)
    delta_column = "target_delta_kw"
    feature_columns = [
        column
        for column in supervised.columns
        if column not in NON_FEATURE_COLUMNS.union({delta_column})
        and pd.api.types.is_numeric_dtype(supervised[column])
    ]

    train_active = train[train["active_power_kw"] > 100]
    val_active = validation[validation["active_power_kw"] > 100]
    test_active = test[test["active_power_kw"] > 100]

    if len(train_active) < 100:
        train_active = train

    if len(val_active) < 50:
        val_active = validation

    if len(test_active) < 50:
        test_active = test

    sample_weights = np.ones(len(train_active))
    active_mask = train_active["active_power_kw"] > 500
    sample_weights[active_mask] = 2.0

    use_lightgbm = lgb is not None
    if use_lightgbm:
        regressor = lgb.LGBMRegressor(
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
        )
        eval_set: list[tuple[pd.DataFrame, pd.Series]] = []
        if not val_active.empty:
            eval_set.append((val_active[feature_columns], val_active[delta_column]))
        regressor.fit(
            train_active[feature_columns],
            train_active[delta_column],
            sample_weight=sample_weights,
            eval_set=eval_set or None,
            eval_metric="l1",
            callbacks=[lgb.early_stopping(100, verbose=False)] if eval_set else None,
        )
        validation_predicted = regressor.predict(
            val_active[feature_columns],
            num_iteration=getattr(regressor, "best_iteration_", None),
        )
    else:
        regressor = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
        )
        regressor.fit(
            train_active[feature_columns],
            train_active[delta_column],
            sample_weight=sample_weights,
        )
        validation_predicted = regressor.predict(val_active[feature_columns])

    residuals = val_active[delta_column].to_numpy() - validation_predicted
    bias_correction = float(np.mean(residuals))
    residual_p90 = float(np.quantile(np.abs(residuals), 0.9))
    residual_p90 = max(50.0, residual_p90)

    predict_kwargs: dict[str, Any] = {}
    if use_lightgbm and hasattr(regressor, "best_iteration_") and regressor.best_iteration_ is not None:
        predict_kwargs["num_iteration"] = regressor.best_iteration_

    test_pred_delta = regressor.predict(test_active[feature_columns], **predict_kwargs) + bias_correction
    persistence_predicted = test_active["active_power_kw"].to_numpy()
    test_predicted = persistence_predicted + test_pred_delta
    lower = np.maximum(0.0, test_predicted - residual_p90)
    upper = test_predicted + residual_p90

    model_id = f"gradient-boosting-{asset_id}-{horizon_hours}h"
    test_outputs_path = metrics_dir / f"{model_id}_test_outputs.csv"
    test_outputs = pd.DataFrame(
        {
            "timestamp": test_active["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "actual_kw": test_active[TARGET_COLUMN].to_numpy(),
            "predicted_kw": test_predicted,
            "baseline_predicted_kw": persistence_predicted,
            "p10": lower,
            "p90": upper,
        }
    )
    test_outputs["residual_kw"] = test_outputs["actual_kw"] - test_outputs["predicted_kw"]

    metrics = build_metrics(
        model_id="gradient_boosting",
        horizon_hours=horizon_hours,
        actual=test_active[TARGET_COLUMN].to_numpy(),
        predicted=test_predicted,
        baseline_predicted=persistence_predicted,
        lower=lower,
        upper=upper,
    )

    model = GradientBoostingPowerModel(
        regressor=regressor,
        feature_columns=feature_columns,
        residual_p90=residual_p90,
        bias_correction=bias_correction,
        horizon_hours=horizon_hours,
        model_version=model_id,
        lags=lags,
        windows=windows,
        predicts_delta=True,
    )
    metadata = ModelArtifactMetadata(
        model_type="gradient_boosting",
        model_version=model_id,
        feature_schema={column: str(supervised[column].dtype) for column in feature_columns},
        target="active_power_kw",
        horizon_hours=horizon_hours,
        metrics=metrics,
        config_snapshot={
            "asset_id": asset_id,
            "rows": len(normalized),
            "training_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "bias_correction": bias_correction,
            "step_minutes": step_minutes,
            "horizon_steps": horizon_steps,
            "lags": list(lags),
            "windows": list(windows),
            "test_outputs_path": str(test_outputs_path),
        },
    )

    metrics_dir.mkdir(parents=True, exist_ok=True)
    repository = FileSystemModelRepository(model_dir)
    model_path = repository.save_model(model_id, model, metadata)

    metrics_path = metrics_dir / f"{model_id}.json"
    with metrics_path.open("w") as file:
        json.dump(metrics.to_dict(), file, indent=2)

    test_outputs.to_csv(test_outputs_path, index=False)

    return TrainingResult(
        model_id=model_id,
        metadata=metadata,
        metrics_path=metrics_path,
        test_predictions_path=test_outputs_path,
        model_path=model_path,
    )


def load_latest_gradient_boosting(
    model_dir: Path,
) -> tuple[GradientBoostingPowerModel, ModelArtifactMetadata] | None:
    """Load the latest persisted gradient boosting model, if present."""
    repository = FileSystemModelRepository(model_dir)
    model_id = repository.get_latest_model(model_type="gradient_boosting")
    if model_id is None:
        return None
    model, metadata = repository.load_model(model_id)
    if not isinstance(model, GradientBoostingPowerModel):
        msg = f"Model artifact {model_id} is not a GradientBoostingPowerModel"
        raise TypeError(msg)
    return model, metadata


def metric_summary(metadata: ModelArtifactMetadata | None) -> dict[str, Any]:
    """Return a JSON-serializable metadata summary for API capabilities."""
    if metadata is None:
        return {}
    metrics: ModelMetrics | None = metadata.metrics
    return {
        "model_version": metadata.model_version,
        "model_type": metadata.model_type,
        "horizon_hours": metadata.horizon_hours,
        "feature_count": len(metadata.feature_schema),
        "metrics": metrics.to_dict() if metrics is not None else None,
    }


def _infer_step_minutes(timestamp: pd.Series) -> float:
    """Infer median sampling cadence in minutes."""
    deltas = timestamp.sort_values().diff().dropna().dt.total_seconds() / 60.0
    if deltas.empty:
        return 10.0
    return float(deltas.median())


def _chronological_frame_split(
    frame: pd.DataFrame,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a supervised frame chronologically."""
    sorted_frame = frame.sort_values("timestamp").reset_index(drop=True)
    train_end = max(1, int(len(sorted_frame) * train_ratio))
    validation_end = max(train_end + 1, int(len(sorted_frame) * (train_ratio + validation_ratio)))
    validation_end = min(validation_end, len(sorted_frame) - 1)

    train = sorted_frame.iloc[:train_end]
    validation = sorted_frame.iloc[train_end:validation_end]
    test = sorted_frame.iloc[validation_end:]
    if train.empty or validation.empty or test.empty:
        msg = "Chronological split produced an empty train, validation, or test set"
        raise ValueError(msg)
    return train, validation, test


def _build_training_frame(
    normalized: pd.DataFrame,
    horizon_steps: int,
) -> tuple[pd.DataFrame, tuple[int, ...], tuple[int, ...]]:
    """Build training features, falling back to smaller lags for tiny smoke-test data."""
    feature_sets = [
        ((1, 3, 6, 12, 24, 36, 48), (3, 6, 12, 24, 48)),
        ((1, 3, 6, 12, 24), (3, 6, 12)),
        ((1, 3, 6), (3, 6)),
        ((1,), (2,)),
    ]
    for lags, windows in feature_sets:
        supervised = build_supervised_feature_frame(
            normalized,
            horizon_steps=horizon_steps,
            lags=lags,
            windows=windows,
        )
        if len(supervised) >= 10:
            return supervised, lags, windows
    return supervised, lags, windows


def forecast_points_from_model(
    model: GradientBoostingPowerModel,
    observations: list[TurbineObservation],
    created_at: datetime,
) -> list[dict[str, Any]]:
    """Build forecast point dictionaries from a trained model.

    The model is rolled forward recursively so each predicted step becomes the
    next step's input. This avoids the common failure mode where a single
    point forecast is repeated across the whole horizon and shows up as a flat
    line in the dashboard.
    """
    if not observations:
        msg = "At least one observation is required"
        raise ValueError(msg)

    history = list(observations)
    last_timestamp = max(item.timestamp for item in observations)
    step = timedelta(minutes=10)
    points = []
    count = model.horizon_hours * 6
    last_observation = history[-1]

    for index in range(1, count + 1):
        p50 = model.predict_from_observations(history)
        p50 = max(0.0, p50)
        horizon_scale = 1.0 + (index - 1) / max(1, count - 1)
        p90_margin = model.residual_p90 * horizon_scale

        points.append(
            {
                "timestamp": last_timestamp + step * index,
                "p50": round(p50, 3),
                "p10": round(max(0.0, p50 - p90_margin), 3),
                "p90": round(p50 + p90_margin, 3),
            }
        )

        history.append(
            TurbineObservation(
                timestamp=last_timestamp + step * index,
                active_power_kw=p50,
                wind_speed_mps=last_observation.wind_speed_mps,
                wind_direction_deg=last_observation.wind_direction_deg,
                theoretical_power_kwh=last_observation.theoretical_power_kwh,
            )
        )

    _ = created_at
    return points
