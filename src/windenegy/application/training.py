"""Model training and artifact wiring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

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
    model_path: Path


@dataclass(frozen=True)
class GradientBoostingPowerModel:
    """Serializable model wrapper with its inference feature contract."""

    regressor: GradientBoostingRegressor
    feature_columns: list[str]
    residual_p90: float
    bias_correction: float
    horizon_hours: int
    model_version: str
    lags: tuple[int, ...]
    windows: tuple[int, ...]

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
        latest_features = feature_frame.iloc[[-1]][self.feature_columns]
        prediction = float(self.regressor.predict(latest_features)[0])
        prediction = prediction + self.bias_correction
        return max(0.0, prediction)


def train_gradient_boosting_from_csv(
    csv_path: Path,
    model_dir: Path,
    metrics_dir: Path,
    horizon_hours: int = 1,
    asset_id: str = "T1",
    random_state: int = 42,
) -> TrainingResult:
    """Train and persist a gradient boosting forecaster from a SCADA CSV."""
    raw_frame = pd.read_csv(csv_path)
    normalized = normalize_scada_frame(raw_frame)
    step_minutes = _infer_step_minutes(normalized["timestamp"])
    horizon_steps = max(1, round((horizon_hours * 60) / step_minutes))
    supervised, lags, windows = _build_training_frame(normalized, horizon_steps)
    if len(supervised) < 10:
        msg = (
            "Not enough rows after feature engineering for training. "
            "Use a longer SCADA file or a shorter horizon."
        )
        raise ValueError(msg)

    train, validation, test = _chronological_frame_split(supervised)
    feature_columns = [
        column
        for column in supervised.columns
        if column not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(supervised[column])
    ]

    train_active = train[train["active_power_kw"] > 100]
    val_active = validation[validation["active_power_kw"] > 100]
    test_active = test[test["active_power_kw"] > 100]

    if len(train_active) < 100:
        train_active = train
        val_active = validation
        test_active = test

    sample_weights = np.ones(len(train_active))
    active_mask = train_active["active_power_kw"] > 500
    sample_weights[active_mask] = 2.0

    regressor = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=random_state,
    )
    regressor.fit(train_active[feature_columns], train_active[TARGET_COLUMN], sample_weight=sample_weights)

    validation_predicted = regressor.predict(val_active[feature_columns])
    residuals = val_active[TARGET_COLUMN].to_numpy() - validation_predicted
    bias_correction = float(np.mean(residuals))
    residual_p90 = float(np.quantile(np.abs(residuals), 0.9))
    residual_p90 = max(50.0, residual_p90)

    test_predicted = regressor.predict(test_active[feature_columns]) + bias_correction
    persistence_predicted = test_active["active_power_kw"].to_numpy()
    lower = np.maximum(0.0, test_predicted - residual_p90)
    upper = test_predicted + residual_p90

    metrics = build_metrics(
        model_id="gradient_boosting",
        horizon_hours=horizon_hours,
        actual=test_active[TARGET_COLUMN].to_numpy(),
        predicted=test_predicted,
        baseline_predicted=persistence_predicted,
        lower=lower,
        upper=upper,
    )

    model_id = f"gradient-boosting-{asset_id}-{horizon_hours}h"
    model = GradientBoostingPowerModel(
        regressor=regressor,
        feature_columns=feature_columns,
        residual_p90=residual_p90,
        bias_correction=bias_correction,
        horizon_hours=horizon_hours,
        model_version=model_id,
        lags=lags,
        windows=windows,
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
        },
    )

    repository = FileSystemModelRepository(model_dir)
    model_path = repository.save_model(model_id, model, metadata)

    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{model_id}.json"
    with metrics_path.open("w") as file:
        json.dump(metrics.to_dict(), file, indent=2)

    return TrainingResult(
        model_id=model_id,
        metadata=metadata,
        metrics_path=metrics_path,
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
    """Build forecast point dictionaries from a trained direct model."""
    prediction = model.predict_from_observations(observations)
    last_timestamp = max(item.timestamp for item in observations)
    step = timedelta(minutes=10)
    points = []
    count = model.horizon_hours * 6
    for index in range(1, count + 1):
        points.append(
            {
                "timestamp": last_timestamp + step * index,
                "p50": round(prediction, 3),
                "p10": round(max(0.0, prediction - model.residual_p90), 3),
                "p90": round(prediction + model.residual_p90, 3),
            }
        )
    _ = created_at
    return points
