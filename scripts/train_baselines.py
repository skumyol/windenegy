#!/usr/bin/env python3
"""Train and evaluate baseline models.

Usage:
    python scripts/train_baselines.py --data data/processed/T1_train.parquet

Evaluates:
    - Persistence baseline
    - Power curve baseline
    - Rolling mean baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from structlog import get_logger

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from windenegy.application.baseline import (
    PersistenceBaseline,
    PowerCurveBaseline,
    RollingMeanBaseline,
)
from windenegy.application.evaluation import build_metrics
from windenegy.infrastructure.config import AppConfig
from windenegy.infrastructure.logger import configure_logging

logger = get_logger(__name__)


def load_data(train_path: Path, test_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load train and test data."""
    train = pl.read_parquet(train_path)
    test = pl.read_parquet(test_path)
    logger.info("Loaded data", train_rows=len(train), test_rows=len(test))
    return train, test


def evaluate_persistence(train: pl.DataFrame, test: pl.DataFrame) -> dict:
    """Evaluate persistence baseline."""
    model = PersistenceBaseline()
    model.fit(train)

    # For each row in test, predict using previous observation
    actuals = []
    predictions = []

    test_power = test["active_power_kw"].to_numpy()

    # Persistence: predict next value equals current value
    # For test set, shift by 1
    actuals = test_power[1:]
    predictions = test_power[:-1]

    # Calculate horizon (assuming 10-min data)
    # For multi-step, we just use persistence

    metrics = build_metrics(
        model_id="persistence",
        horizon_hours=1,
        actual=actuals,
        predicted=predictions,
    )

    logger.info("Persistence metrics", **metrics.model_dump(mode="json"))

    return {
        "model": "persistence",
        "metrics": metrics.model_dump(mode="json"),
    }


def evaluate_power_curve(train: pl.DataFrame, test: pl.DataFrame) -> dict:
    """Evaluate power curve baseline."""
    model = PowerCurveBaseline()

    # Evaluate theoretical vs actual
    eval_result = model.evaluate(test.to_pandas())

    # For forecasting, we'd use wind speed to lookup expected power
    # Here we compare theoretical curve to actuals

    logger.info("Power curve evaluation", **eval_result)

    return {
        "model": "power_curve",
        "evaluation": eval_result,
    }


def evaluate_rolling_mean(train: pl.DataFrame, test: pl.DataFrame) -> dict:
    """Evaluate rolling mean baseline."""
    model = RollingMeanBaseline(window_hours=1)

    test_power = test["active_power_kw"].to_numpy()

    # Rolling mean with 6-step (1 hour) window
    window = 6
    predictions = []
    actuals = []

    for i in range(window, len(test_power)):
        window_mean = np.mean(test_power[i - window : i])
        predictions.append(window_mean)
        actuals.append(test_power[i])

    metrics = build_metrics(
        model_id="rolling_mean_1h",
        horizon_hours=1,
        actual=actuals,
        predicted=predictions,
    )

    logger.info("Rolling mean metrics", **metrics.model_dump(mode="json"))

    return {
        "model": "rolling_mean_1h",
        "metrics": metrics.model_dump(mode="json"),
    }


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train and evaluate baselines")
    parser.add_argument(
        "--train",
        type=Path,
        default=None,
        help="Training data path",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=None,
        help="Test data path",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Configuration file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for results",
    )

    args = parser.parse_args()
    configure_logging()

    # Load config
    config = AppConfig.from_yaml(args.config) if args.config.exists() else AppConfig()

    # Determine paths
    train_path = args.train or config.data.processed_path / "T1_train.parquet"
    test_path = args.test or config.data.processed_path / "T1_test.parquet"

    if not train_path.exists():
        logger.error("Train file not found", path=str(train_path))
        print(f"❌ Train file not found: {train_path}")
        print("\nRun data pipeline first:")
        print("  python scripts/run_data_pipeline.py")
        return 1

    if not test_path.exists():
        logger.error("Test file not found", path=str(test_path))
        print(f"❌ Test file not found: {test_path}")
        return 1

    # Load data
    train, test = load_data(train_path, test_path)

    print("\n" + "=" * 60)
    print("BASELINE MODEL EVALUATION")
    print("=" * 60)

    results = []

    # Evaluate persistence
    print("\n📊 Evaluating Persistence Baseline...")
    persistence_result = evaluate_persistence(train, test)
    results.append(persistence_result)

    # Evaluate power curve
    print("\n📊 Evaluating Power Curve Baseline...")
    power_curve_result = evaluate_power_curve(train, test)
    results.append(power_curve_result)

    # Evaluate rolling mean
    print("\n📊 Evaluating Rolling Mean Baseline...")
    rolling_result = evaluate_rolling_mean(train, test)
    results.append(rolling_result)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for result in results:
        print(f"\n{result['model']}:")
        if "metrics" in result:
            m = result["metrics"]
            print(f"  MAE:  {m['mae']:.2f} kW")
            print(f"  RMSE: {m['rmse']:.2f} kW")
            print(f"  sMAPE: {m['mape']:.2f}%")
        if "evaluation" in result:
            e = result["evaluation"]
            print(f"  MAE vs actual: {e['mae']:.2f} kW")
            print(f"  Capacity ratio: {e['capacity_ratio']:.3f}")

    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {args.output}")

    print("\n" + "=" * 60)
    print("Next: Train gradient boosting model")
    print("  python scripts/train_gradient_boosting.py")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
