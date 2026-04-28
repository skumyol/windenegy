#!/usr/bin/env python3
"""Train gradient boosting model.

Usage:
    python scripts/train_gradient_boosting.py [--horizon 1|6|24]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from structlog import get_logger

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from windenegy.application.training import train_gradient_boosting_from_csv
from windenegy.infrastructure.config import AppConfig
from windenegy.infrastructure.logger import configure_logging

logger = get_logger(__name__)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train gradient boosting model")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Raw CSV path (default: data/raw/T1.csv)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        choices=[1, 6, 24],
        help="Forecast horizon in hours",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Configuration file",
    )
    parser.add_argument(
        "--asset-id",
        type=str,
        default="T1",
        help="Asset identifier",
    )

    args = parser.parse_args()
    configure_logging()

    # Load config
    config = AppConfig.from_yaml(args.config) if args.config.exists() else AppConfig()
    config.ensure_directories()

    csv_path = args.csv or config.data.raw_path / config.data.scada_filename

    if not csv_path.exists():
        logger.error("CSV file not found", path=str(csv_path))
        print(f"❌ CSV file not found: {csv_path}")
        return 1

    print(f"\n{'=' * 60}")
    print(f"TRAINING GRADIENT BOOSTING MODEL")
    print(f"{'=' * 60}")
    print(f"  Data: {csv_path}")
    print(f"  Horizon: {args.horizon}h")
    print(f"  Asset: {args.asset_id}")
    print(f"{'=' * 60}\n")

    try:
        result = train_gradient_boosting_from_csv(
            csv_path=csv_path,
            model_dir=config.model.artifacts_path,
            metrics_dir=config.model.metrics_path,
            horizon_hours=args.horizon,
            asset_id=args.asset_id,
        )
    except Exception as e:
        logger.error("Training failed", error=str(e))
        print(f"❌ Training failed: {e}")
        return 1

    # Display results
    metadata = result.metadata
    metrics = metadata.metrics

    print(f"\n✓ Training complete!")
    print(f"  Model ID: {result.model_id}")
    print(f"  Features: {len(metadata.feature_schema)}")
    print(f"  Training rows: {metadata.config_snapshot.get('training_rows', 'N/A')}")

    if metrics:
        print(f"\n  Test Set Metrics:")
        print(f"    MAE:   {metrics.mae:.2f} kW")
        print(f"    RMSE:  {metrics.rmse:.2f} kW")
        print(f"    sMAPE: {metrics.mape:.2f}%")
        if metrics.skill_score is not None:
            print(f"    Skill vs Persistence: {metrics.skill_score:.3f}")
        if metrics.coverage_p90 is not None:
            print(f"    P90 Coverage: {metrics.coverage_p90:.3f}")

    print(f"\n  Files saved:")
    print(f"    Model: {result.model_path}")
    print(f"    Metrics: {result.metrics_path}")

    print(f"\n{'=' * 60}")
    print("Model ready for inference!")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
