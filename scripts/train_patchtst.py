#!/usr/bin/env python3
"""Train PatchTST sequence model.

Usage:
    python scripts/train_patchtst.py [--horizon 1|6|24]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from structlog import get_logger

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from windenegy.application.evaluation import build_metrics
from windenegy.application.sequence_data import load_and_build_sequences
from windenegy.domain.models import ModelArtifactMetadata, ModelMetrics
from windenegy.domain.sequence import SequenceConfig
from windenegy.infrastructure.config import AppConfig
from windenegy.infrastructure.logger import configure_logging
from windenegy.infrastructure.patchtst_model import PatchTSTModel
from windenegy.infrastructure.persistence import FileSystemModelRepository

logger = get_logger(__name__)


def train_patchtst(
    processed_path: Path,
    model_dir: Path,
    metrics_dir: Path,
    horizon_hours: int = 1,
    asset_id: str = "T1",
) -> tuple[str, ModelArtifactMetadata, Path]:
    """Train PatchTST model and save artifacts.

    Args:
        processed_path: Path to normalized parquet file.
        model_dir: Directory to save model.
        metrics_dir: Directory to save metrics.
        horizon_hours: Forecast horizon.
        asset_id: Asset identifier.

    Returns:
        Tuple of (model_id, metadata, model_path).
    """
    # Configure sequence parameters
    seq_len = 144  # 24 hours of 10-min data
    pred_len = horizon_hours * 6  # Convert hours to 10-min steps

    config = SequenceConfig(
        seq_len=seq_len,
        pred_len=pred_len,
        target_col="active_power_kw",
    )

    logger.info(
        "Loading sequences",
        path=str(processed_path),
        seq_len=seq_len,
        pred_len=pred_len,
    )

    # Build sequences
    train_samples, val_samples, test_samples = load_and_build_sequences(
        processed_path,
        config=config,
    )

    if len(train_samples) < 100:
        msg = f"Not enough training samples: {len(train_samples)}"
        raise ValueError(msg)

    # Train model
    model = PatchTSTModel(config=config)
    model.fit(train_samples, val_samples)

    # Evaluate on test set
    logger.info("Evaluating on test set", n_test=len(test_samples))

    test_predictions = model.predict_samples(test_samples)
    test_actuals = np.array([s.target for s in test_samples])

    # Handle multi-step predictions - evaluate first step only for metrics
    if test_actuals.ndim > 1:
        if test_actuals.shape[1] == 1:
            test_actuals = test_actuals.flatten()
        else:
            # Multi-step: use first prediction step for metrics
            test_actuals = test_actuals[:, 0]
    if test_predictions.ndim > 1:
        if test_predictions.shape[1] == 1:
            test_predictions = test_predictions.flatten()
        else:
            # Multi-step: use first prediction step for metrics
            test_predictions = test_predictions[:, 0]

    # Build metrics
    # For baseline comparison, we need persistence predictions
    # Persistence = last value of input sequence
    persistence_preds = []
    for sample in test_samples:
        # Last value of active_power_kw from input
        # Assuming it's the first feature (active_power_kw)
        last_power = sample.input_sequence[-1, 0]
        persistence_preds.append(last_power)
    persistence_preds = np.array(persistence_preds)

    # Confidence intervals
    lower = np.maximum(0.0, test_predictions - model.residual_p90)
    upper = test_predictions + model.residual_p90

    metrics = build_metrics(
        model_id="patchtst",
        horizon_hours=horizon_hours,
        actual=test_actuals,
        predicted=test_predictions,
        baseline_predicted=persistence_preds,
        lower=lower,
        upper=upper,
    )

    # Save model
    model_id = f"patchtst-{asset_id}-{horizon_hours}h"
    model_path = model_dir / model_id
    model.save(model_path)

    # Create metadata
    metadata = ModelArtifactMetadata(
        model_type="patchtst",
        model_version=model_id,
        feature_schema={
            "seq_len": str(seq_len),
            "n_features": str(config.n_features),
            "target": config.target_col,
        },
        target=config.target_col,
        horizon_hours=horizon_hours,
        training_window_hours=seq_len // 6,
        metrics=metrics,
        config_snapshot={
            "seq_len": seq_len,
            "pred_len": pred_len,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": len(test_samples),
            "residual_p90": model.residual_p90,
        },
    )

    # Save metadata via repository
    repository = FileSystemModelRepository(model_dir)
    repository.save_model(model_id, model, metadata)

    # Save metrics
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{model_id}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)

    logger.info("Training complete", model_id=model_id, metrics=metrics.to_dict())

    return model_id, metadata, model_path


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train PatchTST sequence model")
    parser.add_argument(
        "--processed",
        type=Path,
        default=None,
        help="Processed parquet path (default: data/processed/T1_normalized.parquet)",
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

    processed_path = args.processed or config.data.processed_path / "T1_normalized.parquet"

    if not processed_path.exists():
        logger.error("Processed file not found", path=str(processed_path))
        print(f"❌ Processed file not found: {processed_path}")
        print("\nRun data pipeline first:")
        print("  python scripts/run_data_pipeline.py")
        return 1

    print(f"\n{'=' * 60}")
    print(f"TRAINING PATCHTST SEQUENCE MODEL")
    print(f"{'=' * 60}")
    print(f"  Data: {processed_path}")
    print(f"  Horizon: {args.horizon}h")
    print(f"  Asset: {args.asset_id}")
    print(f"{'=' * 60}\n")

    try:
        model_id, metadata, model_path = train_patchtst(
            processed_path=processed_path,
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
    metrics = metadata.metrics

    print(f"\n✓ Training complete!")
    print(f"  Model ID: {model_id}")
    print(f"  Seq len: {metadata.config_snapshot.get('seq_len', 'N/A')}")
    print(f"  Train samples: {metadata.config_snapshot.get('train_samples', 'N/A')}")

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
    print(f"    Model: {model_path}")

    print(f"\n{'=' * 60}")
    print("Sequence model ready for inference!")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
