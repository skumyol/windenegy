#!/usr/bin/env python3
"""Generate calibration report for trained models.

Usage:
    python scripts/generate_calibration_report.py --model-id patchtst-T1-1h
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from windenegy.application.calibration_report import generate_full_report
from windenegy.application.sequence_data import SequenceDatasetBuilder, SequenceDatasetSplitter
from windenegy.domain.sequence import SequenceConfig
from windenegy.infrastructure.config import AppConfig
from windenegy.infrastructure.logger import configure_logging
from windenegy.infrastructure.patchtst_model import PatchTSTModel

import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate calibration report")
    parser.add_argument(
        "--model-id",
        type=str,
        default="patchtst-T1-1h",
        help="Model ID to evaluate",
    )
    parser.add_argument(
        "--processed",
        type=Path,
        default=None,
        help="Processed parquet path",
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
        help="Output path for report",
    )

    args = parser.parse_args()
    configure_logging()

    config = AppConfig.from_yaml(args.config) if args.config.exists() else AppConfig()
    config.ensure_directories()

    processed_path = args.processed or config.data.processed_path / "T1_normalized.parquet"
    output_path = args.output or config.model.metrics_path / f"{args.model_id}_calibration.json"

    # Load model
    model_path = config.model.artifacts_path / args.model_id
    if not model_path.exists():
        logger.error("Model not found", path=str(model_path))
        print(f"❌ Model not found: {model_path}")
        return 1

    model = PatchTSTModel.load(model_path)
    logger.info("Loaded model", model_id=args.model_id)

    # Load and build sequences
    df = pl.read_parquet(processed_path)
    seq_config = SequenceConfig(
        seq_len=144,
        pred_len=6,
    )

    builder = SequenceDatasetBuilder(seq_config)
    samples = builder.build_from_dataframe(df)

    splitter = SequenceDatasetSplitter()
    train, val, test = splitter.split(samples)

    # Generate report
    report = generate_full_report(
        model_id=args.model_id,
        model_type="patchtst",
        model=model,
        val_samples=val,
        test_samples=test,
        horizon_hours=1,
        output_path=output_path,
    )

    # Display summary
    print(f"\n{'=' * 60}")
    print(f"CALIBRATION REPORT: {args.model_id}")
    print(f"{'=' * 60}")

    for horizon, hr in report.horizon_reports.items():
        print(f"\nHorizon: {horizon}")
        print(f"  Samples: {hr.n_samples}")
        print(f"  Target Coverage: {hr.metrics.target_coverage:.0%}")
        print(f"  Empirical Coverage: {hr.metrics.empirical_coverage:.2%}")
        print(f"  Mean Interval Width: {hr.metrics.mean_interval_width:.2f} kW")
        print(f"  Sharpness: {hr.metrics.sharpness:.4f}")
        print(f"  Well Calibrated: {hr.metrics.is_well_calibrated}")
        print(f"  Coverage by Power Regime:")
        for regime, cov in hr.coverage_by_power_bin.items():
            print(f"    {regime}: {cov:.2%}")

    print(f"\n✓ Report saved to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
