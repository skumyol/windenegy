#!/usr/bin/env python3
"""Run the complete data ingestion and validation pipeline.

Usage:
    python scripts/run_data_pipeline.py [--config configs/default.yaml]

This script:
1. Loads raw SCADA CSV
2. Normalizes column names
3. Validates data quality
4. Creates chronological train/validation/test splits
5. Outputs summary report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from structlog import get_logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from windenegy.application.ingestion import DataIngestionService
from windenegy.infrastructure.config import AppConfig
from windenegy.infrastructure.logger import configure_logging

logger = get_logger(__name__)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run data ingestion pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Configuration file path",
    )
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=None,
        help="Override raw CSV path",
    )
    parser.add_argument(
        "--asset-id",
        type=str,
        default="T1",
        help="Asset identifier",
    )
    parser.add_argument(
        "--skip-splits",
        action="store_true",
        help="Skip creating train/validation/test splits",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    # Configure logging
    configure_logging()

    # Load configuration
    config = AppConfig.from_yaml(args.config) if args.config.exists() else AppConfig()
    config.ensure_directories()

    # Initialize service
    service = DataIngestionService(config=config)

    # Determine raw CSV path
    raw_csv = args.raw_csv or config.data.raw_path / config.data.scada_filename

    logger.info("Starting data pipeline", raw_csv=str(raw_csv), asset_id=args.asset_id)

    # Check if file exists
    if not raw_csv.exists():
        logger.error("Raw CSV file not found", path=str(raw_csv))
        print(f"\n❌ Error: Raw CSV file not found: {raw_csv}")
        print("\nPlease ensure data/raw/T1.csv exists.")
        print("You can download it from:")
        print("  https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset")
        return 1

    # Step 1: Ingest and validate
    print("\n" + "=" * 60)
    print("STEP 1: Ingesting and Validating Raw Data")
    print("=" * 60)

    try:
        report = service.ingest_raw_csv(raw_csv, asset_id=args.asset_id)
    except Exception as e:
        logger.error("Ingestion failed", error=str(e))
        print(f"\n❌ Ingestion failed: {e}")
        return 1

    print(f"\n✓ Ingestion complete")
    print(f"  Total records: {report.total_records:,}")
    print(f"  Time range: {report.start_time} to {report.end_time}")
    print(f"  Monotonic: {'✓' if report.monotonic else '✗'}")
    print(f"  Duplicates: {report.duplicate_timestamps}")

    if report.range_violations:
        print(f"\n  Range violations:")
        for check, count in report.range_violations.items():
            status = "✗" if count > 0 else "✓"
            print(f"    {status} {check}: {count}")

    # Check for missing values
    missing = {k: v for k, v in report.missing_counts.items() if v > 0}
    if missing:
        print(f"\n  Missing values:")
        for col, count in missing.items():
            print(f"    {col}: {count}")
    else:
        print(f"\n  ✓ No missing values")

    # Step 2: Create splits (if not skipped)
    if not args.skip_splits:
        print("\n" + "=" * 60)
        print("STEP 2: Creating Train/Validation/Test Splits")
        print("=" * 60)

        try:
            split_paths = service.create_splits(asset_id=args.asset_id)
        except Exception as e:
            logger.error("Split creation failed", error=str(e))
            print(f"\n❌ Split creation failed: {e}")
            return 1

        print(f"\n✓ Splits created:")
        for split_name, path in split_paths.items():
            print(f"  {split_name}: {path}")

    # Step 3: Generate summary
    print("\n" + "=" * 60)
    print("STEP 3: Data Summary")
    print("=" * 60)

    summary = service.get_data_summary(asset_id=args.asset_id)

    if "error" in summary:
        print(f"\n✗ {summary['error']}")
        return 1

    print(f"\n  Records: {summary['total_records']:,}")
    print(f"  Time range: {summary['time_range']['start']} to {summary['time_range']['end']}")

    if summary.get("power_stats"):
        ps = summary["power_stats"]
        print(f"\n  Power (kW):")
        print(f"    Mean: {ps['mean']:.1f}")
        print(f"    Range: [{ps['min']:.1f}, {ps['max']:.1f}]")

    if summary.get("wind_stats"):
        ws = summary["wind_stats"]
        print(f"\n  Wind Speed (m/s):")
        print(f"    Mean: {ws['mean']:.2f}")
        print(f"    Range: [{ws['min']:.2f}, {ws['max']:.2f}]")

    # Output as JSON if requested
    if args.json_output:
        output = {
            "ingestion": report.model_dump(mode="json"),
            "summary": summary,
        }
        print("\n" + json.dumps(output, indent=2))

    print("\n" + "=" * 60)
    print("✓ Data pipeline complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
