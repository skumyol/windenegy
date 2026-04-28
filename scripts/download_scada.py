#!/usr/bin/env python3
"""Download the Kaggle Turkey Wind Turbine SCADA dataset.

Usage:
    python scripts/download_scada.py [--kaggle-cred ~/.kaggle/kaggle.json]

The dataset will be saved to data/raw/T1.csv

Requires:
    - kaggle CLI: pip install kaggle
    - Kaggle API credentials (see https://github.com/Kaggle/kaggle-api#api-credentials)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import NoReturn

import structlog

logger = structlog.get_logger(__name__)

DATASET_NAME = "berkerisen/wind-turbine-scada-dataset"
OUTPUT_FILE = "T1.csv"


def download_with_kaggle_cli(output_dir: Path) -> bool:
    """Download using kaggle CLI.

    Args:
        output_dir: Directory to save the dataset.

    Returns:
        True if successful, False otherwise.
    """
    import subprocess

    try:
        logger.info("Downloading with kaggle CLI", dataset=DATASET_NAME)

        # Create temp directory for download
        temp_dir = output_dir / ".kaggle_download"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Download dataset
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", DATASET_NAME, "-p", str(temp_dir)],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            logger.error("Kaggle CLI failed", stderr=result.stderr)
            return False

        # Find and extract zip file
        zip_files = list(temp_dir.glob("*.zip"))
        if not zip_files:
            logger.error("No zip file found after download")
            return False

        zip_path = zip_files[0]
        logger.info("Extracting", zip_file=str(zip_path))

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(temp_dir)

        # Find T1.csv in extracted files
        csv_files = list(temp_dir.rglob("T1.csv"))
        if not csv_files:
            # Try any CSV
            csv_files = list(temp_dir.rglob("*.csv"))

        if not csv_files:
            logger.error("No CSV file found in extracted archive")
            return False

        # Copy to output directory
        source = csv_files[0]
        destination = output_dir / OUTPUT_FILE
        shutil.copy2(source, destination)

        # Cleanup
        shutil.rmtree(temp_dir)

        logger.info("Download complete", destination=str(destination))
        return True

    except FileNotFoundError:
        logger.error("kaggle CLI not found. Install with: pip install kaggle")
        return False
    except subprocess.SubprocessError as e:
        logger.error("Subprocess error", error=str(e))
        return False


def download_with_requests(output_dir: Path) -> bool:
    """Alternative download method using direct HTTP request.

    Note: Kaggle requires authentication, so this is only a fallback
    that provides instructions.

    Args:
        output_dir: Directory where file should be saved.

    Returns:
        False (this method just provides instructions).
    """
    logger.warning("Direct download not available for Kaggle datasets")
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD REQUIRED")
    print("=" * 60)
    print(f"\nDataset: {DATASET_NAME}")
    print(f"\n1. Visit: https://www.kaggle.com/datasets/{DATASET_NAME}")
    print("2. Click 'Download' to get the ZIP file")
    print(f"3. Extract T1.csv to: {output_dir / OUTPUT_FILE}")
    print("\nAlternative - use Kaggle CLI:")
    print("  pip install kaggle")
    print("  kaggle datasets download -d berkerisen/wind-turbine-scada-dataset")
    print("  unzip wind-turbine-scada-dataset.zip")
    print(f"  mv T1.csv {output_dir / OUTPUT_FILE}")
    print("=" * 60 + "\n")
    return False


def verify_download(output_dir: Path) -> bool:
    """Verify the downloaded file exists and has expected structure.

    Args:
        output_dir: Directory containing the CSV file.

    Returns:
        True if file is valid, False otherwise.
    """
    file_path = output_dir / OUTPUT_FILE

    if not file_path.exists():
        logger.error("File not found", path=str(file_path))
        return False

    # Check file size (should be several MB)
    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb < 1:
        logger.warning("File seems small", size_mb=size_mb)

    # Verify CSV structure
    try:
        import pandas as pd

        df = pd.read_csv(file_path, nrows=5)
        expected_cols = [
            "Date/Time",
            "LV ActivePower (kW)",
            "Wind Speed (m/s)",
            "Theoretical_Power_Curve (KWh)",
            "Wind Direction (°)",
        ]

        missing = set(expected_cols) - set(df.columns)
        if missing:
            logger.error("Missing expected columns", missing=list(missing))
            return False

        logger.info(
            "File verified",
            path=str(file_path),
            size_mb=round(size_mb, 2),
            columns=list(df.columns),
        )
        return True

    except Exception as e:
        logger.error("Failed to verify CSV", error=str(e))
        return False


def main() -> NoReturn:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download Kaggle Turkey Wind Turbine SCADA dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory (default: data/raw)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing file, don't download",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing file",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_file = args.output_dir / OUTPUT_FILE

    # Verify only mode
    if args.verify_only:
        success = verify_download(args.output_dir)
        sys.exit(0 if success else 1)

    # Check if file already exists
    if output_file.exists() and not args.force:
        logger.info("File already exists", path=str(output_file))
        if verify_download(args.output_dir):
            logger.info("Existing file is valid, skipping download")
            sys.exit(0)
        else:
            logger.warning("Existing file is invalid, re-downloading")

    # Try download methods
    success = False

    if shutil.which("kaggle"):
        success = download_with_kaggle_cli(args.output_dir)
    else:
        logger.info("Kaggle CLI not found, showing manual instructions")
        success = download_with_requests(args.output_dir)

    if success:
        success = verify_download(args.output_dir)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
