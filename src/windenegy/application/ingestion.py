"""Data ingestion application service.

Orchestrates loading SCADA data, validation, normalization,
and preparation for modeling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import polars as pl
from structlog import get_logger

from windenegy.domain.models import DataQualityReport
from windenegy.domain.repository import ChronologicalSplitStrategy, TurbineDataRepository
from windenegy.infrastructure.config import AppConfig

logger = get_logger(__name__)


class DataIngestionService:
    """Service for ingesting and preparing SCADA data.

    This service coordinates between repositories and domain logic
    to produce clean, validated datasets ready for modeling.
    """

    # Column mapping from Kaggle format to internal format
    COLUMN_MAP: ClassVar[dict[str, str]] = {
        "Date/Time": "timestamp",
        "LV ActivePower (kW)": "active_power_kw",
        "Wind Speed (m/s)": "wind_speed_mps",
        "Theoretical_Power_Curve (KWh)": "theoretical_power_kwh",
        "Wind Direction (°)": "wind_direction_deg",
    }

    def __init__(
        self,
        config: AppConfig | None = None,
        repository: TurbineDataRepository | None = None,
    ) -> None:
        """Initialize ingestion service.

        Args:
            config: Application configuration. Uses default if None.
            repository: Data repository. Creates default if None.
        """
        self._config = config or AppConfig()
        self._repository = repository
        self._split_strategy = ChronologicalSplitStrategy(
            train_ratio=self._config.data.train_split,
            validation_ratio=self._config.data.validation_split,
        )
        self._logger = logger.bind(service="DataIngestionService")

    def ingest_raw_csv(
        self,
        csv_path: Path,
        asset_id: str = "T1",
    ) -> DataQualityReport:
        """Ingest raw CSV file and create normalized dataset.

        Args:
            csv_path: Path to raw Kaggle-format CSV.
            asset_id: Asset identifier for the output.

        Returns:
            Data quality report for the ingested data.
        """
        self._logger.info("Starting ingestion", csv_path=str(csv_path), asset_id=asset_id)

        # Load raw data
        df = pl.read_csv(csv_path, try_parse_dates=True)
        self._logger.info("Loaded raw data", rows=len(df), columns=list(df.columns))

        # Normalize columns
        df = self._normalize_columns(df)

        # Parse timestamps (Kaggle format: "DD MM YYYY HH:MM")
        df = df.with_columns(
            pl.col("timestamp")
            .str.to_datetime(format="%d %m %Y %H:%M")
            .dt.cast_time_unit("ms")
            .alias("timestamp")
        )

        # Sort and deduplicate
        df = df.sort("timestamp").unique(subset=["timestamp"], keep="first")

        # Validate
        report = self._validate_dataframe(df)
        self._logger.info("Validation complete", **report.model_dump(mode="json"))

        # Save to processed directory
        output_path = self._config.data.processed_path / f"{asset_id}_normalized.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_path)
        self._logger.info("Saved normalized data", output_path=str(output_path))

        return report

    def create_splits(self, asset_id: str = "T1") -> dict[str, Path]:
        """Create train/validation/test splits from normalized data.

        Args:
            asset_id: Asset identifier.

        Returns:
            Dictionary mapping split name to file path.
        """
        normalized_path = self._config.data.processed_path / f"{asset_id}_normalized.parquet"

        if not normalized_path.exists():
            msg = f"Normalized data not found: {normalized_path}"
            raise FileNotFoundError(msg)

        df = pl.read_parquet(normalized_path)

        # Get time range for splitting
        start = df["timestamp"].min()
        end = df["timestamp"].max()

        if start is None or end is None:
            msg = "Cannot determine time range from data"
            raise ValueError(msg)

        splits = self._split_strategy.create_splits(start, end)
        self._logger.info(
            "Created chronological splits",
            train_start=splits.train_start.isoformat(),
            train_end=splits.train_end.isoformat(),
            val_start=splits.validation_start.isoformat(),
            val_end=splits.validation_end.isoformat(),
            test_start=splits.test_start.isoformat(),
            test_end=splits.test_end.isoformat(),
        )

        # Create split files
        result: dict[str, Path] = {}

        # Train split
        train_df = df.filter(
            (pl.col("timestamp") >= splits.train_start) & (pl.col("timestamp") < splits.train_end)
        )
        train_path = self._config.data.processed_path / f"{asset_id}_train.parquet"
        train_df.write_parquet(train_path)
        result["train"] = train_path
        self._logger.info("Created train split", rows=len(train_df), path=str(train_path))

        # Validation split
        val_df = df.filter(
            (pl.col("timestamp") >= splits.validation_start) & (pl.col("timestamp") < splits.validation_end)
        )
        val_path = self._config.data.processed_path / f"{asset_id}_validation.parquet"
        val_df.write_parquet(val_path)
        result["validation"] = val_path
        self._logger.info("Created validation split", rows=len(val_df), path=str(val_path))

        # Test split
        test_df = df.filter(
            (pl.col("timestamp") >= splits.test_start) & (pl.col("timestamp") <= splits.test_end)
        )
        test_path = self._config.data.processed_path / f"{asset_id}_test.parquet"
        test_df.write_parquet(test_path)
        result["test"] = test_path
        self._logger.info("Created test split", rows=len(test_df), path=str(test_path))

        return result

    def _normalize_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Normalize column names from Kaggle format to internal format."""
        # Only rename columns that exist
        rename_map = {
            old: new for old, new in self.COLUMN_MAP.items() if old in df.columns
        }
        return df.rename(rename_map)

    def _validate_dataframe(self, df: pl.DataFrame) -> DataQualityReport:
        """Validate dataframe and return quality report."""
        total_records = len(df)

        # Missing values
        missing_counts: dict[str, int] = {
            col: int(df[col].null_count()) for col in df.columns
        }

        # Range violations
        range_violations: dict[str, int] = {}

        # Check wind speed non-negative
        if "wind_speed_mps" in df.columns:
            range_violations["wind_speed_mps_negative"] = int(
                (df["wind_speed_mps"] < 0).sum()
            )

        # Check wind direction range
        if "wind_direction_deg" in df.columns:
            out_of_range = (
                (df["wind_direction_deg"] < 0) | (df["wind_direction_deg"] >= 360)
            ).sum()
            range_violations["wind_direction_out_of_range"] = int(out_of_range)

        # Check power non-negative
        if "active_power_kw" in df.columns:
            range_violations["active_power_negative"] = int(
                (df["active_power_kw"] < 0).sum()
            )

        # Check monotonic timestamps
        timestamps = df["timestamp"].to_list()
        monotonic = all(
            timestamps[i] < timestamps[i + 1] for i in range(len(timestamps) - 1)
        )

        # Count duplicates
        unique_count = len(df.unique(subset=["timestamp"]))
        duplicate_timestamps = total_records - unique_count

        return DataQualityReport(
            total_records=total_records,
            missing_counts=missing_counts,
            range_violations=range_violations,
            duplicate_timestamps=duplicate_timestamps,
            monotonic=monotonic,
            start_time=timestamps[0] if timestamps else None,
            end_time=timestamps[-1] if timestamps else None,
        )

    def get_data_summary(self, asset_id: str = "T1") -> dict[str, Any]:
        """Get summary statistics for the dataset.

        Args:
            asset_id: Asset identifier.

        Returns:
            Dictionary with summary statistics.
        """
        normalized_path = self._config.data.processed_path / f"{asset_id}_normalized.parquet"

        if not normalized_path.exists():
            return {"error": f"Normalized data not found for {asset_id}"}

        df = pl.read_parquet(normalized_path)

        summary = {
            "total_records": len(df),
            "time_range": {
                "start": df["timestamp"].min().isoformat() if df["timestamp"].min() else None,
                "end": df["timestamp"].max().isoformat() if df["timestamp"].max() else None,
            },
            "power_stats": {
                "mean": df["active_power_kw"].mean() if "active_power_kw" in df.columns else None,
                "min": df["active_power_kw"].min() if "active_power_kw" in df.columns else None,
                "max": df["active_power_kw"].max() if "active_power_kw" in df.columns else None,
            },
            "wind_stats": {
                "mean": df["wind_speed_mps"].mean() if "wind_speed_mps" in df.columns else None,
                "min": df["wind_speed_mps"].min() if "wind_speed_mps" in df.columns else None,
                "max": df["wind_speed_mps"].max() if "wind_speed_mps" in df.columns else None,
            },
        }

        return summary
