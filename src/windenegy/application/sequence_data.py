"""Sequence dataset generation for time series forecasting.

Creates sliding window sequences from tabular data for models like PatchTST.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from structlog import get_logger

from windenegy.domain.sequence import SequenceConfig, SequenceDatasetMeta, SequenceSample

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)


class SequenceDatasetBuilder:
    """Builds sequence datasets from tabular time series data.

    Creates sliding window samples for sequence-to-sequence forecasting.
    """

    def __init__(self, config: SequenceConfig | None = None) -> None:
        """Initialize with configuration.

        Args:
            config: Sequence configuration. Uses defaults if None.
        """
        self.config = config or SequenceConfig()
        self._feature_names: list[str] = []
        self._target_mean: float = 0.0
        self._target_std: float = 1.0

    def build_from_dataframe(
        self,
        df: pl.DataFrame,
        feature_cols: list[str] | None = None,
        target_col: str = "active_power_kw",
    ) -> list[SequenceSample]:
        """Build sequence samples from a dataframe.

        Args:
            df: Input dataframe with time series data.
            feature_cols: Columns to use as features. Uses all numeric if None.
            target_col: Target column name.

        Returns:
            List of sequence samples.
        """
        if df.is_empty():
            return []

        # Determine feature columns
        if feature_cols is None:
            feature_cols = [
                col for col in df.columns
                if col not in [target_col, "timestamp"]
                and df[col].dtype in (pl.Float64, pl.Float32, pl.Int64)
            ]

        self._feature_names = feature_cols
        logger.info(
            "Building sequences",
            rows=len(df),
            seq_len=self.config.seq_len,
            pred_len=self.config.pred_len,
            n_features=len(feature_cols),
        )

        # Sort by timestamp
        df = df.sort("timestamp")

        # Extract arrays
        feature_array = df.select(feature_cols).to_numpy()
        target_array = df[target_col].to_numpy()
        timestamps = df["timestamp"].to_list()

        # Calculate normalization statistics
        self._target_mean = float(np.mean(target_array))
        self._target_std = float(np.std(target_array)) or 1.0

        # Create sliding window samples
        samples: list[SequenceSample] = []
        total_len = self.config.seq_len + self.config.pred_len

        for i in range(len(df) - total_len + 1):
            input_seq = feature_array[i : i + self.config.seq_len]
            target = target_array[
                i + self.config.seq_len : i + self.config.seq_len + self.config.pred_len
            ]

            # Use timestamp of first prediction step
            pred_timestamp = timestamps[i + self.config.seq_len]

            samples.append(
                SequenceSample(
                    input_sequence=input_seq.copy(),
                    target=target.copy(),
                    timestamp=str(pred_timestamp),
                    asset_id="T1",
                )
            )

        logger.info("Created samples", n_samples=len(samples))
        return samples

    def normalize_target(self, target: np.ndarray) -> np.ndarray:
        """Normalize target values using training statistics."""
        return (target - self._target_mean) / self._target_std

    def denormalize_target(self, normalized: np.ndarray) -> np.ndarray:
        """Denormalize target values back to original scale."""
        return normalized * self._target_std + self._target_mean

    def get_metadata(self) -> SequenceDatasetMeta:
        """Get metadata about the dataset."""
        return SequenceDatasetMeta(
            n_samples=0,  # Will be set after build
            seq_len=self.config.seq_len,
            pred_len=self.config.pred_len,
            n_features=len(self._feature_names),
            feature_names=self._feature_names,
            target_mean=self._target_mean,
            target_std=self._target_std,
        )


class SequenceDatasetSplitter:
    """Split sequence datasets chronologically.

    Maintains temporal ordering to prevent data leakage.
    """

    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> None:
        """Initialize with split ratios.

        Args:
            train_ratio: Proportion for training.
            val_ratio: Proportion for validation.
        """
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

    def split(
        self,
        samples: list[SequenceSample],
    ) -> tuple[list[SequenceSample], list[SequenceSample], list[SequenceSample]]:
        """Split samples chronologically.

        Args:
            samples: List of samples (assumed chronological).

        Returns:
            Tuple of (train, val, test) samples.
        """
        n = len(samples)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        train = samples[:train_end]
        val = samples[train_end:val_end]
        test = samples[val_end:]

        logger.info(
            "Split dataset",
            total=n,
            train=len(train),
            val=len(val),
            test=len(test),
        )

        return train, val, test


def load_and_build_sequences(
    parquet_path: Path,
    config: SequenceConfig | None = None,
) -> tuple[list[SequenceSample], list[SequenceSample], list[SequenceSample]]:
    """Load parquet and build train/val/test sequence datasets.

    Args:
        parquet_path: Path to normalized parquet file.
        config: Sequence configuration.

    Returns:
        Tuple of (train, val, test) samples.
    """
    df = pl.read_parquet(parquet_path)

    builder = SequenceDatasetBuilder(config)
    samples = builder.build_from_dataframe(df)

    splitter = SequenceDatasetSplitter()
    return splitter.split(samples)
