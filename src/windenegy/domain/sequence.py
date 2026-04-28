"""Sequence model domain contracts and data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class SequenceSample:
    """A single sequence sample for time series forecasting.

    Attributes:
        input_sequence: Array of shape (seq_len, n_features)
        target: Array of shape (pred_len,) for single target
        timestamp: Timestamp for the prediction point
        asset_id: Identifier for the asset
    """

    input_sequence: np.ndarray
    target: np.ndarray
    timestamp: str
    asset_id: str

    def __post_init__(self) -> None:
        """Validate shapes."""
        if self.input_sequence.ndim != 2:
            raise ValueError(f"input_sequence must be 2D, got {self.input_sequence.ndim}D")
        if self.target.ndim != 1:
            raise ValueError(f"target must be 1D, got {self.target.ndim}D")


class SequenceModel(Protocol):
    """Protocol for sequence forecasting models."""

    def fit(
        self,
        train_sequences: list[SequenceSample],
        val_sequences: list[SequenceSample] | None = None,
    ) -> SequenceModel:
        """Train the model on sequence samples."""
        ...

    def predict(self, input_sequence: np.ndarray) -> np.ndarray:
        """Generate prediction for a single input sequence."""
        ...

    def save(self, path: str) -> None:
        """Save model to disk."""
        ...

    @classmethod
    def load(cls, path: str) -> SequenceModel:
        """Load model from disk."""
        ...


@dataclass(frozen=True)
class SequenceConfig:
    """Configuration for sequence models.

    Attributes:
        seq_len: Input sequence length (number of time steps)
        pred_len: Prediction horizon (number of time steps)
        n_features: Number of input features
        target_col: Name of target column
    """

    seq_len: int = 144  # 24 hours of 10-min data
    pred_len: int = 6   # 1 hour ahead
    n_features: int = 5
    target_col: str = "active_power_kw"

    @property
    def horizon_hours(self) -> int:
        """Convert prediction length to hours (assuming 10-min data)."""
        return self.pred_len // 6


@dataclass(frozen=True)
class SequenceDatasetMeta:
    """Metadata about a sequence dataset."""

    n_samples: int
    seq_len: int
    pred_len: int
    n_features: int
    feature_names: list[str]
    target_mean: float
    target_std: float
