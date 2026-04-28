"""Risk detection for wind power operations.

Includes ramp event detection and underproduction risk assessment.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from structlog import get_logger

if TYPE_CHECKING:
    from windenegy.domain.sequence import SequenceSample

logger = get_logger(__name__)


class RiskLevel(str, Enum):
    """Risk severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class RampEvent:
    """A detected power ramp event.

    Attributes:
        start_time: Event start timestamp.
        end_time: Event end timestamp.
        power_change_kw: Absolute power change.
        power_change_pct: Percentage change relative to start.
        duration_minutes: Event duration.
        direction: "up" or "down" ramp.
        risk_level: Severity classification.
    """

    start_time: str
    end_time: str
    power_change_kw: float
    power_change_pct: float
    duration_minutes: int
    direction: str
    risk_level: RiskLevel


@dataclass(frozen=True)
class UnderproductionRisk:
    """Risk of power shortfall.

    Attributes:
        timestamp: Forecast time.
        expected_power_kw: P50 forecast.
        shortfall_threshold_kw: Threshold considered underproduction.
        shortfall_probability: Probability of falling below threshold.
        expected_shortfall_kw: Expected magnitude of shortfall if it occurs.
        risk_level: Severity classification.
    """

    timestamp: str
    expected_power_kw: float
    shortfall_threshold_kw: float
    shortfall_probability: float
    expected_shortfall_kw: float
    risk_level: RiskLevel


class RampDetector:
    """Detect power ramp events in time series data.

    Ramps are defined as rapid changes in power output that could stress
    grid operations or trigger protective controls.
    """

    def __init__(
        self,
        ramp_threshold_kw: float = 500.0,
        ramp_threshold_pct: float = 30.0,
        min_duration_minutes: int = 10,
        capacity_kw: float = 2000.0,
    ) -> None:
        """Initialize detector with thresholds.

        Args:
            ramp_threshold_kw: Absolute change threshold (kW).
            ramp_threshold_pct: Percentage change threshold (%).
            min_duration_minutes: Minimum event duration.
            capacity_kw: Turbine rated capacity.
        """
        self.ramp_threshold_kw = ramp_threshold_kw
        self.ramp_threshold_pct = ramp_threshold_pct
        self.min_duration_minutes = min_duration_minutes
        self.capacity_kw = capacity_kw

    def detect_from_series(
        self,
        timestamps: list[str],
        power_values: list[float],
        interval_minutes: int = 10,
    ) -> list[RampEvent]:
        """Detect ramps in a power time series.

        Args:
            timestamps: ISO timestamps.
            power_values: Power measurements.
            interval_minutes: Data interval.

        Returns:
            List of detected ramp events.
        """
        if len(timestamps) != len(power_values):
            raise ValueError("Length mismatch between timestamps and values")
        if len(power_values) < 2:
            return []

        events: list[RampEvent] = []
        in_ramp = False
        ramp_start_idx = 0
        ramp_start_power = 0.0

        for i in range(1, len(power_values)):
            change_kw = power_values[i] - power_values[i - 1]
            change_pct = (
                (change_kw / power_values[i - 1] * 100)
                if power_values[i - 1] > 0
                else 0
            )

            # Check if significant change
            is_ramp = abs(change_kw) >= self.ramp_threshold_kw or abs(change_pct) >= self.ramp_threshold_pct

            if is_ramp and not in_ramp:
                # Start new ramp
                in_ramp = True
                ramp_start_idx = i - 1
                ramp_start_power = power_values[i - 1]
            elif not is_ramp and in_ramp:
                # End ramp
                duration = (i - ramp_start_idx) * interval_minutes
                if duration >= self.min_duration_minutes:
                    end_power = power_values[i - 1]
                    total_change = end_power - ramp_start_power
                    event = RampEvent(
                        start_time=timestamps[ramp_start_idx],
                        end_time=timestamps[i - 1],
                        power_change_kw=abs(total_change),
                        power_change_pct=abs(total_change / ramp_start_power * 100) if ramp_start_power > 0 else 0,
                        duration_minutes=duration,
                        direction="up" if total_change > 0 else "down",
                        risk_level=self._classify_ramp(abs(total_change)),
                    )
                    events.append(event)
                in_ramp = False

        # Handle ongoing ramp at end
        if in_ramp:
            duration = (len(power_values) - ramp_start_idx) * interval_minutes
            if duration >= self.min_duration_minutes:
                end_power = power_values[-1]
                total_change = end_power - ramp_start_power
                event = RampEvent(
                    start_time=timestamps[ramp_start_idx],
                    end_time=timestamps[-1],
                    power_change_kw=abs(total_change),
                    power_change_pct=abs(total_change / ramp_start_power * 100) if ramp_start_power > 0 else 0,
                    duration_minutes=duration,
                    direction="up" if total_change > 0 else "down",
                    risk_level=self._classify_ramp(abs(total_change)),
                )
                events.append(event)

        logger.info("Ramp detection complete", n_events=len(events))
        return events

    def _classify_ramp(self, change_kw: float) -> RiskLevel:
        """Classify ramp severity."""
        capacity_ratio = change_kw / self.capacity_kw if self.capacity_kw > 0 else 0

        if capacity_ratio > 0.5:
            return RiskLevel.CRITICAL
        if capacity_ratio > 0.3:
            return RiskLevel.HIGH
        if capacity_ratio > 0.15:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW


class UnderproductionAnalyzer:
    """Analyze risk of power shortfall relative to expected production.

    Identifies periods where actual or forecast power is likely to fall
    below operational thresholds.
    """

    def __init__(
        self,
        capacity_kw: float = 2000.0,
        shortfall_threshold_pct: float = 70.0,
    ) -> None:
        """Initialize analyzer.

        Args:
            capacity_kw: Rated capacity.
            shortfall_threshold_pct: Threshold as % of capacity.
        """
        self.capacity_kw = capacity_kw
        self.shortfall_threshold = capacity_kw * (shortfall_threshold_pct / 100)

    def analyze_forecast(
        self,
        timestamps: list[str],
        p50_forecast: list[float],
        p10_forecast: list[float],
        wind_speeds: list[float] | None = None,
    ) -> list[UnderproductionRisk]:
        """Identify underproduction risk from forecast intervals.

        Args:
            timestamps: Forecast timestamps.
            p50_forecast: Point forecasts.
            p10_forecast: Lower bound (pessimistic) forecasts.
            wind_speeds: Optional wind speeds for context.

        Returns:
            List of underproduction risk events.
        """
        risks: list[UnderproductionRisk] = []

        for i, ts in enumerate(timestamps):
            expected = p50_forecast[i]
            pessimistic = p10_forecast[i]

            # Skip if already above threshold
            if pessimistic >= self.shortfall_threshold:
                continue

            # Calculate shortfall probability based on interval position
            if expected >= self.shortfall_threshold:
                # P50 above threshold but p10 below
                # Rough estimate: ~50% probability of shortfall
                prob = 0.5
                expected_shortfall = self.shortfall_threshold - pessimistic
            else:
                # Both below threshold
                prob = 0.8
                expected_shortfall = self.shortfall_threshold - expected

            risk = UnderproductionRisk(
                timestamp=ts,
                expected_power_kw=expected,
                shortfall_threshold_kw=self.shortfall_threshold,
                shortfall_probability=prob,
                expected_shortfall_kw=expected_shortfall,
                risk_level=self._classify_risk(prob, expected_shortfall),
            )
            risks.append(risk)

        logger.info("Underproduction analysis complete", n_risks=len(risks))
        return risks

    def _classify_risk(self, probability: float, magnitude_kw: float) -> RiskLevel:
        """Classify underproduction risk."""
        magnitude_ratio = magnitude_kw / self.capacity_kw if self.capacity_kw > 0 else 0

        if probability > 0.7 and magnitude_ratio > 0.2:
            return RiskLevel.CRITICAL
        if probability > 0.5 and magnitude_ratio > 0.15:
            return RiskLevel.HIGH
        if probability > 0.3:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW


def detect_ramps_in_samples(
    samples: list[SequenceSample],
    interval_minutes: int = 10,
) -> list[RampEvent]:
    """Convenience: detect ramps from sequence samples.

    Args:
        samples: Sequence samples (uses targets as power values).
        interval_minutes: Data interval.

    Returns:
        List of detected ramp events.
    """
    detector = RampDetector()
    timestamps = [s.timestamp for s in samples]
    power_values = [
        float(s.target[0] if hasattr(s.target, "__len__") else s.target)
        for s in samples
    ]
    return detector.detect_from_series(timestamps, power_values, interval_minutes)
