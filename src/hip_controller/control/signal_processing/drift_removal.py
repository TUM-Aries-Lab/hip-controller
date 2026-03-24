"""Drift-removal strategies for sensor preprocessing.

This module defines the strategy interface and two concrete implementations:
- LowPassDriftRemoval
- NotchDriftRemoval
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from hip_controller.control.signal_processing.notch_filter import NotchFilter
from hip_controller.control.signal_processing.second_order_low_pass_filter import (
    SecondOrderLowPassFilter,
)


class DriftRemovalStrategy(ABC):
    """Abstract base for all drift removal algorithms.

    Each concrete strategy accepts the raw angle and the timestamp of the
    sample, and returns the drift-compensated angle.
    """

    @abstractmethod
    def filter(self, raw_angle: float, time_difference: float) -> float:
        """Remove drift from :paramref:`raw_angle` and return the compensated angle.

        :param float raw_angle: Raw angle reading from the sensor [rad].
        :param float timestamp: Wall-clock or monotonic timestamp [s].
        :return: Drift-compensated angle [rad].
        :rtype: float
        """


class LowPassDriftRemoval(DriftRemovalStrategy):
    """Drift removal via second-order LPF subtraction.

    The filter tracks the slow drift component; subtracting its output from the
    raw angle acts as a high-pass and yields *angle_no_drift_low_pass*.

    :param lpf: A configured :class:`SecondOrderLowPassFilter` instance whose
        cut-off frequency sits well below the motion band.
    """

    def __init__(self, lpf: SecondOrderLowPassFilter) -> None:
        """Create a low-pass drift removal strategy.

        :param SecondOrderLowPassFilter lpf: Low-pass filter for drift estimation.
        :return: None
        :rtype: None
        """
        self._low_pass_filter = lpf

    def filter(self, raw_angle: float, time_difference: float) -> float:
        """Execute one drift-removal step.

        :param float raw_angle: Raw angle reading [rad].
        :param float time_difference: Difference dt between current timestamp and previous timestamp.
        :return: Drift-compensated angle [rad].
        :rtype: float
        """
        drift_estimate, _ = self._low_pass_filter.step(
            x=raw_angle, time_difference=time_difference
        )
        angle_no_drift_low_pass = raw_angle - drift_estimate
        return angle_no_drift_low_pass


class NotchDriftRemoval(DriftRemovalStrategy):
    """Drift removal via a notch filter tuned to the drift frequency.

    The notch attenuates the narrow drift band while leaving the rest of the
    spectrum intact, yielding *angle_no_drift_notch*.

    :param notch: A configured :class:`NotchFilter` instance.
    """

    def __init__(self, notch: NotchFilter) -> None:
        """Create a notch-based drift removal strategy.

        :param NotchFilter notch: Notch filter instance.
        :return: None
        :rtype: None
        """
        self._notch = notch

    def filter(self, raw_angle: float, time_difference: float) -> float:
        """Execute one drift-removal step.

        :param float raw_angle: Raw angle reading [rad].
        :param float timestamp: Wall-clock or monotonic timestamp [s].
        :return: Drift-compensated angle [rad].
        :rtype: float
        """
        angle_no_drift_notch = self._notch.filter(raw_value=raw_angle)
        return angle_no_drift_notch
