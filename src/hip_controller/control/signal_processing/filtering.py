"""Filtering strategies for sensor preprocessing.

This module defines the strategy interface and two concrete implementations:
- SogiFllFiltering
- LowPassFiltering
- KalmanFiltering
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from hip_controller.definitions import LowPassFilterConfig, SogiFllConfig
from hip_controller.filters.second_order_low_pass_filter import (
    SecondOrderLowPassFilter,
)
from hip_controller.filters.sogi_fll_filter import SogiFllFilter


class FilteringStrategy(ABC):
    """Abstract base for all filtering algorithms.

    Each concrete strategy accepts the raw angle and the timestamp of the
    sample, and returns the drift-compensated angle.
    """

    @abstractmethod
    def filter(self, angle_rad: float, time_difference: float) -> float:
        """Filter and return the compensated angle.

        :param float raw_angle: Raw angle reading from the sensor [rad].
        :param float timestamp: Wall-clock or monotonic timestamp [s].
        :return: Drift-compensated angle [rad].
        :rtype: float
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the filter to a known initial condition.

        :return: None
        """

    def set_walking(self, walking: bool) -> None:
        """Toggle walking/paused mode (no-op for strategies that don't track gait state)."""
        return

    def set_config(self, config: SogiFllConfig) -> None:
        """Swap parameter set at runtime (no-op for strategies without per-mode config)."""
        return


class SogiFllFiltering(FilteringStrategy):
    """Velocity estimation via Second-Order Generalized Integrator (SOGI).

    The SOGI resonant structure simultaneously produces a surrogate angle
    (*angle_surrogate*) and an orthogonal quadrature velocity signal
    (*vel_quadrature*).
    """

    def __init__(self, config: SogiFllConfig) -> None:
        """Create a SOGI velocity estimation strategy.

        :param SogiFllConfig config: SOGI-FLL configuration.
        :return: None
        :rtype: None
        """
        self._sogi_filter: SogiFllFilter = SogiFllFilter(config=config)

        # Quadrature output of the inner SOGI on the most recent call. This is
        # a smoothed proxy for velocity (90 deg phase-shifted from
        # angle_surrogate). Exposed so external code can log or gate on it.
        self.last_quadrature: float = 0.0

    def filter(self, angle_rad: float, time_difference: float) -> float:
        """Estimate velocity using SOGI phase-locked structure.

        :param float angle: Drift-compensated angle [rad].
        :param float time_difference: Time elapsed since previous sample [s].
        :param float gyro_velocity: Unused in this implementation.

        :return: (angle_surrogate, velocity_quadrature).
        :rtype: tuple[float, float]
        """
        angle_surrogate, quadrature = self._sogi_filter.filter(
            raw_theta_rad=angle_rad, time_difference=time_difference
        )
        self.last_quadrature = quadrature
        return angle_surrogate

    def reset(self) -> None:
        """Reset the filter to a known initial condition.

        :return: None
        """
        self._sogi_filter.reset()
        self.last_quadrature = 0.0

    def set_walking(self, walking: bool) -> None:
        """Toggle SOGI/FLL between walking-adapt and paused-decay mode.

        Used by stand-still detection upstream so the FLL frequency does
        not drift toward the lower clamp during pauses.
        """
        if walking:
            self._sogi_filter.start_walking()
        else:
            self._sogi_filter.stop_walking()

    def set_config(self, config: SogiFllConfig) -> None:
        """Swap the SOGI-FLL parameter set without resetting state.

        Pass-through to ``SogiFllFilter.set_config()``. Used by
        ``SensorPreprocessor.set_locomotion_mode()`` to switch per-mode
        SOGI tuning at runtime.
        """
        self._sogi_filter.set_config(config)

    @property
    def is_walking(self) -> bool:
        """Pass-through of the inner SOGI filter walking flag."""
        return self._sogi_filter.is_walking


class LowPassFiltering(FilteringStrategy):
    """Drift removal via second-order LPF subtraction.

    The filter tracks the slow drift component; subtracting its output from the
    raw angle acts as a high-pass and yields *angle_no_drift_low_pass*.

    :param lpf: A configured :class:`SecondOrderLowPassFilter` instance whose
        cut-off frequency sits well below the motion band.
    """

    def __init__(self, config: LowPassFilterConfig) -> None:
        """Create a low-pass drift removal strategy.

        :param SecondOrderLowPassFilter lpf: Low-pass filter for drift estimation.
        :return: None
        :rtype: None
        """
        self._low_pass_filter = SecondOrderLowPassFilter(config=config)

    def filter(self, angle_rad: float, time_difference: float) -> float:
        """Execute one drift-removal step.

        :param float raw_angle: Raw angle reading [rad].
        :param float time_difference: Difference dt between current timestamp and previous timestamp.
        :return: Drift-compensated angle [rad].
        :rtype: float
        """
        drift_estimate, _ = self._low_pass_filter.step(
            x=angle_rad, time_difference=time_difference
        )
        return drift_estimate

    def reset(self) -> None:
        """Reset the filter to a known initial condition.

        :return: None
        """
        self._low_pass_filter.reset()
