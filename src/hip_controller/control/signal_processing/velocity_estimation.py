"""Algorithms for the velocity estimation."""

from __future__ import annotations

from abc import ABC, abstractmethod

from hip_controller.control.signal_processing.discrete_derivative_filter import (
    DiscreteDerivativeFilter,
)
from hip_controller.control.signal_processing.second_order_low_pass_filter import (
    SecondOrderLowPassFilter,
)
from hip_controller.control.signal_processing.sogi_fll_filter import SogiFllFilter


class VelocityEstimationStrategy(ABC):
    """Abstract base for all velocity estimation algorithms.

    Each concrete strategy consumes the drift-compensated angle (and optionally
    the raw gyroscope reading) and returns a ``(angle_out, velocity_out)`` pair
    so that both signals can be updated coherently.
    """

    @abstractmethod
    def filter(
        self,
        angle: float,
        time_difference: float,
        gyro_velocity: float = 0.0,
    ) -> tuple[float, float]:
        """Estimate velocity from the drift-compensated angle.

        :param float angle: Drift-compensated angle [rad].
        :param float time_difference: Time elapsed since the previous sample [s].
        :param float gyro_velocity: Raw gyroscope angular rate [rad/s].
        :return: Tuple of filtered ``(angle_out [rad], velocity_out [rad/s])``.
        :rtype: tuple[float, float]
        """


class SogiVelocityEstimation(VelocityEstimationStrategy):
    """Velocity estimation via Second-Order Generalized Integrator (SOGI).

    The SOGI resonant structure simultaneously produces a surrogate angle
    (*angle_surrogate*) and an orthogonal quadrature velocity signal
    (*vel_quadrature*).

    :param sogi_filter: A configured :class:`SogiFllFilter` instance.
    """

    def __init__(self, sogi_filter: SogiFllFilter) -> None:
        """Create a SOGI velocity estimation strategy.

        :param SogiFllFilter sogi_filter: SOGI filter instance.
        :return: None
        :rtype: None
        """
        self._sogi_filter = sogi_filter

    def filter(
        self,
        angle: float,
        time_difference: float,
        gyro_velocity: float = 0.0,
    ) -> tuple[float, float]:
        """Estimate velocity using SOGI phase-locked structure.

        :param float angle: Drift-compensated angle [rad].
        :param float time_difference: Time elapsed since previous sample [s].
        :param float gyro_velocity: Unused in this implementation.
        :return: (angle_surrogate, velocity_quadrature).
        :rtype: tuple[float, float]
        """
        angle_surrogate, vel_quadrature = self._sogi_filter.filter(
            theta=angle, time_difference=time_difference
        )
        return angle_surrogate, vel_quadrature


class DiscreteDerivativeVelocityEstimation(VelocityEstimationStrategy):
    """Velocity estimation via backward-difference discrete differentiation.

    Computationally minimal; passes the angle through unchanged and returns
    *velocity_discrete_derivative*.

    :param discrete_filter: A configured :class:`DiscreteDerivativeFilter` instance.
    """

    def __init__(self, discrete_filter: DiscreteDerivativeFilter) -> None:
        """Create a discrete derivative velocity estimation strategy.

        :param DiscreteDerivativeFilter discrete_filter: Discrete derivative filter instance.
        :return: None
        :rtype: None
        """
        self._discrete_filter: DiscreteDerivativeFilter = discrete_filter

    def filter(
        self,
        angle: float,
        time_difference: float,
        gyro_velocity: float = 0.0,
    ) -> tuple[float, float]:
        """Estimate velocity via discrete backwards derivative.

        :param float angle: Drift-compensated angle [rad].
        :param float time_difference: Time elapsed since previous sample [s].
        :param float gyro_velocity: Unused in this implementation.
        :return: (angle, velocity_discrete_derivative).
        :rtype: tuple[float, float]
        """
        velocity_discrete_derivative = self._discrete_filter.filter(
            theta=angle, time_difference=time_difference
        )
        return angle, velocity_discrete_derivative


class LowPassVelocityEstimation(VelocityEstimationStrategy):
    """Velocity estimation via second-order LPF applied to the angle.

    The filter's ``step`` method returns the smoothed angle
    (*angle_filtered*) and its internal derivative (*velocity_derivative_filtered*),
    providing simultaneous noise attenuation on both signals.

    :param lpf: A configured :class:`SecondOrderLowPassFilter` instance.
    """

    def __init__(self, lpf: SecondOrderLowPassFilter) -> None:
        """Create a low-pass velocity estimation strategy.

        :param SecondOrderLowPassFilter lpf: Low-pass filter instance.
        :return: None
        :rtype: None
        """
        self._lpf = lpf

    def filter(
        self,
        angle: float,
        time_difference: float,
        gyro_velocity: float = 0.0,
    ) -> tuple[float, float]:
        """Estimate velocity by filtering the angle signal.

        :param float angle: Drift-compensated angle [rad].
        :param float time_difference: Time elapsed since previous sample [s].
        :param float gyro_velocity: Unused in this implementation.
        :return: (angle_filtered, velocity_derivative_filtered).
        :rtype: tuple[float, float]
        """
        angle_filtered, velocity_derivative_filtered = self._lpf.step(
            x=angle, time_difference=time_difference
        )
        return angle_filtered, velocity_derivative_filtered


class GyroscopeVelocityEstimation(VelocityEstimationStrategy):
    """Velocity estimation by reading the sensor's own gyroscope channel.

    Bypasses all numerical differentiation; the raw gyroscope rate is used
    directly as the velocity output, providing the lowest-latency estimate.
    The angle passes through unchanged.
    """

    def filter(
        self,
        angle: float,
        time_difference: float,
        gyro_velocity: float = 0.0,
    ) -> tuple[float, float]:
        """Return gyroscope angular velocity directly.

        :param float angle: Drift-compensated angle [rad] (passed through).
        :param float time_difference: Time elapsed since previous sample [s] (unused).
        :param float gyro_velocity: Raw gyroscope angular rate [rad/s].
        :return: (angle, gyro_velocity).
        :rtype: tuple[float, float]
        """
        return angle, gyro_velocity
