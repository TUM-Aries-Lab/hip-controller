"""Algorithms for the velocity estimation."""

from __future__ import annotations

from abc import ABC, abstractmethod

from hip_controller.definitions import LowPassFilterConfig
from hip_controller.filters.discrete_derivative_filter import (
    DiscreteDerivativeFilter,
)
from hip_controller.filters.second_order_low_pass_filter import (
    SecondOrderLowPassFilter,
)


class VelocityEstimationStrategy(ABC):
    """Abstract base for all velocity estimation algorithms.

    Each concrete strategy consumes the drift-compensated angle (and optionally
    the raw gyroscope reading) and returns a ``(angle_out, velocity_out)`` pair
    so that both signals can be updated coherently.
    """

    @abstractmethod
    def filter(
        self,
        angle_rad: float,
        time_difference: float,
        gyro_velocity_rad_per_sec: float = 0.0,
    ) -> tuple[float, float]:
        """Estimate velocity from the drift-compensated angle.

        :param float angle: Drift-compensated angle [rad].
        :param float time_difference: Time elapsed since the previous sample [s].
        :param float gyro_velocity: Raw gyroscope angular rate [rad/s].

        :return: Tuple of filtered ``(angle_out [rad], velocity_out [rad/s])``.
        :rtype: tuple[float, float]
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the filter to a known initial condition.

        :return: None
        """


class DiscreteDerivativeVelocityEstimation(VelocityEstimationStrategy):
    """Velocity estimation via backward-difference discrete differentiation.

    Computationally minimal; passes the angle through unchanged and returns
    *velocity_discrete_derivative*.
    """

    def __init__(self) -> None:
        """Create a discrete derivative velocity estimation strategy.

        :return: None
        """
        self._discrete_filter: DiscreteDerivativeFilter = DiscreteDerivativeFilter()

    def filter(
        self,
        angle_rad: float,
        time_difference: float,
        gyro_velocity_rad_per_sec: float = 0.0,
    ) -> tuple[float, float]:
        """Estimate velocity via discrete backwards derivative.

        :param float angle: Drift-compensated angle [rad].
        :param float time_difference: Time elapsed since previous sample [s].
        :param float gyro_velocity: Unused in this implementation.

        :return: (angle, velocity_discrete_derivative).
        :rtype: tuple[float, float]
        """
        velocity_discrete_derivative = self._discrete_filter.filter(
            theta=angle_rad, time_difference=time_difference
        )
        return angle_rad, velocity_discrete_derivative

    def reset(self) -> None:
        """Reset the filter to a known initial condition.

        :return: None
        """
        self._discrete_filter.reset()


class LowPassVelocityEstimation(VelocityEstimationStrategy):
    """Velocity estimation via second-order LPF applied to the angle.

    The filter's ``step`` method returns the smoothed angle
    (*angle_filtered*) and its internal derivative (*velocity_derivative_filtered*),
    providing simultaneous noise attenuation on both signals.

    :param lpf: A configured :class:`SecondOrderLowPassFilter` instance.
    """

    def __init__(self, config: LowPassFilterConfig) -> None:
        """Create a low-pass velocity estimation strategy.

        :param LowPassFilterConfig config: Low-pass filter configurations.
        :return: None
        :rtype: None
        """
        self._lpf: SecondOrderLowPassFilter = SecondOrderLowPassFilter(config=config)

    def filter(
        self,
        angle_rad: float,
        time_difference: float,
        gyro_velocity_rad_per_sec: float = 0.0,
    ) -> tuple[float, float]:
        """Estimate velocity by filtering the angle signal.

        :param float angle: Drift-compensated angle [rad].
        :param float time_difference: Time elapsed since previous sample [s].
        :param float gyro_velocity: Unused in this implementation.
        :return: (angle_filtered, velocity_derivative_filtered).
        :rtype: tuple[float, float]
        """
        angle_filtered, velocity_derivative_filtered = self._lpf.step(
            x=angle_rad, time_difference=time_difference
        )
        return angle_filtered, velocity_derivative_filtered

    def reset(self) -> None:
        """Reset the filter to a known initial condition.

        :return: None
        """
        self._lpf.reset()


class GyroscopeVelocityEstimation(VelocityEstimationStrategy):
    """Velocity estimation by reading the sensor's own gyroscope channel.

    Bypasses all numerical differentiation; the raw gyroscope rate is used
    directly as the velocity output, providing the lowest-latency estimate.
    The angle passes through unchanged.
    """

    def filter(
        self,
        angle_rad: float,
        time_difference: float,
        gyro_velocity_rad_per_sec: float = 0.0,
    ) -> tuple[float, float]:
        """Return gyroscope angular velocity directly.

        :param float angle: Drift-compensated angle [rad] (passed through).
        :param float time_difference: Time elapsed since previous sample [s] (unused).
        :param float gyro_velocity: Raw gyroscope angular rate [rad/s].
        :return: (angle, gyro_velocity).
        :rtype: tuple[float, float]
        """
        return angle_rad, gyro_velocity_rad_per_sec

    def reset(self) -> None:
        """Pass without internal filter.

        :return: None
        """
        pass
