"""Algorithms for the velocity estimation."""

from __future__ import annotations

from abc import ABC, abstractmethod

from hip_controller.utils.second_order_low_pass_filter import SecondOrderLowPassFilter


def sogi_fill(angle: float, dt: float) -> tuple[float, float]:
    """Compute SOGI surrogate angle and quadrature velocity.

    :param float angle: Input angle [rad].
    :param float dt: Time elapsed since last sample [s].
    :return: Tuple (angle_surrogate, vel_quadrature).
    :rtype: tuple[float, float]
    """
    ...


def discrete_derivative(angle: float, dt: float) -> float:
    """Compute discrete derivative of angle.

    :param float angle: Input angle [rad].
    :param float dt: Time elapsed since last sample [s].
    :return: Angular velocity [rad/s].
    :rtype: float
    """
    ...


class VelocityEstimationStrategy(ABC):
    """Abstract base for all velocity estimation algorithms.

    Each concrete strategy consumes the drift-compensated angle (and optionally
    the raw gyroscope reading) and returns a ``(angle_out, velocity_out)`` pair
    so that both signals can be updated coherently.
    """

    @abstractmethod
    def step(
        self,
        angle: float,
        dt: float,
        gyro_velocity: float = 0.0,
    ) -> tuple[float, float]:
        """Estimate velocity from the drift-compensated angle.

        :param float angle: Drift-compensated angle [rad].
        :param float dt: Time elapsed since the previous sample [s].
        :param float gyro_velocity: Raw gyroscope angular rate [rad/s].
        :return: Tuple of ``(angle_out [rad], velocity_out [rad/s])``.
        :rtype: tuple[float, float]
        """


class SogiVelocityEstimation(VelocityEstimationStrategy):
    """Velocity estimation via Second-Order Generalized Integrator (SOGI).

    The SOGI resonant structure simultaneously produces a surrogate angle
    (*angle_surrogate*) and an orthogonal quadrature velocity signal
    (*vel_quadrature*).
    """

    def step(
        self,
        angle: float,
        dt: float,
        gyro_velocity: float = 0.0,
    ) -> tuple[float, float]:
        """Estimate velocity using SOGI phase-locked structure.

        :param float angle: Drift-compensated angle [rad].
        :param float dt: Time elapsed since previous sample [s].
        :param float gyro_velocity: Unused in this implementation.
        :return: (angle_surrogate, velocity_quadrature).
        :rtype: tuple[float, float]
        """
        angle_surrogate, vel_quadrature = sogi_fill(angle, dt)
        return angle_surrogate, vel_quadrature


class DiscreteDerivativeVelocityEstimation(VelocityEstimationStrategy):
    """Velocity estimation via backward-difference discrete differentiation.

    Computationally minimal; passes the angle through unchanged and returns
    *velocity_discrete_derivative*.
    """

    def step(
        self,
        angle: float,
        dt: float,
        gyro_velocity: float = 0.0,
    ) -> tuple[float, float]:
        """Estimate velocity via discrete backwards derivative.

        :param float angle: Drift-compensated angle [rad].
        :param float dt: Time elapsed since previous sample [s].
        :param float gyro_velocity: Unused in this implementation.
        :return: (angle, velocity_discrete_derivative).
        :rtype: tuple[float, float]
        """
        velocity_discrete_derivative = discrete_derivative(angle, dt)
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

    def step(
        self,
        angle: float,
        dt: float,
        gyro_velocity: float = 0.0,
    ) -> tuple[float, float]:
        """Estimate velocity by filtering the angle signal.

        :param float angle: Drift-compensated angle [rad].
        :param float dt: Time elapsed since previous sample [s].
        :param float gyro_velocity: Unused in this implementation.
        :return: (angle_filtered, velocity_derivative_filtered).
        :rtype: tuple[float, float]
        """
        angle_filtered, velocity_derivative_filtered = self._lpf.step(angle, dt)
        return angle_filtered, velocity_derivative_filtered


class GyroscopeVelocityEstimation(VelocityEstimationStrategy):
    """Velocity estimation by reading the sensor's own gyroscope channel.

    Bypasses all numerical differentiation; the raw gyroscope rate is used
    directly as the velocity output, providing the lowest-latency estimate.
    The angle passes through unchanged.
    """

    def step(
        self,
        angle: float,
        dt: float,
        gyro_velocity: float = 0.0,
    ) -> tuple[float, float]:
        """Return gyroscope angular velocity directly.

        :param float angle: Drift-compensated angle [rad] (passed through).
        :param float dt: Time elapsed since previous sample [s] (unused).
        :param float gyro_velocity: Raw gyroscope angular rate [rad/s].
        :return: (angle, gyro_velocity).
        :rtype: tuple[float, float]
        """
        return angle, gyro_velocity
