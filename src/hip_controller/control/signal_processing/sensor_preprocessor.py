"""Preprocessor of the controller."""

from __future__ import annotations

from enum import Enum, auto

from hip_controller.control.signal_processing.drift_removal import DriftRemovalStrategy
from hip_controller.control.signal_processing.velocity_estimation import (
    VelocityEstimationStrategy,
)
from hip_controller.definitions import SensorSignal


class SensorPreprocessor:
    """Two-stage preprocessing pipeline: drift removal → velocity estimation.

    Composes one :class:`DriftRemovalStrategy` and one
    :class:`VelocityEstimationStrategy` into a single ``step()`` call that
    maps raw sensor readings to a typed :class:`SensorSignal`.

    :param DriftRemovalStrategy drift_removal: Drift removal strategy.
    :param VelocityEstimationStrategy velocity_estimation: Velocity estimation strategy.
    """

    def __init__(
        self,
        drift_removal: DriftRemovalStrategy,
        velocity_estimation: VelocityEstimationStrategy,
    ) -> None:
        """Initialize the sensor pre-processor."""
        self._drift_removal = drift_removal
        self._velocity_estimation = velocity_estimation

    def filter(
        self,
        raw_angle: float,
        gyro_velocity: float,
        timestamp: float,
    ) -> SensorSignal:
        """Run one preprocessing step and return a :class:`SensorSignal`.

        :param float raw_angle: Raw angle from the sensor [rad].
        :param float gyro_velocity: Gyroscope angular rate [rad/s].
        :param float | None timestamp: Timestamp of the current sample [s].
        :return: Preprocessed :class:`SensorSignal` with angle and velocity.
        :rtype: SensorSignal
        """
        angle_no_drift = self._drift_removal.filter(
            raw_angle=raw_angle, timestamp=timestamp
        )
        angle_out, velocity_out = self._velocity_estimation.filter(
            angle_no_drift, gyro_velocity
        )
        return SensorSignal(
            timestamp=timestamp,
            angle_rad=angle_out,
            velocity_rad_per_sec=velocity_out,
        )


class DriftRemovalMethod(Enum):
    """Methods of drift removal."""

    LOW_PASS = auto()
    NOTCH = auto()


class VelocityEstimationMethod(Enum):
    """Methods of velocity estimation."""

    SOGI = auto()
    DISCRETE_DERIVATIVE = auto()
    LOW_PASS = auto()
    GYROSCOPE = auto()
