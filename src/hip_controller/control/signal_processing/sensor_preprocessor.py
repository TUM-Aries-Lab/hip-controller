"""Preprocessor of the controller."""

from __future__ import annotations

from hip_controller.control.signal_processing.drift_removal import (
    DriftRemovalStrategy,
)
from hip_controller.control.signal_processing.velocity_estimation import (
    VelocityEstimationStrategy,
)
from hip_controller.definitions import PreprocessorConfig, SensorSignal


class SensorPreprocessor:
    """Two-stage preprocessing pipeline: drift removal → velocity estimation.

    Composes one :class:`DriftRemovalStrategy` and one
    :class:`VelocityEstimationStrategy` into a single ``filter()`` call that
    maps raw sensor readings to a typed :class:`SensorSignal`.

    """

    def __init__(self, config: PreprocessorConfig) -> None:
        """Initialize the sensor pre-processor.

        :param PreprocessorConfig config: Preprocessor configuration.
        :return: None
        """
        self._drift_removal: DriftRemovalStrategy = config.drift_removal_strategy
        self._velocity_estimation: VelocityEstimationStrategy = (
            config.velocity_estimation_strategy
        )

        self._prev_timestamp: float | None = None

    def filter(self, raw_signal: SensorSignal) -> SensorSignal:
        """Run one preprocessing step and return a :class:`SensorSignal`.

        :param float raw_angle: Raw angle from the sensor [rad].
        :param float gyro_velocity: Gyroscope angular rate [rad/s].
        :param float timestamp: Timestamp of the current sample [s].
        :return: Preprocessed :class:`SensorSignal` with timestamp, angle and velocity.
        :rtype: SensorSignal
        """
        if self._prev_timestamp is None or raw_signal.timestamp is None:
            self._prev_timestamp = raw_signal.timestamp
            return raw_signal

        time_difference = raw_signal.timestamp - self._prev_timestamp
        if time_difference <= 0.0:
            raise ValueError(f"Non-positive time_difference: {time_difference}")

        self._prev_timestamp = raw_signal.timestamp

        angle_no_drift = self._drift_removal.filter(
            raw_angle=raw_signal.angle_rad, time_difference=time_difference
        )

        angle_out, velocity_out = self._velocity_estimation.filter(
            angle=angle_no_drift,
            time_difference=time_difference,
            gyro_velocity=raw_signal.velocity_rad_per_sec,
        )

        return SensorSignal(
            timestamp=raw_signal.timestamp,
            angle_rad=angle_out,
            velocity_rad_per_sec=velocity_out,
        )
