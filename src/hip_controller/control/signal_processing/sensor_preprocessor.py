"""Two-stage sensor preprocessing pipeline: drift removal followed by velocity estimation.

There are two strategies for drift removal and four strategies for velocity estimation implemented in the control module, which can be selected and configured in the :class:`PreprocessorConfig` when initializing the :class:`WalkOnController`.

The drift removal strategies include: ``LowPassDriftRemoval`` and ``NotchDriftRemoval``.

The velocity estimation strategies include: ``SogifllVelocityEstimation``, ``LowPassVelocityEstimation``, ``DiscreteDerivativeVelocityEstimation``, and ``GyroscopeVelocityEstimation``.
"""

from __future__ import annotations

from hip_controller.control.signal_processing.drift_removal import (
    DriftRemovalStrategy,
)
from hip_controller.control.signal_processing.velocity_estimation import (
    SogifllVelocityEstimation,
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
        self.config = config
        self._drift_removal: DriftRemovalStrategy = config.drift_removal_strategy
        self._sogi_fll: VelocityEstimationStrategy = SogifllVelocityEstimation(
            config=config.filtering_sogifll_config
        )
        self._velocity_estimation: VelocityEstimationStrategy = (
            config.velocity_estimation_strategy
        )

        self._prev_timestamp: float | None = None

    def filter(self, raw_signal: SensorSignal) -> SensorSignal:
        """Run one preprocessing step and return a :class:`SensorSignal`.

        :return: Preprocessed :class:`SensorSignal` with timestamp of the current sample [s], raw angle from the sensor [rad] and gyroscope angular rate [rad/s] read from sensor.
        :rtype: SensorSignal
        """
        if self._prev_timestamp is None or raw_signal.timestamp is None:
            self._prev_timestamp = raw_signal.timestamp
            return raw_signal

        time_difference = raw_signal.timestamp - self._prev_timestamp

        if time_difference <= 0.0:
            raise ValueError(f"Non-positive time_difference: {time_difference}")

        # # TODO: check dt too big
        if time_difference > 1.0:
            self._drift_removal = self.config.drift_removal_strategy
            self._velocity_estimation = self.config.velocity_estimation_strategy
            time_difference = 0.01

        self._prev_timestamp = raw_signal.timestamp

        angle_no_drift_rad = self._drift_removal.filter(
            raw_angle=raw_signal.angle_rad, time_difference=time_difference
        )

        angle_out_rad, _ = self._sogi_fll.filter(
            angle_rad=angle_no_drift_rad,
            time_difference=time_difference,
            gyro_velocity_rad_per_sec=raw_signal.velocity_rad_per_sec,
        )

        _, velocity_out_rad_per_sec = self._velocity_estimation.filter(
            angle_rad=angle_out_rad,
            time_difference=time_difference,
            gyro_velocity_rad_per_sec=raw_signal.velocity_rad_per_sec,
        )

        return SensorSignal(
            timestamp=raw_signal.timestamp,
            angle_rad=angle_out_rad,
            velocity_rad_per_sec=velocity_out_rad_per_sec,
        )

    def reset(self) -> None:
        """Reset the Signal Preprocessor if exosuit is disconnected or timeout occured.

        :return: None
        """
        self._prev_timestamp = None
