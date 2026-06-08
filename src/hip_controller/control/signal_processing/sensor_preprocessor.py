"""Three-stage sensor preprocessing pipeline: drift removal, filtering, and velocity estimation.

There are two strategies for drift removal, three strategies for filtering, and three strategies for velocity estimation
implemented in the control module, which can be selected and configured in the :class:`PreprocessorConfig` when initializing the :class:`WalkOnController`.

The drift removal strategies include: ``LowPassDriftRemoval`` and ``NotchDriftRemoval``.

The filtering strategies include: ``LowPassFiltering``, ``SogiFllFiltering``, and ``KalmanFiltering``.

The velocity estimation strategies include: ``LowPassVelocityEstimation``, ``DiscreteDerivativeVelocityEstimation``, and ``GyroscopeVelocityEstimation``.
"""

from __future__ import annotations

from hip_controller.control.signal_processing.drift_removal import (
    DriftRemovalStrategy,
    LowPassDriftRemoval,
    NotchDriftRemoval,
)
from hip_controller.control.signal_processing.filtering import (
    FilteringStrategy,
    KalmanFiltering,
    LowPassFiltering,
    SogiFllFiltering,
)
from hip_controller.control.signal_processing.velocity_estimation import (
    DiscreteDerivativeVelocityEstimation,
    GyroscopeVelocityEstimation,
    LowPassVelocityEstimation,
    VelocityEstimationStrategy,
)
from hip_controller.definitions import (
    BASELINE_REMOVAL_SAMPLE_NUM,
    BasicConfig,
    DriftRemovalMethod,
    FilteringMethod,
    PreprocessorConfig,
    SensorSignal,
    VelocityEstimationMethod,
    VelocityInputAngle,
)


class SensorPreprocessor:
    """Two-stage preprocessing pipeline: drift removal → velocity estimation.

    Composes one :class:`DriftRemovalStrategy` and one
    :class:`VelocityEstimationStrategy` into a single ``filter()`` call that
    maps raw sensor readings to a typed :class:`SensorSignal`.

    """

    def __init__(self, basic_config: BasicConfig) -> None:
        """Initialize the sensor pre-processor.

        :param basic_config: controller configuration.
        :return: None
        """
        self._basic_config: BasicConfig = basic_config
        self._velocity_input_angle = PreprocessorConfig.velocity_input_angle

        self._drift_removal: DriftRemovalStrategy
        self._filtering: FilteringStrategy
        self._velocity_estimation: VelocityEstimationStrategy

        self._prev_timestamp: float | None = None
        self._baseline: float = 0.0
        self._baseline_count: int = 0
        self._baseline_sum: float = 0.0

        self._init_strategies()

        # SOGI-FLL quadrature output from the most recent filter() call.
        # Reflects a smoothed velocity-like signal (90 deg phase-shifted from
        # the SOGI in-phase angle). None until the first non-trivial filter()
        # call. Exposed for logging by external code.
        self.last_velocity_surrogate_rad_per_sec: float | None = None

        # Angle as seen *inside* the velocity-estimation LPF, i.e. the LPF's
        # smoothed output that is then differentiated to produce the velocity.
        # For LowPassVelocityEstimation this is the second-order-LPF-filtered
        # version of velocity_input_angle_rad; for other strategies it's the
        # first element of their (angle, velocity) return tuple. Useful for
        # diagnosing where velocity spikes come from. None on first call /
        # after reset.
        self.last_velocity_lpf_angle_rad: float | None = None

        # Output of the drift-removal stage (LPF subtraction or notch),
        # measured between drift removal and SOGI. None on first call /
        # after reset.
        self.last_drift_removed_angle_rad: float | None = None

    def filter(self, raw_signal: SensorSignal) -> SensorSignal:
        """Run one preprocessing step and return a :class:`SensorSignal`.

        :return: Preprocessed :class:`SensorSignal` with timestamp of the current sample [s], raw angle from the sensor [rad] and gyroscope angular rate [rad/s] read from sensor.
        :rtype: SensorSignal
        """
        # Baseline capture by taking avg of first N samples
        if self._baseline_count < BASELINE_REMOVAL_SAMPLE_NUM:
            self._baseline_count += 1
            self._baseline_sum += raw_signal.angle_rad

            if self._baseline_count == BASELINE_REMOVAL_SAMPLE_NUM:
                self._baseline = self._baseline_sum / BASELINE_REMOVAL_SAMPLE_NUM

            raw_signal.angle_rad = 0.0
        else:
            # normal operation: baseline removal
            raw_signal.angle_rad -= self._baseline

        if self._prev_timestamp is None or raw_signal.timestamp is None:
            self._prev_timestamp = raw_signal.timestamp
            return raw_signal

        time_difference = raw_signal.timestamp - self._prev_timestamp

        if time_difference <= 0.0:
            raise ValueError(f"Non-positive time_difference: {time_difference}")

        # check dt too big
        if time_difference > 1.0:
            self.reset()
            return raw_signal

        self._prev_timestamp = raw_signal.timestamp

        angle_no_drift_rad = self._drift_removal.filter(
            raw_angle=raw_signal.angle_rad, time_difference=time_difference
        )
        self.last_drift_removed_angle_rad = angle_no_drift_rad

        angle_out_rad = self._filtering.filter(
            angle_rad=angle_no_drift_rad, time_difference=time_difference
        )
        # Surface the SOGI quadrature for downstream logging / experimentation.
        # The SogiFllFiltering wrapper caches it on every filter() call; other
        # FilteringStrategy implementations (none yet) would need to expose the
        # same attribute.
        self.last_velocity_surrogate_rad_per_sec = getattr(
            self._filtering, "last_quadrature", None
        )

        # See PreprocessorConfig.velocity_input_angle for the trade-off between
        # latency / smoothness (more filtering) and freshness (less filtering).
        if self._velocity_input_angle == VelocityInputAngle.RAW:
            velocity_input_angle_rad = raw_signal.angle_rad
        elif self._velocity_input_angle == VelocityInputAngle.DRIFT_REMOVED:
            velocity_input_angle_rad = angle_no_drift_rad
        else:
            velocity_input_angle_rad = angle_out_rad

        velocity_lpf_angle_rad, velocity_out_rad_per_sec = (
            self._velocity_estimation.filter(
                angle_rad=velocity_input_angle_rad,
                time_difference=time_difference,
                gyro_velocity_rad_per_sec=raw_signal.velocity_rad_per_sec,
            )
        )
        self.last_velocity_lpf_angle_rad = velocity_lpf_angle_rad

        return SensorSignal(
            timestamp=raw_signal.timestamp,
            angle_rad=angle_out_rad,
            velocity_rad_per_sec=velocity_out_rad_per_sec,
        )

    def _init_strategies(self):
        """Get instance of different options of drift removal, filtering, and velocity estimation."""
        if self._basic_config.drift_removal_method == DriftRemovalMethod.LOW_PASS:
            self._drift_removal = LowPassDriftRemoval(
                PreprocessorConfig.drift_removal_second_order_lpf_config
            )

        elif self._basic_config.drift_removal_method == DriftRemovalMethod.NOTCH:
            self._drift_removal = NotchDriftRemoval(
                PreprocessorConfig.drift_removal_notch_config
            )
        else:
            raise ValueError(
                f"Unrecognized drift-removal method: {self._basic_config.drift_removal_method}"
            )

        if self._basic_config.filtering_method == FilteringMethod.SOGI:
            self._filtering = SogiFllFiltering(
                PreprocessorConfig.filtering_sogifll_config
            )

        elif self._basic_config.filtering_method == FilteringMethod.LOW_PASS:
            self._filtering = LowPassFiltering(
                PreprocessorConfig.filtering_lowpass_config
            )

        elif self._basic_config.filtering_method == FilteringMethod.KALMAN:
            self._filtering = KalmanFiltering(
                PreprocessorConfig.filtering_kalman_config
            )

        else:
            raise ValueError(
                f"Unrecognized filtering method: {self._basic_config.filtering_method}"
            )

        if (
            self._basic_config.velocity_estimation_method
            == VelocityEstimationMethod.DISCRETE_DERIVATIVE
        ):
            self._velocity_estimation = DiscreteDerivativeVelocityEstimation()

        elif (
            self._basic_config.velocity_estimation_method
            == VelocityEstimationMethod.LOW_PASS
        ):
            self._velocity_estimation = LowPassVelocityEstimation(
                PreprocessorConfig.velocity_estimation_low_pass_config
            )

        elif (
            self._basic_config.velocity_estimation_method
            == VelocityEstimationMethod.GYROSCOPE
        ):
            self._velocity_estimation = GyroscopeVelocityEstimation()

        else:
            raise ValueError(
                f"Unrecognized velocity-estimation method: {self._basic_config.velocity_estimation_method}"
            )

    def reset(self) -> None:
        """Reset the Signal Preprocessor if exosuit is disconnected or timeout occured.

        :return: None
        """
        self._prev_timestamp = None
        self._baseline: float = 0.0
        self._baseline_count: int = 0
        self._baseline_sum: float = 0.0        
        self.last_velocity_surrogate_rad_per_sec = None
        self.last_velocity_lpf_angle_rad = None
        self.last_drift_removed_angle_rad = None

        self._drift_removal.reset()
        self._filtering.reset()
        self._velocity_estimation.reset()
