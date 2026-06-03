"""Two-stage sensor preprocessing pipeline: drift removal followed by velocity estimation.

There are two strategies for drift removal and four strategies for velocity estimation implemented in the control module, which can be selected and configured in the :class:`PreprocessorConfig` when initializing the :class:`WalkOnController`.

The drift removal strategies include: ``LowPassDriftRemoval`` and ``NotchDriftRemoval``.

The velocity estimation strategies include: ``SogifllVelocityEstimation``, ``LowPassVelocityEstimation``, ``DiscreteDerivativeVelocityEstimation``, and ``GyroscopeVelocityEstimation``.
"""

from __future__ import annotations

from loguru import logger

from hip_controller.control.signal_processing.drift_removal import (
    DriftRemovalStrategy,
    LowPassDriftRemoval,
    NotchDriftRemoval,
)
from hip_controller.control.signal_processing.filtering import (
    FilteringStrategy,
    LowPassFiltering,
    SogiFllFiltering,
    KalmanFiltering,
    
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

        :param PreprocessorConfig config: Preprocessor configuration.
        :return: None
        """
        self._basic_config = basic_config
        self._filtering: FilteringStrategy
        self._drift_removal: DriftRemovalStrategy
        self._sogi_fll: SogiFllFiltering = SogiFllFiltering(
            config=PreprocessorConfig.filtering_sogifll_config
        )
        self._use_sogi_velocity: bool = (
            basic_config.velocity_estimation_method == VelocityEstimationMethod.SOGI
        )
        self._velocity_estimation: VelocityEstimationStrategy | None
        # Drift removal applied to the SOGI quadrature when the SOGI path is
        # active. Always constructed so reset() and config switches at runtime
        # remain consistent.
        self._velocity_drift_removal: DriftRemovalStrategy
        self._velocity_input_angle: VelocityInputAngle = PreprocessorConfig.velocity_input_angle
        self._apply_velocity_drift_removal: bool = PreprocessorConfig.apply_velocity_drift_removal
        self._prev_timestamp: float | None = None
        self._baseline: float = 0.0
        self._baseline_count: int = 0
        self._baseline_sum: float = 0.0

        self.__init_strategies__()

    def filter(self, raw_signal: SensorSignal) -> SensorSignal:
        """Run one preprocessing step and return a :class:`SensorSignal`.

        :return: Preprocessed :class:`SensorSignal` with timestamp of the current sample [s], raw angle from the sensor [rad] and gyroscope angular rate [rad/s] read from sensor.
        :rtype: SensorSignal
        """
        # Baseline capture by taking avg of first N samples
        if self._baseline_count < BASELINE_REMOVAL_SAMPLE_NUM:
            self._baseline_count += 1
            self._baseline_sum += raw_signal.angle_rad
            raw_signal.angle_rad = 0.0
        elif self._baseline_count == BASELINE_REMOVAL_SAMPLE_NUM:
            self._baseline = self._baseline_sum / BASELINE_REMOVAL_SAMPLE_NUM
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
            self._sogi_fll, "last_quadrature", None
        )

        if self._use_sogi_velocity:
            # SOGI path: take the quadrature already produced by the angle-stage
            # SOGI-FLL. No second SOGI runs.
            velocity_pre_drift_rad_per_sec = (
                self.last_velocity_surrogate_rad_per_sec or 0.0
            )
            self.last_velocity_lpf_angle_rad = None
        else:
            # See PreprocessorConfig.velocity_input_angle for the trade-off
            # between latency / smoothness (more filtering) and freshness
            # (less filtering).
            if self._velocity_input_angle == VelocityInputAngle.RAW:
                velocity_input_angle_rad = raw_signal.angle_rad
            elif self._velocity_input_angle == VelocityInputAngle.DRIFT_REMOVED:
                velocity_input_angle_rad = angle_no_drift_rad
            else:
                velocity_input_angle_rad = angle_out_rad

            # Outside the SOGI path, a velocity-estimation strategy must be
            # configured (the `else` branch above is only reached when
            # _use_sogi_velocity is False, which in turn requires the config
            # to provide a strategy at construction time).
            assert self._velocity_estimation is not None
            velocity_lpf_angle_rad, velocity_pre_drift_rad_per_sec = (
                self._velocity_estimation.filter(
                    angle_rad=velocity_input_angle_rad,
                    time_difference=time_difference,
                    gyro_velocity_rad_per_sec=raw_signal.velocity_rad_per_sec,
                )
            )
            self.last_velocity_lpf_angle_rad = velocity_lpf_angle_rad

        self.last_velocity_pre_drift_removal_rad_per_sec = (
            velocity_pre_drift_rad_per_sec
        )

        # Optional post-estimation drift removal applied uniformly to every
        # velocity_estimation_method.
        if self._apply_velocity_drift_removal:
            velocity_out_rad_per_sec = self._velocity_drift_removal.filter(
                raw_angle=velocity_pre_drift_rad_per_sec,
                time_difference=time_difference,
            )
        else:
            velocity_out_rad_per_sec = velocity_pre_drift_rad_per_sec

        return SensorSignal(
            timestamp=raw_signal.timestamp,
            angle_rad=angle_out_rad,
            velocity_rad_per_sec=velocity_out_rad_per_sec,
        )

    def __init_strategies__(self):
        """Get instance of different options of drift removal, filtering, and velocity estimation."""
        # drift removal
        if self._basic_config.drift_removal_method == DriftRemovalMethod.LOW_PASS:
            self._drift_removal = LowPassDriftRemoval(
                PreprocessorConfig.drift_removal_second_order_lpf_config
            )

        elif self._basic_config.drift_removal_method == DriftRemovalMethod.NOTCH:
            self._drift_removal = NotchDriftRemoval(
                PreprocessorConfig.drift_removal_notch_config
            )
        else:
            logger.warning("Selected method does not exist.")

        # filtering
        if self._basic_config.filtering_method == FilteringMethod.SOGI:
            self._filtering = SogiFllFiltering(
                PreprocessorConfig.filtering_sogifll_config
            )

        elif self._basic_config.filtering_method == FilteringMethod.LOW_PASS:
            self._filtering = LowPassFiltering(
                PreprocessorConfig.filtering_second_order_lpf_config
            )

        elif self._basic_config.filtering_method == FilteringMethod.KALMAN:
            self._filtering = KalmanFiltering(
                PreprocessorConfig.filtering_kalman_config
            )

        else:
            logger.warning("Selected method does not exist.")

        # velocity estimation
        if (
            self._basic_config.velocity_estimation_method
            == VelocityEstimationMethod.SOGI
        ):
            self._velocity_estimation = None

        elif (
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
            logger.warning("Selected method does not exist.")

        # velocity drift removal
        self._velocity_drift_removal = NotchDriftRemoval(PreprocessorConfig.velocity_drift_removal_notch_config)

    def reset(self) -> None:
        """Reset the Signal Preprocessor if exosuit is disconnected or timeout occured.

        :return: None
        """
        self._prev_timestamp = None
        self.last_velocity_surrogate_rad_per_sec = None
        self.last_velocity_lpf_angle_rad = None
        self.last_drift_removed_angle_rad = None
        self.last_velocity_pre_drift_removal_rad_per_sec = None

        self._drift_removal.reset()
        self._sogi_fll.reset()
        self._velocity_drift_removal.reset()
        if self._velocity_estimation is not None:
            self._velocity_estimation.reset()

    def set_walking_mode(self, walking: bool) -> None:
        """Tell the SOGI/FLL whether the user is actively walking.

        Called by the upstream stand-still detector in :class:`WalkOnController`.
        When walking=False, the FLL freezes its frequency tracking (states
        decay slowly toward initial guess) instead of drifting toward the
        lower clamp under noise-only input during pauses.
        """
        if hasattr(self._sogi_fll, "set_walking"):
            self._sogi_fll.set_walking(walking)

    def set_locomotion_mode(self, class_id: int) -> None:
        """Swap the SOGI-FLL config to the variant tuned for this locomotion mode.

        ``class_id`` matches the TCN classifier output: 0=Level, 1=Ascend,
        2=Descend. The corresponding `filtering_sogifll_config_*` field from
        :class:`PreprocessorConfig` is selected and pushed into the active
        SOGI filter. SOGI state (in-phase, quadrature, omega_est, etc.) is
        preserved; only the parameter values change, so the FLL re-adapts
        smoothly across a mode change rather than re-locking from scratch.
        Unknown class_ids fall back to the LEVEL config.

        DESCEND -> non-DESCEND exception: the in-phase / quadrature get
        cleared via ``clear_state_keep_frequency()``. The descent gait
        shape (brief flexion peak + long controlled-descent slope) charges
        the SOGI oscillator with harmonics that don't match LG/ASC
        kinematics; preserved as-is, those harmonics drive a phantom
        oscillation for 1-2 strides after stepping onto flat ground and
        time the motor command against the wrong phase of the user's
        actual stride. Frequency estimate is intentionally kept so the
        FLL stays locked on the cadence; only the SOGI oscillator state
        is wiped. See raw-vs-filtered IMU overlays on
        savedData_Thu_Jun_25_15-42-00_2026.csv for the diagnostic.
        """
        if not hasattr(self._sogi_fll, "set_config"):
            return
        if class_id == 1:
            self._sogi_fll.set_config(PreprocessorConfig.filtering_sogifll_config_ascend)
        elif class_id == 2:
            self._sogi_fll.set_config(PreprocessorConfig.filtering_sogifll_config_descend)
        else:
            self._sogi_fll.set_config(PreprocessorConfig.filtering_sogifll_config_level)

        if self._current_mode_id == 2 and class_id != 2:
            self._sogi_fll.clear_state_keep_frequency()

        self._current_mode_id = class_id

    def set_demo_mode(self) -> None:
        """Swap the SOGI-FLL config to the demo (classification-free) tuning.

        Demo mode bypasses the TCN and applies assist via a fixed
        LUT-and-gain pipeline. Because there is no locomotion class,
        ``set_locomotion_mode()`` isn't called during demo runs and the
        SOGI would otherwise stay on whatever config was last applied
        (typically LEVEL from ``WalkOnController.__init__``). This
        method pushes the demo-tuned ``filtering_sogifll_config_demo``
        into the filter for lower phase lag on the demo signal path.

        SOGI state (in-phase, quadrature, omega_est, frequency_estimate,
        confidence_state) is preserved. Idempotent -- safe to call on
        every demo re-entry.
        """
        if not hasattr(self._sogi_fll, "set_config"):
            return
        self._sogi_fll.set_config(PreprocessorConfig.filtering_sogifll_config_demo)
