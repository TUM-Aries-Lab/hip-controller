"""Two-stage sensor preprocessing pipeline: drift removal followed by velocity estimation.

There are two strategies for drift removal and four strategies for velocity estimation implemented in the control module, which can be selected and configured in the :class:`PreprocessorConfig` when initializing the :class:`WalkOnController`.

The drift removal strategies include: ``LowPassDriftRemoval`` and ``NotchDriftRemoval``.

The velocity estimation strategies include: ``SogifllVelocityEstimation``, ``LowPassVelocityEstimation``, ``DiscreteDerivativeVelocityEstimation``, and ``GyroscopeVelocityEstimation``.
"""

from __future__ import annotations

from hip_controller.control.signal_processing.drift_removal import (
    DriftRemovalStrategy,
)
from hip_controller.control.signal_processing.filtering import (
    FilteringStrategy,
    SogiFllFiltering,
)
from hip_controller.control.signal_processing.velocity_estimation import (
    VelocityEstimationStrategy,
)
from hip_controller.definitions import (
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

    def __init__(self, config: PreprocessorConfig) -> None:
        """Initialize the sensor pre-processor.

        :param PreprocessorConfig config: Preprocessor configuration.
        :return: None
        """
        self.config = config
        self._drift_removal: DriftRemovalStrategy = config.drift_removal_strategy
        self._sogi_fll: FilteringStrategy = SogiFllFiltering(
            config=config.filtering_sogifll_config
        )
        self._use_sogi_velocity: bool = (
            config.velocity_estimation_method == VelocityEstimationMethod.SOGI
        )
        self._velocity_estimation: VelocityEstimationStrategy | None = (
            config.velocity_estimation_strategy
        )
        # Drift removal applied to the SOGI quadrature when the SOGI path is
        # active. Always constructed so reset() and config switches at runtime
        # remain consistent.
        self._velocity_drift_removal: DriftRemovalStrategy = (
            config.velocity_drift_removal_strategy
        )

        self._prev_timestamp: float | None = None

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

        # Velocity *before* the optional post-estimation drift-removal notch.
        # For the SOGI path this equals last_velocity_surrogate_rad_per_sec;
        # for other methods this is the raw output of the velocity-estimation
        # strategy. None on first call / after reset.
        self.last_velocity_pre_drift_removal_rad_per_sec: float | None = None

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

        # check dt too big
        if time_difference > 1.0:
            self._drift_removal = self.config.drift_removal_strategy
            self._velocity_estimation = self.config.velocity_estimation_strategy
            self._velocity_drift_removal = self.config.velocity_drift_removal_strategy
            time_difference = 0.01

        self._prev_timestamp = raw_signal.timestamp

        angle_no_drift_rad = self._drift_removal.filter(
            raw_angle=raw_signal.angle_rad, time_difference=time_difference
        )
        self.last_drift_removed_angle_rad = angle_no_drift_rad

        angle_out_rad = self._sogi_fll.filter(
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
            if self.config.velocity_input_angle == VelocityInputAngle.RAW:
                velocity_input_angle_rad = raw_signal.angle_rad
            elif self.config.velocity_input_angle == VelocityInputAngle.DRIFT_REMOVED:
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
        if self.config.apply_velocity_drift_removal:
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
        """
        if not hasattr(self._sogi_fll, "set_config"):
            return
        if class_id == 1:
            self._sogi_fll.set_config(self.config.filtering_sogifll_config_ascend)
        elif class_id == 2:
            self._sogi_fll.set_config(self.config.filtering_sogifll_config_descend)
        else:
            self._sogi_fll.set_config(self.config.filtering_sogifll_config_level)
