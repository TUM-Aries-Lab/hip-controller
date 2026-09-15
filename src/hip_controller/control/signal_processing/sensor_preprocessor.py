"""Three-stage sensor preprocessing pipeline: drift removal, filtering, velocity estimation.

Which strategy runs at each stage is selected on :class:`BasicConfig`; the
parameter sets they are built from live in :class:`PreprocessorConfig`.

The drift removal strategies include: ``LowPassDriftRemoval`` and ``NotchDriftRemoval``.

The filtering strategies include: ``SogiFllFiltering``, ``LowPassFiltering`` and
``KalmanFiltering``.

The velocity estimation strategies include the SOGI quadrature path (no separate
estimator), ``LowPassVelocityEstimation``, ``DiscreteDerivativeVelocityEstimation``,
and ``GyroscopeVelocityEstimation``.

There is exactly one angle-stage filter instance, ``_filtering``. The runtime
mode switches (:meth:`set_locomotion_mode`, :meth:`set_demo_mode`,
:meth:`set_walking_mode`) reconfigure that same object. Holding a second filter
for the mode switches to talk to would make per-mode retuning a silent no-op and
zero the SOGI velocity path -- see ``sensor_preprocessor_test.py``.
"""

from __future__ import annotations

from hip_controller.control.signal_processing.baseline_removal import (
    BaselineRemoval,
)
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
    AssistMode,
    BasicConfig,
    DriftRemovalMethod,
    FilteringMethod,
    PreprocessorConfig,
    SensorSignal,
    SogiFllConfig,
    VelocityEstimationMethod,
    VelocityInputAngle,
)

# A sample gap longer than this is treated as a dropout rather than a real dt:
# the stateful stages are reset and the step is taken with a nominal dt instead
# of integrating a huge one.
MAX_PLAUSIBLE_TIME_DIFFERENCE_S = 1.0
NOMINAL_TIME_DIFFERENCE_S = 0.01


class SensorPreprocessor:
    """Three-stage preprocessing pipeline: drift removal, filtering, velocity estimation.

    Composes one :class:`DriftRemovalStrategy`, one :class:`FilteringStrategy`
    and one :class:`VelocityEstimationStrategy` into a single ``filter()`` call
    that maps raw sensor readings to a typed :class:`SensorSignal`.
    """

    def __init__(self, basic_config: BasicConfig) -> None:
        """Initialize the sensor pre-processor.

        :param BasicConfig basic_config: Controller configuration; supplies both
            the strategy selection and, via ``preprocessor_config``, the
            parameter sets each strategy is built from.
        :return: None
        :raises ValueError: if the SOGI velocity path is selected without the
            SOGI filtering stage that produces the quadrature it reads.
        """
        self._basic_config: BasicConfig = basic_config
        self._config: PreprocessorConfig = basic_config.preprocessor_config

        self._use_sogi_velocity: bool = (
            basic_config.velocity_estimation_method == VelocityEstimationMethod.SOGI
        )
        if self._use_sogi_velocity and (
            basic_config.filtering_method != FilteringMethod.SOGI
        ):
            raise ValueError(
                "velocity_estimation_method=SOGI reads the quadrature produced by "
                "the angle-stage SOGI-FLL, so filtering_method must also be SOGI "
                f"(got {basic_config.filtering_method})."
            )

        # Baseline removal, ahead of every filter stage. The offset is zero
        # until a window completes, so this is a pass-through until something
        # drives set_baseline_removal_trigger() -- the main switch live, its
        # recorded value on playback.
        self._baseline_removal = BaselineRemoval(self._config.baseline_removal_config)

        self._drift_removal: DriftRemovalStrategy
        self._filtering: FilteringStrategy
        self._velocity_estimation: VelocityEstimationStrategy | None
        # Drift removal applied to the estimated velocity. Always constructed so
        # reset() and runtime switches remain consistent.
        self._velocity_drift_removal: DriftRemovalStrategy
        self._init_strategies()

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
        # measured between drift removal and the filtering stage. None on first
        # call / after reset.
        self.last_drift_removed_angle_rad: float | None = None

        # Velocity *before* the optional post-estimation drift-removal notch.
        # For the SOGI path this equals last_velocity_surrogate_rad_per_sec;
        # for other methods this is the raw output of the velocity-estimation
        # strategy. None on first call / after reset.
        self.last_velocity_pre_drift_removal_rad_per_sec: float | None = None

        # Last applied locomotion-mode class_id (0=Level, 1=Ascend, 2=Descend).
        # Tracked so set_locomotion_mode() can detect the DSC -> non-DSC
        # transition specifically and wipe the SOGI's descent-charged
        # in-phase / quadrature -- see set_locomotion_mode() docstring.
        self._current_mode_id: int = AssistMode.LEVEL

        # SOGI-FLL parameter set per assist mode. A table rather than a branch
        # chain so adding a mode is one entry. Anything not listed falls back to
        # the level config, which is what makes it safe for a caller to pass a
        # class_id this version does not know about.
        self._sogi_config_by_mode: dict[int, SogiFllConfig] = {
            AssistMode.LEVEL: self._config.filtering_sogifll_config_level,
            AssistMode.ASCEND_STAIRS: self._config.filtering_sogifll_config_ascend,
            AssistMode.DESCEND_STAIRS: self._config.filtering_sogifll_config_descend,
            AssistMode.RAMP_2_5: self._config.filtering_sogifll_config_ramp_2_5,
            AssistMode.RAMP_5: self._config.filtering_sogifll_config_ramp_5,
        }

    def _init_strategies(self) -> None:
        """Construct one strategy per stage from the configured methods.

        :return: None
        :raises ValueError: if a configured method has no implementation.
        """
        self._drift_removal = self._build_drift_removal()
        self._filtering = self._build_filtering()
        self._velocity_estimation = self._build_velocity_estimation()
        self._velocity_drift_removal = NotchDriftRemoval(
            self._config.velocity_drift_removal_notch_config
        )

    def _build_drift_removal(self) -> DriftRemovalStrategy:
        """Build the drift-removal stage.

        :return: Configured drift-removal strategy.
        :rtype: DriftRemovalStrategy
        :raises ValueError: if the configured method is unknown.
        """
        method = self._basic_config.drift_removal_method
        if method == DriftRemovalMethod.LOW_PASS:
            return LowPassDriftRemoval(
                self._config.drift_removal_second_order_lpf_config
            )
        if method == DriftRemovalMethod.NOTCH:
            return NotchDriftRemoval(self._config.drift_removal_notch_config)
        raise ValueError(f"Unrecognized drift-removal method: {method}")

    def _build_filtering(self) -> FilteringStrategy:
        """Build the angle-filtering stage.

        :return: Configured filtering strategy.
        :rtype: FilteringStrategy
        :raises ValueError: if the configured method is unknown.
        """
        method = self._basic_config.filtering_method
        if method == FilteringMethod.SOGI:
            return SogiFllFiltering(config=self._config.filtering_sogifll_config)
        if method == FilteringMethod.LOW_PASS:
            return LowPassFiltering(self._config.filtering_second_order_lpf_config)
        if method == FilteringMethod.KALMAN:
            return KalmanFiltering(self._config.filtering_kalman_config)
        raise ValueError(f"Unrecognized filtering method: {method}")

    def _build_velocity_estimation(self) -> VelocityEstimationStrategy | None:
        """Build the velocity-estimation stage.

        :return: Configured strategy, or ``None`` on the SOGI path, which reads
            the quadrature from the angle stage instead of running an estimator.
        :rtype: VelocityEstimationStrategy | None
        :raises ValueError: if the configured method is unknown.
        """
        method = self._basic_config.velocity_estimation_method
        if method == VelocityEstimationMethod.SOGI:
            return None
        if method == VelocityEstimationMethod.DISCRETE_DERIVATIVE:
            return DiscreteDerivativeVelocityEstimation()
        if method == VelocityEstimationMethod.LOW_PASS:
            return LowPassVelocityEstimation(
                self._config.filtering_second_order_lpf_config
            )
        if method == VelocityEstimationMethod.GYROSCOPE:
            return GyroscopeVelocityEstimation()
        raise ValueError(f"Unrecognized velocity-estimation method: {method}")

    def filter(self, raw_signal: SensorSignal) -> SensorSignal:
        """Run one preprocessing step and return a :class:`SensorSignal`.

        :param SensorSignal raw_signal: Current sample: timestamp [s], raw angle
            from the sensor [rad] and gyroscope angular rate [rad/s].
        :return: Preprocessed signal with the filtered angle and estimated velocity.
        :rtype: SensorSignal
        :raises ValueError: if the timestamp did not advance since the last call.
        """
        # Baseline removal first: every later stage sees an angle whose DC
        # offset is already gone, so the drift-removal LPF has far less to
        # settle out at start-up. ``raw_signal`` itself is never modified --
        # callers keep the raw sample for logging.
        angle_rad = self._baseline_removal.apply(
            angle_rad=raw_signal.angle_rad,
            velocity_rad_per_sec=raw_signal.velocity_rad_per_sec,
        )

        if self._prev_timestamp is None or raw_signal.timestamp is None:
            self._prev_timestamp = raw_signal.timestamp
            return SensorSignal(
                timestamp=raw_signal.timestamp,
                angle_rad=angle_rad,
                velocity_rad_per_sec=raw_signal.velocity_rad_per_sec,
            )

        time_difference = raw_signal.timestamp - self._prev_timestamp

        if time_difference <= 0.0:
            raise ValueError(f"Non-positive time_difference: {time_difference}")

        # A gap this large is a dropout, not a real dt. Clear the stages that
        # integrate over dt and take this step at the nominal rate instead, so a
        # huge dt cannot spike the motor reference. The angle-stage filter keeps
        # its state and its per-mode config, matching the previous behaviour.
        if time_difference > MAX_PLAUSIBLE_TIME_DIFFERENCE_S:
            self._drift_removal.reset()
            self._velocity_drift_removal.reset()
            if self._velocity_estimation is not None:
                self._velocity_estimation.reset()
            time_difference = NOMINAL_TIME_DIFFERENCE_S

        self._prev_timestamp = raw_signal.timestamp

        angle_no_drift_rad = self._drift_removal.filter(
            raw_angle=angle_rad, time_difference=time_difference
        )
        self.last_drift_removed_angle_rad = angle_no_drift_rad

        angle_out_rad = self._filtering.filter(
            angle_rad=angle_no_drift_rad, time_difference=time_difference
        )
        # Surface the SOGI quadrature for downstream logging / experimentation.
        # The SogiFllFiltering wrapper caches it on every filter() call; other
        # FilteringStrategy implementations do not expose it, hence the getattr.
        self.last_velocity_surrogate_rad_per_sec = getattr(
            self._filtering, "last_quadrature", None
        )

        velocity_pre_drift_rad_per_sec = self._estimate_velocity(
            raw_signal=raw_signal,
            baseline_removed_angle_rad=angle_rad,
            angle_no_drift_rad=angle_no_drift_rad,
            angle_out_rad=angle_out_rad,
            time_difference=time_difference,
        )
        self.last_velocity_pre_drift_removal_rad_per_sec = (
            velocity_pre_drift_rad_per_sec
        )

        # Optional post-estimation drift removal applied uniformly to every
        # velocity_estimation_method.
        if self._config.apply_velocity_drift_removal:
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

    def _estimate_velocity(
        self,
        raw_signal: SensorSignal,
        baseline_removed_angle_rad: float,
        angle_no_drift_rad: float,
        angle_out_rad: float,
        time_difference: float,
    ) -> float:
        """Estimate angular velocity via the SOGI quadrature or a configured strategy.

        :param SensorSignal raw_signal: Current raw sample.
        :param float baseline_removed_angle_rad: Raw angle after baseline removal [rad].
        :param float angle_no_drift_rad: Output of the drift-removal stage [rad].
        :param float angle_out_rad: Output of the filtering stage [rad].
        :param float time_difference: Elapsed time since the previous sample [s].
        :return: Velocity before the optional drift-removal notch [rad/s].
        :rtype: float
        """
        if self._use_sogi_velocity:
            # SOGI path: take the quadrature already produced by the angle-stage
            # SOGI-FLL. No second SOGI runs.
            self.last_velocity_lpf_angle_rad = None
            return self.last_velocity_surrogate_rad_per_sec or 0.0

        # See PreprocessorConfig.velocity_input_angle for the trade-off between
        # latency / smoothness (more filtering) and freshness (less filtering).
        if self._config.velocity_input_angle == VelocityInputAngle.RAW:
            velocity_input_angle_rad = baseline_removed_angle_rad
        elif self._config.velocity_input_angle == VelocityInputAngle.DRIFT_REMOVED:
            velocity_input_angle_rad = angle_no_drift_rad
        else:
            velocity_input_angle_rad = angle_out_rad

        # Outside the SOGI path a strategy is always configured: _use_sogi_velocity
        # is False exactly when _build_velocity_estimation returned a strategy.
        assert self._velocity_estimation is not None
        velocity_lpf_angle_rad, velocity_pre_drift_rad_per_sec = (
            self._velocity_estimation.filter(
                angle_rad=velocity_input_angle_rad,
                time_difference=time_difference,
                gyro_velocity_rad_per_sec=raw_signal.velocity_rad_per_sec,
            )
        )
        self.last_velocity_lpf_angle_rad = velocity_lpf_angle_rad
        return velocity_pre_drift_rad_per_sec

    def reset(self) -> None:
        """Reset the Signal Preprocessor if exosuit is disconnected or timeout occured.

        :return: None
        """
        self._prev_timestamp = None
        self.last_velocity_surrogate_rad_per_sec = None
        self.last_velocity_lpf_angle_rad = None
        self.last_drift_removed_angle_rad = None
        self.last_velocity_pre_drift_removal_rad_per_sec = None

        self._baseline_removal.reset()
        self._drift_removal.reset()
        self._filtering.reset()
        self._velocity_drift_removal.reset()
        if self._velocity_estimation is not None:
            self._velocity_estimation.reset()

    def set_baseline_removal_trigger(self, active: bool) -> None:
        """Drive the baseline-removal trigger.

        Wired on the exosuit to the main switch (motor enable): pass its current
        state every loop iteration, or on each edge. Re-triggering takes a fresh
        offset without touching any filter state, so it is safe mid-session
        after a strap slips.

        :param bool active: Current trigger state.
        :return: None
        """
        self._baseline_removal.set_trigger(active=active)

    @property
    def baseline_offset_rad(self) -> float:
        """Angle offset currently subtracted by the baseline removal [rad].

        Exposed for logging: recording it alongside the trigger is what lets a
        session be replayed offline with the offset it actually ran with.

        :return: The active offset; 0.0 before the first completed window.
        :rtype: float
        """
        return self._baseline_removal.offset_rad

    def set_walking_mode(self, walking: bool) -> None:
        """Tell the SOGI/FLL whether the user is actively walking.

        Called by the upstream stand-still detector in :class:`WalkOnController`.
        When walking=False, the FLL freezes its frequency tracking (states
        decay slowly toward initial guess) instead of drifting toward the
        lower clamp under noise-only input during pauses.

        :param bool walking: True while the user is actively walking.
        :return: None
        """
        if hasattr(self._filtering, "set_walking"):
            self._filtering.set_walking(walking)

    def set_locomotion_mode(self, class_id: int) -> None:
        """Swap the SOGI-FLL config to the variant tuned for this locomotion mode.

        ``class_id`` is an :class:`AssistMode` value (0=Level, 1=Ascend,
        2=Descend, 3=Ramp 2.5%, 4=Ramp 5%); 0-2 match the TCN classifier
        output. The corresponding `filtering_sogifll_config_*` field from
        :class:`PreprocessorConfig` is selected and pushed into the active
        filter. SOGI state (in-phase, quadrature, omega_est, etc.) is
        preserved; only the parameter values change, so the FLL re-adapts
        smoothly across a mode change rather than re-locking from scratch.
        Unknown class_ids fall back to the LEVEL config. A no-op when the
        filtering stage is not the SOGI-FLL.

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

        :param int class_id: :class:`AssistMode` value; unknown values apply
            the level config.
        :return: None
        """
        if not hasattr(self._filtering, "set_config"):
            return
        self._filtering.set_config(
            self._sogi_config_by_mode.get(
                class_id, self._config.filtering_sogifll_config_level
            )
        )

        left_descend = (
            self._current_mode_id == AssistMode.DESCEND_STAIRS
            and class_id != AssistMode.DESCEND_STAIRS
        )
        if left_descend:
            self._filtering.clear_state_keep_frequency()

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
        every demo re-entry. A no-op when the filtering stage is not the
        SOGI-FLL.

        :return: None
        """
        if not hasattr(self._filtering, "set_config"):
            return
        self._filtering.set_config(self._config.filtering_sogifll_config_demo)
