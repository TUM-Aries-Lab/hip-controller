"""Walk On Controller for single-limb and bilateral hip flexion exosuit control."""

from hip_controller.control.gait_phase_control.gait_controller import GaitController
from hip_controller.control.motor_reference_control.amplitude_modulation import (
    AmplitudeModulation,
    AscendStairsMode,
    DescendStairsMode,
    LevelGroundMode,
    ModeStrategy,
)
from hip_controller.control.motor_reference_control.motor_reference_controller import (
    MotionReferenceController,
)
from hip_controller.control.signal_processing.sensor_preprocessor import (
    SensorPreprocessor,
)
from hip_controller.definitions import PreprocessorConfig, SensorSignal

# Pause-detection thresholds for the SOGI/FLL walking-mode gate. The envelope
# is an exponential moving average of |raw velocity| with time constant
# 1 / PAUSE_DETECT_ENVELOPE_ALPHA samples (at 100 Hz default loop).
#
# Hysteresis (envelope rad/s):
#   below PAUSE_ENTER_THRESHOLD -> declare paused, freeze FLL
#   above PAUSE_EXIT_THRESHOLD  -> declare walking, resume FLL
#
# Defaults tuned for a healthy walking velocity peak ~2-3 rad/s. If walking
# velocity stays well below 1 rad/s in your setup, drop these accordingly.
PAUSE_DETECT_ENVELOPE_ALPHA = 0.10  # EMA weight per sample (= ~100 ms time constant)
PAUSE_ENTER_THRESHOLD = 0.2  # rad/s -- below this for sustained time = pause
PAUSE_EXIT_THRESHOLD = 0.5  # rad/s -- above this = walking again


class WalkOnController:
    """Walk ON Controller for a single lower limb.

    This controller implements a gait phase-based control strategy for a single limb, which can be used for both unilateral and bilateral hip flexion exosuits. The controller processes raw sensor signals to compute the current gait phase, applies amplitude modulation based on the sensor signals, and generates motor velocity commands for the exosuit's actuators.
    """

    def __init__(self, reverse: bool, plot: bool = False, filtered=False):
        """Initialize the controller.

        :param bool reverse: Whether to reverse the motor command output (for mirrored wiring).
        :param bool plot: Whether to enable live plotting of the controller's internal states.
        :param bool filtered: Whether to use pre-filtered sensor signals instead of raw signals.

        :return: None
        """
        self.plot = plot
        self.filtered = filtered
        if plot:
            from hip_controller.plotter.live_phase_portrait import PortraitWindow

            # Execute the Qt plot application.
            self.plotter = PortraitWindow(left=not reverse)
            self.plotter.show()

        self.pre_processor = SensorPreprocessor(PreprocessorConfig())
        self.gait_controller = GaitController()

        # due to different wire settings one of them might need to be reversed - mirrored with -1
        self.amplitude_modulation = AmplitudeModulation(reverse=reverse)
        self.motion_reference_controller = MotionReferenceController()

        # Apply the LEVEL SOGI config at construction so the SOGI starts
        # with the level-walking tuning (cadence bounds, gains, smoother
        # BWs) rather than the bare-default `filtering_sogifll_config`.
        # The amplitude modulation already defaults to LevelGroundMode in
        # its own __init__, so this aligns both halves.
        self.pre_processor.set_locomotion_mode(0)

        self._prev_timestamp: float | None = None

        # Most recent signal passed downstream from the preprocessor (raw input
        # when filtered=True, otherwise the filtered angle + derived velocity).
        # Exposed so external code (e.g. the simulator) can log it.
        self.last_filtered_signal: SensorSignal | None = None

        # Most recent gait phase produced by the gait controller (rad). None
        # until the first step or after a reset.
        self.last_gait_phase_rad: float | None = None

        # Pause-detection state. `_raw_vel_envelope` is an EMA of
        # |raw velocity| in rad/s. `_is_walking_mode` tracks the hysteretic
        # walking/paused state and is mirrored into the SOGI/FLL via
        # `pre_processor.set_walking_mode()` so the FLL frequency does not
        # drift during stand-still periods.
        self._raw_vel_envelope: float = 0.0
        self._is_walking_mode: bool = True

    def step(self, curr_signal: SensorSignal) -> float:
        """Step the controller ahead.

        :param SensorSignal curr_signal: Current timestamp, raw hip angle in radians, raw hip angle velocity in radians per second read from sensor.

        :return: Motor velocity command in radians per second for motion reference.
        :rtype: float
        """
        # ---- Stand-still detection ------------------------------------------
        # Update an EMA envelope of |raw velocity| and apply hysteresis to
        # decide whether the user is actively walking. When the user pauses,
        # tell the SOGI/FLL to freeze its frequency tracking so it does not
        # drift toward the lower clamp under noise-only input. When walking
        # resumes, re-enable adaptation -- the frequency estimate is preserved
        # from before the pause, so the SOGI is correctly tuned from sample
        # one of the next stride.
        self._raw_vel_envelope = (
            1.0 - PAUSE_DETECT_ENVELOPE_ALPHA
        ) * self._raw_vel_envelope + PAUSE_DETECT_ENVELOPE_ALPHA * abs(
            curr_signal.velocity_rad_per_sec
        )
        if self._is_walking_mode and self._raw_vel_envelope < PAUSE_ENTER_THRESHOLD:
            self._is_walking_mode = False
            self.pre_processor.set_walking_mode(False)
        elif (
            not self._is_walking_mode
        ) and self._raw_vel_envelope > PAUSE_EXIT_THRESHOLD:
            self._is_walking_mode = True
            self.pre_processor.set_walking_mode(True)

        # Pre-processing
        if self.filtered:
            filtered_signal = curr_signal
        else:
            filtered_signal = self.pre_processor.filter(raw_signal=curr_signal)
        self.last_filtered_signal = filtered_signal

        # Gait phase calculation
        gait_phase = self.gait_controller.update_and_compute(
            curr_signal=filtered_signal
        )
        self.last_gait_phase_rad = gait_phase

        # Apply amplitude modulation
        amplitude = self.amplitude_modulation.compute_amplitude(signal=filtered_signal)

        # Compute motor command velocity. The angle-sign safety gate that used
        # to zero motor_command whenever filtered angle < 0 has been removed:
        # the asymmetric LookUp table in MotionMapping already produces the
        # intended shape (strong assist during flexion, small counter-pull
        # during extension, smooth transition through zero). The gate was
        # destroying the designed extension counter-pull AND turning every
        # zero-crossing into a step input the motor could not follow. Keeping
        # flexion_active=True lets the natural shape through; LAG_COMPENSATION
        # phase advance now actually fires before flexion onset.
        motor_command = self.motion_reference_controller.compute_motor_command(
            gait_phase=gait_phase,
            amplitude=amplitude,
            timestamp=curr_signal.timestamp,
            flexion_active=True,
        )

        # Plotting
        if self.plot and curr_signal.timestamp is not None:
            steady = self.gait_controller.get_signal_steady_state()
            self.plotter.update_plots(
                timestamp=curr_signal.timestamp,
                reference_motor=motor_command,
                steady=steady,
            )

        return motor_command

    def set_locomotion_mode(self, class_id: int) -> None:
        """Apply per-mode tuning across the whole controller for one limb.

        ``class_id`` is the TCN classifier output: 0=Level, 1=Ascend,
        2=Descend. Fans out to:

        * :class:`AmplitudeModulation`: switches the per-mode amplitude
          parameters (scale, sigmoid_power, gain, velocity_weight) via the
          existing ``set_mode`` API.
        * :class:`SensorPreprocessor`: swaps the active ``SogiFllConfig`` to
          the variant tuned for this mode (cadence bounds, FLL/SOGI gains,
          smoother bandwidths, initial frequency guess). SOGI state is
          preserved so the lock continues smoothly across the boundary.

        Unknown class_ids fall back to Level Ground.
        """
        mode: ModeStrategy
        if class_id == 1:
            mode = AscendStairsMode()
        elif class_id == 2:
            mode = DescendStairsMode()
        else:
            mode = LevelGroundMode()
        self.amplitude_modulation.set_mode(mode)
        self.pre_processor.set_locomotion_mode(class_id)
        # Per-mode motion-mapping table: zeroes the extension-side
        # counter-pull on ASC and DSC so the motor doesn't pay out cable
        # between strides on stair modes (the source of cumulative slack).
        self.motion_reference_controller.set_locomotion_mode(class_id)

    def set_demo_mode(self) -> None:
        """Apply the demo-mode SOGI tuning to this limb's pre-processor.

        Delegates to :meth:`SensorPreprocessor.set_demo_mode`, which
        swaps in ``filtering_sogifll_config_demo`` for lower phase lag
        on the classification-free demo signal path. The amplitude
        modulation and motion-reference-controller modes are left
        untouched -- demo mode uses its own downstream mapping in the
        caller, not the WalkOn per-locomotion modes.
        """
        self.pre_processor.set_demo_mode()

    def reset(self) -> None:
        """Reset the WalkOnController if exosuit is disconnected or timeout occured.

        :return: None
        """
        # TODO add reset functions for gait controller, motor controller and so on..
        self.pre_processor.reset()
        self.last_filtered_signal = None
        self.last_gait_phase_rad = None
        self.amplitude_modulation.reset()
        self.motion_reference_controller.reset()
        # Clear pause-detection state and resume in walking mode so the next
        # session does not start in a frozen FLL state.
        self._raw_vel_envelope = 0.0
        self._is_walking_mode = True
        self.pre_processor.set_walking_mode(True)
