"""Walk On Controller for single-limb and bilateral hip flexion exosuit control."""

from hip_controller.control.gait_phase_control.gait_controller import GaitController
from hip_controller.control.motor_reference_control.amplitude_modulation import (
    AmplitudeModulation,
)
from hip_controller.control.motor_reference_control.motor_reference_controller import (
    MotionReferenceController,
)
from hip_controller.control.signal_processing.sensor_preprocessor import (
    SensorPreprocessor,
)
from hip_controller.definitions import BasicConfig, SensorSignal


class WalkOnController:
    """Walk ON Controller for a single lower limb.

    This controller implements a gait phase-based control strategy for a single limb, which can be used for both unilateral and bilateral hip flexion exosuits. The controller processes raw sensor signals to compute the current gait phase, applies amplitude modulation based on the sensor signals, and generates motor velocity commands for the exosuit's actuators.
    """

    def __init__(self, left_limb: bool, config: BasicConfig):
        """Initialize the controller.

        :param bool left_limb: True if the controller is for left lower limb, False if for right lower limb.
        :param BasicConfig config: Configurations including whether to reverse the motor command output (for mirrored wiring), whether to enable live plotting of the controller's internal states, whether to use pre-filtered sensor signals instead of raw signals and so on.

        :return: None
        """
        self.filtered = config.filtered
        if left_limb:
            self.plot = config.left_limb_plot
            self.amplitude_modulation = AmplitudeModulation(
                reverse=config.left_limb_reverse
            )
        else:
            self.plot = config.right_limb_plot
            self.amplitude_modulation = AmplitudeModulation(
                reverse=config.right_limb_reverse
            )

        if self.plot:
            from hip_controller.plotter.live_phase_portrait import PortraitWindow

            # Execute the Qt plot application.
            self.plotter = PortraitWindow(left=left_limb)
            self.plotter.show()

        self.pre_processor = SensorPreprocessor(basic_config=config)
        self.gait_controller = GaitController()

        # due to different wire settings one of them might need to be reversed - mirrored with -1

        self.motion_reference_controller = MotionReferenceController()

        self._prev_timestamp: float | None = None

        # Most recent signal passed downstream from the preprocessor (raw input
        # when filtered=True, otherwise the filtered angle + derived velocity).
        # Exposed so external code (e.g. the simulator) can log it.
        self.last_filtered_signal: SensorSignal | None = None

        # Most recent gait phase produced by the gait controller (rad). None
        # until the first step or after a reset.
        self.last_gait_phase_rad: float | None = None

    def step(self, curr_signal: SensorSignal) -> float:
        """Step the controller ahead.

        :param SensorSignal curr_signal: Current timestamp, raw hip angle in radians, raw hip angle velocity in radians per second read from sensor.

        :return: Motor velocity command in radians per second for motion reference.
        :rtype: float
        """
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

        # Compute motor command velocity
        motor_command = self.motion_reference_controller.compute_motor_command(
            gait_phase=gait_phase, amplitude=amplitude
        )

        # Safety gate: only assist during hip flexion (positive angle).
        # Negative filtered angle indicates extension / unclean signal -- in
        # both cases driving the tendon further would be wrong, so cut the
        # command to zero.
        if filtered_signal.angle_rad < 0:
            motor_command = 0.0

        # Plotting
        if self.plot and curr_signal.timestamp is not None:
            steady = self.gait_controller.get_signal_steady_state()
            self.plotter.update_plots(
                timestamp=curr_signal.timestamp,
                reference_motor=motor_command,
                steady=steady,
            )

        return motor_command

    def reset(self) -> None:
        """Reset the WalkOnController if exosuit is disconnected or timeout occured.

        :return: None
        """
        # TODO add reset functions for gait controller, motor controller and so on..
        self.pre_processor.reset()
        self.last_filtered_signal = None
        self.last_gait_phase_rad = None
        self.amplitude_modulation.last_intermediates = None
        self.motion_reference_controller.last_mapping_value = None
