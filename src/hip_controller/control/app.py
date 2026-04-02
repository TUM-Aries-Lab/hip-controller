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
from hip_controller.definitions import PreprocessorConfig, SensorSignal


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
            self.plotter = PortraitWindow(left=reverse)
            self.plotter.show()

        self.pre_processor = SensorPreprocessor(PreprocessorConfig())
        self.gait_controller = GaitController()

        # due to different wire settings one of them might need to be reversed - mirrored with -1
        self.amplitude_modulation = AmplitudeModulation(reverse=reverse)
        self.motion_reference_controller = MotionReferenceController()

        self._prev_timestamp: float | None = None

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

        # Gait phase calculation
        gait_phase = self.gait_controller.update_and_compute(
            curr_signal=filtered_signal
        )

        # Apply amplitude modulation
        amplitude = self.amplitude_modulation.compute_amplitude(signal=filtered_signal)

        # Compute motor command velocity
        motor_command = self.motion_reference_controller.compute_motor_command(
            gait_phase=gait_phase, amplitude=amplitude
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

    def reset(self) -> None:
        """Reset the WalkOnController if exosuit is disconnected or timeout occured.

        :return: None
        """
        # TODO add reset functions for gait controller, motor controller and so on..
        self.pre_processor.reset()
