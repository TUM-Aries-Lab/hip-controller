"""Walk ON Controller for a single lower limb."""

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
    """Walk ON Controller for a single lower limb."""

    def __init__(self, reverse: bool, plot: bool = False, filtered=False):
        """Initialize the controller.

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

        :param SensorSignal curr_signal: Current timestamp, raw hip angle in radians, raw hip angle velocity in radians per second.

        :return: Motor velocity command for motion reference.
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
