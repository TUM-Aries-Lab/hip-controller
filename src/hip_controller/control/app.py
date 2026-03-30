"""Sample doc string."""

from loguru import logger

from hip_controller.control.assistance_control.amplitude_modulation import (
    AmplitudeModulation,
)
from hip_controller.control.assistance_control.assistance_controller import (
    MotionReferenceController,
)
from hip_controller.control.gait_phase_control.gait_controller import GaitController
from hip_controller.control.signal_processing.sensor_preprocessor import (
    SensorPreprocessor,
)
from hip_controller.definitions import (
    BasicConfig,
    ExosuitData,
    PreprocessorConfig,
    SensorSignal,
)


class ExoController:
    """Walk ON Controller for the lower limb exosuit."""

    def __init__(self):
        """Initialize the controller.

        :return: None
        """
        logger.info("Initializing the lower limb controller.")
        self.left_controller = WalkOnController(
            reverse=BasicConfig.left_limb_reverse, plot=BasicConfig.left_limb_plot
        )
        self.right_controller = WalkOnController(
            reverse=BasicConfig.right_limb_reverse, plot=BasicConfig.right_limb_plot
        )

    def step(self, sensor_data: ExosuitData):
        """Step the controller ahead.

        :param ang_left: hip angle of the left lower limb in radians.
        :param vel_left: hip angle velocity of the left lower limb in radians per second.
        :param ang_right: hip angle of the right lower limb in radians.
        :param vel_right: hip angle velocity of the right lower limb in radians per second.
        :param timestamp: current timestamp.
        :return: None

        """
        try:
            self.left_controller.step(curr_signal=sensor_data.left)
            self.right_controller.step(curr_signal=sensor_data.right)
        except Exception as err:
            logger.error(f"{err} - Something went wrong.")


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

        self.high_level_controller = GaitController()
        self.pre_processor = SensorPreprocessor(PreprocessorConfig())

        # due to different wire settings one of them might need to be reversed - mirrored with -1
        self.amplitude_modulation = AmplitudeModulation(reverse=reverse)
        self.mid_level_controller = MotionReferenceController()

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
        gait_phase = self.high_level_controller.update_and_compute(
            curr_signal=filtered_signal
        )

        # Apply amplitude modulation
        amplitude = self.amplitude_modulation.compute_amplitude(signal=curr_signal)

        # Compute motor command velocity
        motor_command = self.mid_level_controller.compute_motor_command(
            gait_phase=gait_phase, amplitude=amplitude
        )

        # Plotting
        if self.plot and curr_signal.timestamp is not None:
            steady = self.high_level_controller.get_signal_steady_state()
            self.plotter.update_plots(
                timestamp=curr_signal.timestamp,
                reference_motor=motor_command,
                steady=steady,
            )

        return motor_command
