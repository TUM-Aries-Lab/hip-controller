"""Sample doc string."""

from loguru import logger

from hip_controller.control.high_level_controller.high_level import HighLevelController
from hip_controller.control.mid_level_controller.amplitude_modulation import (
    AmplitudeModulation,
)
from hip_controller.control.mid_level_controller.mid_level import MidLevelController
from hip_controller.definitions import BasicConfig, ExosuitData, SensorSignal


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
            self.left_controller.step(
                timestamp=sensor_data.timestamp, curr_signal=sensor_data.left
            )
            self.right_controller.step(
                timestamp=sensor_data.timestamp, curr_signal=sensor_data.right
            )
        except Exception as err:
            logger.error(f"{err} - Something went wrong.")


class WalkOnController:
    """Walk ON Controller for a single lower limb."""

    def __init__(self, reverse: bool, plot: bool = False):
        """Initialize the controller.

        :return: None
        """
        self.plot = plot
        if plot:
            from hip_controller.plotter.phase_portrait import PortraitWindow

            self.plotter = PortraitWindow(left=reverse)
            self.plotter.show()
            # Execute the Qt plot application.

        self.high_level_controller = HighLevelController()

        # due to different wire settings one of them might need to be reversed - mirrored with -1
        self.amplitude_modulation = AmplitudeModulation(reverse=reverse)
        self.mid_level_controller = MidLevelController()

    def step(self, curr_signal: SensorSignal, timestamp: float) -> float:
        """Step the controller ahead.

        :param angle: hip angle in radians.
        :param velocity: hip angle velocity in radians per second.
        :param timestamp: current timestamp.
        :return: Motor command for motion reference.
        :rtype: float
        """
        # High-level
        gait_phase = self.high_level_controller.update_and_compute(
            curr_signal=curr_signal, timestamp=timestamp
        )

        # Mid-level
        amplitude = self.amplitude_modulation.compute_amplitude(signal=curr_signal)
        motor_command = self.mid_level_controller.compute_motor_command(
            gait_phase=gait_phase, amplitude=amplitude
        )

        # Low-level

        # Plotting
        if self.plot:
            steady = self.high_level_controller.get_signal_steady_state()
            self.plotter.update_plots(
                timestamp=timestamp, reference_motor=motor_command, steady=steady
            )

        return motor_command
