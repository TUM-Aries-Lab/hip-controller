"""Sample doc string."""

from loguru import logger

from hip_controller.control.high_level import HighLevelController
from hip_controller.control.low_level import get_gait_speed, stop_condition
from hip_controller.control.mid_level import center_and_transform_gait_phase
from hip_controller.definitions import ConfigPlot, ExosuitData, SensorSignal


class ExoController:
    """Walk ON Controller for the lower limb exosuit."""

    def __init__(self):
        """Initialize the controller.

        :return: None
        """
        logger.info("Initializing controller.")
        self.left_controller = WalkOnController(
            left=True, plot=ConfigPlot.left_limb_plot
        )
        self.right_controller = WalkOnController(
            left=False, plot=ConfigPlot.right_limb_plot
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
        # TODO: a better structure for sensordata combining the dataclass sensorSignal

        self.left_controller.step(
            timestamp=sensor_data.timestamp, curr_signal=sensor_data.left
        )
        self.right_controller.step(
            timestamp=sensor_data.timestamp, curr_signal=sensor_data.right
        )
        try:
            left_gait_speed = get_gait_speed(signal=sensor_data.left)
            right_gait_speed = get_gait_speed(signal=sensor_data.right)
            if stop_condition(gait_speed=left_gait_speed) or stop_condition(
                gait_speed=right_gait_speed
            ):
                logger.info("Stop condition reached.")
            return
        except Exception as err:
            logger.error(f"{err} - Something went wrong.")


class WalkOnController:
    """Walk ON Controller for a single lower limb."""

    def __init__(self, left: bool = True, plot: bool = False):
        """Initialize the controller.

        :return: None
        """
        self.left = left
        if left:
            logger.info("Initializing the left lower limb controller.")
        else:
            logger.info("Initializing the right lower limb controller.")

        self.plot = plot
        if plot:
            from hip_controller.plotter.phase_portrait import PortraitWindow

            self.plotter = PortraitWindow(left=left)
            self.plotter.show()

            # Execute the Qt plot application.

        self.high_level_controller = HighLevelController()

    def step(self, curr_signal: SensorSignal, timestamp: float) -> None:
        """Step the controller ahead.

        :param angle: hip angle in radians.
        :param velocity: hip angle velocity in radians per second.
        :param timestamp: current timestamp.
        :return: None
        """
        logger.debug("Stepping controller ahead.")

        # High-level
        gait_phase = self.high_level_controller.update_and_compute(
            curr_signal=curr_signal, timestamp=timestamp
        )

        # Mid-level
        minus_sin_phi = center_and_transform_gait_phase(gait_phase=gait_phase)

        # Low-level

        # Plotting
        if self.plot:
            steady = self.high_level_controller.get_signal_steady_state()
            self.plotter.update_plots(
                timestamp=timestamp, sinusoidal=minus_sin_phi, steady=steady
            )
