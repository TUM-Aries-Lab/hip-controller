"""Sample doc string."""

from loguru import logger

from hip_controller.control.high_level import HighLevelController
from hip_controller.control.low_level import get_gait_speed, stop_condition
from hip_controller.definitions import ConfigPlot
from hip_controller.utils.plotter import PortraitWindow


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

    def step(
        self,
        timestamp: float,
        ang_left: float,
        vel_left: float,
        ang_right: float,
        vel_right: float,
    ):
        """Step the controller ahead.

        :param ang_left: hip angle of the left lower limb in radians.
        :param vel_left: hip angle velocity of the left lower limb in radians per second.
        :param ang_right: hip angle of the right lower limb in radians.
        :param vel_right: hip angle velocity of the right lower limb in radians per second.
        :param timestamp: current timestamp.
        :return: None

        """
        self.left_controller.step(
            timestamp=timestamp, angle=ang_left, velocity=vel_left
        )
        self.right_controller.step(
            timestamp=timestamp, angle=ang_right, velocity=vel_right
        )


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
            self.plotter = PortraitWindow(left=left)
            self.plotter.show()

            # Execute the Qt plot application.

        self.high_level_controller = HighLevelController()

    def step(self, angle: float, velocity: float, timestamp: float) -> None:
        """Step the controller ahead.

        :param angle: hip angle in radians.
        :param velocity: hip angle velocity in radians per second.
        :param timestamp: current timestamp.
        :return: None
        """
        logger.debug("Stepping controller ahead.")

        # High-level
        minus_sin_phi = self.high_level_controller.compute(
            curr_angle=angle, curr_vel=velocity, timestamp=timestamp
        )

        if self.plot:
            normalized = self.high_level_controller.normalized_signal
            self.plotter.update_plots(
                timestamp=timestamp,
                sinusoidal=minus_sin_phi,
                angle=normalized.angle_rad,
                velocity=normalized.velocity_rad_per_sec,
            )

        # Mid-level

        # Low-level
        try:
            gait_speed = get_gait_speed(theta=angle, theta_dot=velocity)
            if stop_condition(gait_speed=gait_speed):
                if self.left:
                    logger.info("Stop condition reached for left leg.")
                else:
                    logger.info("Stop condition reached for right leg.")
                return
        except Exception as err:
            logger.error(f"{err} - Something went wrong.")
