"""Sample doc string."""

import time

from loguru import logger

from hip_controller.control.high_level import HighLevelController
from hip_controller.control.low_level import get_gait_speed, stop_condition


class WalkOnController:
    """Walk ON Controller for the lower limb exosuit."""

    def __init__(self, freq: float):
        """Initialize the controller.

        :param freq: Frequency of the controller.
        :return: None
        """
        self.freq = freq

        self.high_level_controller = HighLevelController()
        self.dt = 0.0

    def step(self, theta: float, theta_dot: float):
        """Step the controller ahead.

        :param theta: hip angle in radians.
        :param theta_dot: hip angle velocity in radians per second.
        :return: None
        """
        logger.debug("Stepping controller ahead.")

        self.dt += 1 / self.freq

        self.curr_angle = theta
        self.curr_velocity = theta_dot

        # High-level

        # Mid-level

        # Low-level
        try:
            gait_speed = get_gait_speed(theta=theta, theta_dot=theta_dot)
            if stop_condition(gait_speed=gait_speed):
                logger.info("Stop condition reached.")
                return
        except Exception as err:
            logger.error(f"{err} - Something went wrong.")

        time.sleep(1 / self.freq)
        self.prev_angle = self.curr_angle
        self.prev_velocity = self.curr_velocity
