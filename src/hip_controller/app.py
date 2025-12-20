"""Sample doc string."""

import time

import numpy as np
from loguru import logger

from hip_controller.control.low_level import get_gait_speed, stop_condition


class WalkOnController:
    """Walk ON Controller for the lower limb exosuit."""

    def __init__(self, freq: float):
        """Initialize the controller.

        :param freq: Frequency of the controller.
        :return: None
        """
        self.freq = freq

    def step(self):
        """Step the controller ahead."""
        logger.debug("Stepping controller ahead.")

        # High-level

        # Mid-level

        # Low-level
        try:
            theta, theta_dot = self.get_sensor_data()
            gait_speed = get_gait_speed(theta=theta, theta_dot=theta_dot)
            if stop_condition(gait_speed=gait_speed):
                logger.info("Stop condition reached.")
                return
        except Exception as err:
            logger.error(f"{err} - Something went wrong.")

        time.sleep(1 / self.freq)

    @staticmethod
    def get_sensor_data() -> tuple[float, float]:
        """Get fake sensor data."""
        logger.debug("Getting fake sensor data.")
        return np.sin(time.monotonic() / 2), np.cos(time.monotonic() / 2)
