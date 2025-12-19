"""Low-level control functions."""

import numpy as np

from hip_controller.definitions import STOP_THRESHOLD


def stop_condition(gait_speed: float) -> bool:
    """Calculate whether the stop condition has been met.

    :param float gait_speed: gait speed
    :return: Whether the stop condition has been met.
    """
    return gait_speed < STOP_THRESHOLD


def get_gait_speed(theta: float, theta_dot: float) -> float:
    """Calculate the s gait.

    :param float theta: angle in radians.
    :param float theta_dot: angle in radians / sec.
    :returns: The gait speed.
    """
    return np.sqrt(theta**2 + theta_dot**2)
