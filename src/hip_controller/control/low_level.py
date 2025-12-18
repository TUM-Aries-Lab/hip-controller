"""Low-level control functions."""

import numpy as np

from hip_controller.definitions import STOP_THRESHOLD


def stop_condition(theta: float, theta_dot: float) -> bool:
    """Calculate whether the stop condition has been met.

    :param float theta: Angle in radians.
    :param float theta_dot: Angle in radians / sec.
    :return: Whether the stop condition has been met.
    """
    gait_speed = get_gait_speed(theta=theta, theta_dot=theta_dot)

    return gait_speed < STOP_THRESHOLD


def get_gait_speed(theta: float, theta_dot: float) -> float:
    """Calculate the s gait.

    :param float theta: angle in radians.
    :param float theta_dot: angle in radians / sec.
    :returns: The gait speed.
    """
    return np.sqrt(theta**2 + theta_dot**2)
