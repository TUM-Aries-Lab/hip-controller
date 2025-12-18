"""Low-level control functions."""

import numpy as np

from hip_controller.definitions import STOP_THRESHOLD


def stop_condition(theta: float, theta_dot: float) -> bool:
    """Calculate whether the stop condition has been met.

    :param float theta: Angle in radians.
    :param float theta_dot: Angle in radians / sec.
    :return: Whether the stop condition has been met.
    """
    s_gait = get_s_gait(theta=theta, theta_dot=theta_dot)

    return s_gait < STOP_THRESHOLD


def get_s_gait(theta: float, theta_dot: float) -> float:
    """Calculate the s gait.

    :param float theta: angle in radians.
    :param float theta_dot: angle in radians / sec.
    :returns: The s gait.
    """
    return np.sqrt(theta**2 + theta_dot**2)
