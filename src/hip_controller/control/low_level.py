"""Low-level control functions."""

import numpy as np

from hip_controller.definitions import STOP_THRESHOLD, SensorSignal


def stop_condition(gait_speed: float) -> bool:
    """Calculate whether the stop condition has been met.

    :param float gait_speed: gait speed
    :return: Whether the stop condition has been met.
    """
    return gait_speed < STOP_THRESHOLD


def get_gait_speed(signal: SensorSignal) -> float:
    """Calculate the s gait.

    :param SensorSignal signal: angle in radians and velocity in radius per second.
    :returns: The gait speed.
    """
    return np.sqrt(signal.angle_rad**2 + signal.velocity_rad_per_sec**2)
