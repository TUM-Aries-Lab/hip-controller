"""Mid-level control functions."""

import math

from hip_controller.definitions import (
    LAG_CORRECTION,
)


@staticmethod
def center_and_transform_gait_phase(gait_phase: float) -> float:
    """Center and transform the gait phase into a sinusoidal control signal.

    Applies a phase offset and sinusoidal transformation to the
    computed gait phase, producing a normalized control signal
    suitable for downstream controllers.


    :param float gait_phase: Gait phase angle in radians.

    :return: Transformed sinusoidal signal derived from the gait phase.
    :rtype: float
    """
    return -math.sin(gait_phase + LAG_CORRECTION)
