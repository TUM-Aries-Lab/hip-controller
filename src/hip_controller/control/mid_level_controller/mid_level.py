"""Mid-level control functions."""

from loguru import logger
from scipy.interpolate import CubicSpline

from hip_controller.definitions import LAG_COMPENSATION, LookUpTable, PositionLimitation
from hip_controller.utils.math_utils import transform_to_cyclic


class MidLevelController:
    """Mid-level controller."""

    def __init__(self) -> None:
        """Initialize the mid-level controller."""
        # Initialize the mid-level controller with a 1-D Lookup Table for motion mapping.
        self.motion_mapping = MotionMapping()

    def compute_motor_command(self, gait_phase: float, amplitude: float) -> float:
        """Compute the motor command based on the gait phase and amplitude.

        :param gait_phase: current gait phase in radians.
        :param amplitude: current amplitude modulation factor.
        :return: Reference motion command for the motor.
        :rtype: float
        """
        # Applies a phase offset and sinusoidal transformation to the computed gait phase
        sinusoidal_behavior_gait_phase = transform_to_cyclic(
            gait_phase + LAG_COMPENSATION
        )

        mapping_value = self.motion_mapping.spline(value=sinusoidal_behavior_gait_phase)

        motor_command = mapping_value * amplitude

        # Saturation
        if motor_command < PositionLimitation.LOWER:
            logger.warning("Motor velocity command reached the lower limitation.")
            return PositionLimitation.LOWER
        elif motor_command > PositionLimitation.UPPER:
            logger.warning("Motor velocity command reached the upper limitation.")
            return PositionLimitation.UPPER
        else:
            return motor_command


class MotionMapping:
    """1-D Lookup Table for motion mapping with cubic spline interpolation and extrapolation."""

    """Evaluate lookup table for an input each time stamp value."""

    def __init__(self):
        """Initialize the table."""
        self._cubic_spline = CubicSpline(
            x=LookUpTable.BREAKPOINTS, y=LookUpTable.TABLEDATA, extrapolate=True
        )

    def spline(self, value: float):
        """Evaluate lookup table for an input each time stamp value."""
        return self._cubic_spline(value)
