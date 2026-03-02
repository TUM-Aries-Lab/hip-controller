"""A look-up table for motion mapping."""

from scipy.interpolate import CubicSpline

from hip_controller.definitions import MotionMapping


class CubicSplineInterpolation:
    """1-D Lookup Table for motion mapping with cubic spline interpolation and extrapolation."""

    def __init__(self):
        """Initialize the table."""
        # Create cubic spline exactly like Simulink
        self.spline = CubicSpline(
            x=MotionMapping.BREAKPOINTS, y=MotionMapping.TABLE, extrapolate=True
        )

    def step(self, u: float):
        """Evaluate lookup table for an input each time stamp value."""
        return self.spline(u)
