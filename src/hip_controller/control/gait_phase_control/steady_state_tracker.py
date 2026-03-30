"""Tracker and calculator for steady-state gait phase parameters."""

from hip_controller.control.gait_phase_control.motion_state_machine import (
    MotionState,
)
from hip_controller.definitions import SensorSignal
from hip_controller.utils.math_utils import align, calculate_center_value


class SteadyStateTracker:
    """Maintain extrema values (angle and velocity maxima and minima) and computes centered and normalized steady-state values for both angle and velocity."""

    def __init__(self) -> None:
        """Set up initial tracking variables for extrema, steady-state parameters and steady-states.

        :return: None
        """
        # extrema values, updated when new motion extrema state is detected
        self._angle_max: float = 0.0
        self._angle_min: float = 0.0
        self._velocity_max: float = 0.0
        self._velocity_min: float = 0.0

        # center velocity, center angle and rescale factor, updated when valid stride event is detected
        self._center_vel = 0.0
        self._center_ang = 0.0
        self._scale_factor: float = 0.0

        # normalized and centered values, updated each step
        self.vel_steady_state: float = 0.0
        self.ang_steady_state: float = 0.0

    def _calculate_rescale_factor(self) -> float:
        """Calculate the velocity-to-angle scaling factor gamma.

        :return: Ratio of velocity range to angle range for normalizing the
        steady-state of angle. This factor scales angle measurements to match
        velocity magnitude for proper phase plane representation.
        :rtype: float
        """
        angle_range = abs(self._angle_max - self._angle_min)

        # Avoid division by zero
        if angle_range <= 0.0:
            return 1.0

        velocity_range = abs(self._velocity_max - self._velocity_min)

        return velocity_range / angle_range

    def recenter(self) -> None:
        """Recompute and update the rescale factor, center values of angle and velocity when a valid stride event is detected.

        :return: None
        """
        self._center_ang = calculate_center_value(
            val_max=self._angle_max, val_min=self._angle_min
        )
        self._center_vel = calculate_center_value(
            val_max=self._velocity_max, val_min=self._velocity_min
        )
        self._scale_factor = self._calculate_rescale_factor()

    def update_steady_state(self, curr_signal: SensorSignal) -> None:
        """Update steady-state of current signal by aligning them with center values.

        :param SensorSignal curr_signal: Current sensor signal containing angle and velocity.
        :return: None
        """
        self.vel_steady_state = align(
            curr_val=curr_signal.velocity_rad_per_sec, center_val=self._center_vel
        )
        self.ang_steady_state = -(
            align(curr_val=curr_signal.angle_rad, center_val=self._center_ang)
            * self._scale_factor
        )

    def update_extrema(self, state: MotionState, curr_signal: SensorSignal) -> None:
        """Keep track of extrema values (angle_max, angle_min, velocity_max, velocity_min) when new motion state is detected.

        :param MotionState state: The newly detected motion state indicating which extremum was reached.
        :param SensorSignal curr_signal: Current sensor signal containing angle and velocity to be recorded.
        :return: None
        """
        if state == MotionState.ANGLE_MAX:
            self._angle_max = curr_signal.angle_rad

        elif state == MotionState.ANGLE_MIN:
            self._angle_min = curr_signal.angle_rad

        elif state == MotionState.VELOCITY_MAX:
            self._velocity_max = curr_signal.velocity_rad_per_sec

        elif state == MotionState.VELOCITY_MIN:
            self._velocity_min = curr_signal.velocity_rad_per_sec

        else:
            return
