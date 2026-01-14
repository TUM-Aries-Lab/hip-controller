"""Sample doc string."""

import time

from loguru import logger

from hip_controller.control.high_level import MotionState
from hip_controller.control.low_level import get_gait_speed, stop_condition


class WalkOnController:
    """Walk ON Controller for the lower limb exosuit."""

    def __init__(self, freq: float):
        """Initialize the controller.

        :param freq: Frequency of the controller.
        :return: None
        """
        self.freq = freq

        self.prev_angle: float = 0.0
        self.prev_velocity: float = 0.0
        self.curr_angle: float = 0.0
        self.curr_velocity: float = 0.0

        self.angle_max: float
        self.angle_min: float
        self.velocity_max: float
        self.velocity_min: float

        self.state = MotionState.INITIAL
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

    def set_state_angle_max(self) -> None:
        """Set the current state to ANGLE_MAX."""
        self.state = MotionState.ANGLE_MAX
        self.angle_max = self.curr_angle

    def set_state_angle_min(self) -> None:
        """Set the current state to ANGLE_MIN."""
        self.state = MotionState.ANGLE_MIN
        self.angle_min = self.curr_angle

    def set_state_velocity_max(self) -> None:
        """Set the current state to VELOCITY_MAX."""
        self.state = MotionState.VELOCITY_MAX
        self.velocity_max = self.curr_velocity

    def set_state_velocity_min(self) -> None:
        """Set the current state to VELOCITY_MIN."""
        self.state = MotionState.VELOCITY_MIN
        self.velocity_min = self.curr_velocity

    def set_state_initial(self) -> None:
        """Set the current state to INITIAL."""
        self.state = MotionState.INITIAL
        # reset state timer
        self.dt = 0.0

    def set_state(self, new_state: MotionState) -> None:
        """Set the current state to the given state.

        :param new_state: The new state to set.
        :return: None
        """
        self.state = new_state
        self.dt = 0.0
        if new_state == MotionState.ANGLE_MAX:
            self.angle_max = self.curr_angle
        elif new_state == MotionState.ANGLE_MIN:
            self.angle_min = self.curr_angle
        elif new_state == MotionState.VELOCITY_MAX:
            self.velocity_max = self.curr_velocity
        elif new_state == MotionState.VELOCITY_MIN:
            self.velocity_min = self.curr_velocity
