"""Sample doc string."""

import time

from loguru import logger

from hip_controller.control.high_level import (
    MotionState,
    angle_max_trigger,
    angle_min_trigger,
    velocity_max_trigger,
    velocity_min_trigger,
)
from hip_controller.control.low_level import get_gait_speed, stop_condition
from hip_controller.definitions import (
    MAX_STATE_CHANGE_TIME_THRESHOLD as TMAX,
    MIN_STATE_CHANGE_TIME_THRESHOLD as TMIN,
)


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
        self.update_state()

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

    def update_state(self):
        """Check if the given state is valid. If valid, update the current state.

        A state transition is valid if:
        - Time threshold has been exceeded, OR
        - State follows expected cyclic order

            :param dt: Time passed since last state change.
        """
        ang_max = angle_max_trigger(self.curr_velocity, self.prev_velocity)
        ang_min = angle_min_trigger(self.curr_velocity, self.prev_velocity)
        vel_max = velocity_max_trigger(self.curr_angle, self.prev_angle)
        vel_min = velocity_min_trigger(self.curr_angle, self.prev_angle)

        if self.dt >= TMAX:
            self.set_state_initial()
            return

        if self.dt < TMIN:
            return

        # State machine transitions
        # before: inclusive, after: exclusive
        if self.state == MotionState.INITIAL:
            self.handle_initial_state(
                vel_max=vel_max, vel_min=vel_min, ang_min=ang_min, ang_max=ang_max
            )

        elif self.state == MotionState.ANGLE_MAX and ang_max and self.curr_angle > 0:
            self.set_state_velocity_min()

        elif self.state == MotionState.ANGLE_MIN and ang_min and self.curr_angle < 0:
            self.set_state_velocity_max()

        elif (
            self.state == MotionState.VELOCITY_MAX
            and vel_max
            and self.curr_velocity > 0
        ):
            self.set_state_angle_max()

        elif (
            self.state == MotionState.VELOCITY_MIN
            and vel_min
            and self.curr_velocity < 0
        ):
            self.set_state_angle_min()

    def handle_initial_state(
        self, vel_max: bool, vel_min: bool, ang_min: bool, ang_max: bool
    ) -> None:
        """Help handle INITIAL state transitions since the update_state function is too complex."""
        if vel_max and self.curr_velocity > 0:
            self.set_state_angle_max()
            return
        if vel_min and self.curr_velocity < 0:
            self.set_state_angle_min()
            return
        if ang_min and self.curr_angle < 0:
            self.set_state_velocity_max()
            return
        if ang_max and self.curr_angle > 0:
            self.set_state_velocity_min()
            return

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
