"""High-level control functions."""

from dataclasses import dataclass
from enum import Enum

from hip_controller.definitions import (
    MAX_STATE_CHANGE_TIME_THRESHOLD as TMAX,
    MIN_STATE_CHANGE_TIME_THRESHOLD as TMIN,
)


class MotionState(Enum):
    """Enum for motion states."""

    INITIAL = -1
    VELOCITY_MAX = 0
    ANGLE_MAX = 1
    VELOCITY_MIN = 2
    ANGLE_MIN = 3


@dataclass
class ExtremaTrigger:
    """Dataclass for motion extrema."""

    vel_max: bool
    ang_max: bool
    vel_min: bool
    ang_min: bool


def hit_zero_crossing_from_upper(prev: float, curr: float) -> bool:
    """Detect zero-crossing from upper to lower."""
    return prev >= 0 and curr < 0


def hit_zero_crossing_from_lower(prev: float, curr: float) -> bool:
    """Detect zero-crossing from lower to upper."""
    return prev <= 0 and curr > 0


def angle_max_trigger(curr_velocity: float, prev_velocity: float) -> bool:
    """Detect angle maximum based on velocity zero-crossing from positive to negative."""
    return hit_zero_crossing_from_upper(prev_velocity, curr_velocity)


def angle_min_trigger(curr_velocity: float, prev_velocity: float) -> bool:
    """Detect angle minimum based on velocity zero-crossing from negative to positive."""
    return hit_zero_crossing_from_lower(prev_velocity, curr_velocity)


def velocity_max_trigger(curr_angle: float, prev_angle: float) -> bool:
    """Detect velocity maximum based on angle zero-crossing from negative to positive."""
    return hit_zero_crossing_from_lower(prev_angle, curr_angle)


def velocity_min_trigger(curr_angle: float, prev_angle: float) -> bool:
    """Detect velocity minimum based on angle zero-crossing from positive to negative."""
    return hit_zero_crossing_from_upper(prev_angle, curr_angle)


def check_timeout(dt: float) -> bool:
    """Check if the time threshold for state change has been exceeded."""
    return dt >= TMAX


class HighLevelController:
    """High-level controller for managing motion states."""

    def __init__(self):
        """Initialize the high-level controller."""
        self.state = MotionState.INITIAL

        self.prev_angle: float = 0.0
        self.prev_velocity: float = 0.0
        self.curr_angle: float = 0.0
        self.curr_velocity: float = 0.0

        self.angle_max: float
        self.angle_min: float
        self.velocity_max: float
        self.velocity_min: float

    def extrema_trigger(
        self,
        curr_angle: float,
        prev_angle: float,
        curr_velocity: float,
        prev_velocity: float,
    ) -> ExtremaTrigger:
        """Check for extrema triggers and return the detected state.

        Returns: vel_max, ang_max, vel_min, ang_min.
        """
        vel_max = velocity_max_trigger(curr_angle, prev_angle) and curr_velocity > 0
        ang_max = angle_max_trigger(curr_velocity, prev_velocity) and curr_angle > 0
        vel_min = velocity_min_trigger(curr_angle, prev_angle) and curr_velocity < 0
        ang_min = angle_min_trigger(curr_velocity, prev_velocity) and curr_angle < 0
        return ExtremaTrigger(vel_max, ang_max, vel_min, ang_min)

    def update_state(
        self,
        current_state: MotionState,
        trigger: ExtremaTrigger,
        dt: float,
    ) -> bool:
        """Check if the given state is valid. If valid, update the current state.

        If two triggers occur simultaneously, priority is given of the next one in the order of vel_max, angle_max, vel_min, angle_min.

        A state transition is valid if:
        - Time threshold has been exceeded, OR
        - State follows expected cyclic order

            :param dt: Time passed since last state change.
        """
        state_changed = False

        # before: inclusive, after: exclusive
        if dt >= TMAX:
            self.state = MotionState.INITIAL
            return True

        elif dt < TMIN:
            return False

        # State machine transitions
        if current_state == MotionState.INITIAL:
            return self.handle_initial_state(
                trigger=trigger,
            )

        if current_state == MotionState.ANGLE_MAX and trigger.ang_max:
            self.state = MotionState.VELOCITY_MIN
            state_changed = True

        elif current_state == MotionState.ANGLE_MIN and trigger.ang_min:
            self.state = MotionState.VELOCITY_MAX
            state_changed = True

        elif current_state == MotionState.VELOCITY_MAX and trigger.vel_max:
            self.state = MotionState.ANGLE_MAX
            state_changed = True

        elif current_state == MotionState.VELOCITY_MIN and trigger.vel_min:
            self.state = MotionState.ANGLE_MIN
            state_changed = True
        return state_changed

    def handle_initial_state(
        self,
        trigger: ExtremaTrigger,
    ) -> bool:
        """Help handle INITIAL state transitions since the update_state function is too complex."""
        if trigger.vel_max:
            self.state = MotionState.ANGLE_MAX
            return True
        elif trigger.ang_max:
            self.state = MotionState.VELOCITY_MIN
            return True
        elif trigger.vel_min:
            self.state = MotionState.ANGLE_MIN
            return True
        elif trigger.ang_min:
            self.state = MotionState.VELOCITY_MAX
            return True

        return False

    def set_state(self, new_state: MotionState) -> None:
        """Set the current state to the given state.

        :param new_state: The new state to set.
        :return: None
        """
        # reset state timer
        self.dt = 0.0
        if new_state == MotionState.INITIAL:
            return
        elif new_state == MotionState.ANGLE_MAX:
            self.angle_max = self.curr_angle
        elif new_state == MotionState.ANGLE_MIN:
            self.angle_min = self.curr_angle
        elif new_state == MotionState.VELOCITY_MAX:
            self.velocity_max = self.curr_velocity
        elif new_state == MotionState.VELOCITY_MIN:
            self.velocity_min = self.curr_velocity
