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


def extrema_trigger(
    curr_angle: float,
    prev_angle: float,
    curr_velocity: float,
    prev_velocity: float,
) -> ExtremaTrigger:
    """Check validity for extrema triggers and return valid triggers.

    Returns: vel_max, ang_max, vel_min, ang_min.
    """
    vel_max = velocity_max_trigger(curr_angle, prev_angle) and curr_velocity > 0
    ang_max = angle_max_trigger(curr_velocity, prev_velocity) and curr_angle > 0
    vel_min = velocity_min_trigger(curr_angle, prev_angle) and curr_velocity < 0
    ang_min = angle_min_trigger(curr_velocity, prev_velocity) and curr_angle < 0
    return ExtremaTrigger(vel_max, ang_max, vel_min, ang_min)


def _handle_initial_state(
    trigger: ExtremaTrigger,
) -> MotionState:
    """Handle INITIAL state transitions."""
    if trigger.vel_max:
        return MotionState.VELOCITY_MAX
    elif trigger.ang_max:
        return MotionState.ANGLE_MAX
    elif trigger.vel_min:
        return MotionState.VELOCITY_MIN
    elif trigger.ang_min:
        return MotionState.ANGLE_MIN
    return MotionState.INITIAL


def detect_state(
    current_state: MotionState,
    trigger: ExtremaTrigger,
) -> MotionState:
    """Check if the given state is valid.

    If two triggers occur simultaneously, priority is given of the next one in the order of vel_max, angle_max, vel_min, angle_min.

    A state transition is valid if:
    - Time threshold has been exceeded, OR
    - State follows expected cyclic order

    :return: state to be updated
    """
    new_state = current_state
    # State machine transitions
    if current_state == MotionState.INITIAL:
        return _handle_initial_state(
            trigger=trigger,
        )

    elif current_state == MotionState.ANGLE_MAX and trigger.vel_min:
        new_state = MotionState.VELOCITY_MIN

    elif current_state == MotionState.ANGLE_MIN and trigger.vel_max:
        new_state = MotionState.VELOCITY_MAX

    elif current_state == MotionState.VELOCITY_MAX and trigger.ang_max:
        new_state = MotionState.ANGLE_MAX

    elif current_state == MotionState.VELOCITY_MIN and trigger.ang_min:
        new_state = MotionState.ANGLE_MIN

    return new_state


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

        self.tick: float | None

    def _enter_state(self, state: MotionState, timestamp: float | None) -> None:
        """Set the current state to the given state.

        :param new_state: The new state to set.
        :return: None
        """
        self.state = state
        self.tick = timestamp

        if state == MotionState.INITIAL:
            return

        if state == MotionState.ANGLE_MAX:
            self.angle_max = self.curr_angle

        elif state == MotionState.ANGLE_MIN:
            self.angle_min = self.curr_angle

        elif state == MotionState.VELOCITY_MAX:
            self.velocity_max = self.curr_velocity

        elif state == MotionState.VELOCITY_MIN:
            self.velocity_min = self.curr_velocity

    def update(self, curr_angle, curr_vel, timestamp) -> None:
        """Update state regarding time.

        :param curr_angle: TODO
        :param curr_vel:
        :param timestamp:
        """
        self.prev_angle = self.curr_angle
        self.prev_velocity = self.curr_velocity
        self.curr_angle = curr_angle
        self.curr_velocity = curr_vel

        if self._check_timeout(timestamp=timestamp):
            return

        else:
            trigger = extrema_trigger(
                self.curr_angle, self.prev_angle, self.curr_velocity, self.prev_velocity
            )
            new_state = detect_state(self.state, trigger)
            if self.state != new_state:
                self._enter_state(state=new_state, timestamp=timestamp)

    def _check_timeout(self, timestamp: float) -> bool:
        """Check if a timeout has occurred."""
        if self.state == MotionState.INITIAL:
            return False

        if self.tick is None:
            return False

        dt = timestamp - self.tick

        # before: inclusive, after: exclusive
        if dt < TMIN:
            return True

        elif dt >= TMAX:
            self._enter_state(MotionState.INITIAL, timestamp=None)
            return True

        return False
