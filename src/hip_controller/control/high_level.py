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
    curr_angle: float, prev_angle: float, curr_velocity: float, prev_velocity: float
) -> ExtremaTrigger:
    """Check for extrema triggers and return the detected state.

    Returns: vel_max, ang_max, vel_min, ang_min.
    """
    vel_max = velocity_max_trigger(curr_angle, prev_angle)
    ang_max = angle_max_trigger(curr_velocity, prev_velocity)
    vel_min = velocity_min_trigger(curr_angle, prev_angle)
    ang_min = angle_min_trigger(curr_velocity, prev_velocity)
    return ExtremaTrigger(vel_max, ang_max, vel_min, ang_min)


def update_state(
    state: MotionState,
    trigger: ExtremaTrigger,
    dt: float,
    curr_angle: float,
    curr_velocity: float,
) -> MotionState:
    """Check if the given state is valid. If valid, update the current state.

    A state transition is valid if:
    - Time threshold has been exceeded, OR
    - State follows expected cyclic order

        :param dt: Time passed since last state change.
    """
    output_state = state
    if dt >= TMAX:
        return MotionState.INITIAL

    elif dt < TMIN:
        return state

    # State machine transitions
    # before: inclusive, after: exclusive
    if state == MotionState.INITIAL:
        return handle_initial_state(
            trigger=trigger,
            curr_angle=curr_angle,
            curr_velocity=curr_velocity,
        )

    if state == MotionState.ANGLE_MAX and trigger.ang_max and curr_angle > 0:
        output_state = MotionState.VELOCITY_MIN

    elif state == MotionState.ANGLE_MIN and trigger.ang_min and curr_angle < 0:
        output_state = MotionState.VELOCITY_MAX

    elif state == MotionState.VELOCITY_MAX and trigger.vel_max and curr_velocity > 0:
        output_state = MotionState.ANGLE_MAX

    elif state == MotionState.VELOCITY_MIN and trigger.vel_min and curr_velocity < 0:
        output_state = MotionState.ANGLE_MIN
    return output_state


def handle_initial_state(
    trigger: ExtremaTrigger,
    curr_angle: float,
    curr_velocity: float,
) -> MotionState:
    """Help handle INITIAL state transitions since the update_state function is too complex. If two triggers occur simultaneously, priority is given of the next one in the order of vel_max, angle_max, vel_min, angle_min."""
    # TODO: check the order of these transitions and handle simultaneous triggers

    if trigger.ang_min and curr_angle < 0:
        return MotionState.VELOCITY_MAX

    if trigger.vel_max and curr_velocity > 0:
        return MotionState.ANGLE_MAX
    if trigger.ang_max and curr_angle > 0:
        return MotionState.VELOCITY_MIN
    if trigger.vel_min and curr_velocity < 0:
        return MotionState.ANGLE_MIN

    return MotionState.INITIAL
