"""High-level control functions."""

from enum import Enum


class MotionState(Enum):
    """Enum for motion states."""

    INITIAL = -1
    VELOCITY_MAX = 0
    ANGLE_MAX = 1
    VELOCITY_MIN = 2
    ANGLE_MIN = 3


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


def angle_extrema_trigger(curr_velocity: float, prev_velocity: float) -> MotionState:
    """Check if angle extrema is reached based on velocity zero-crossing."""
    if hit_zero_crossing_from_upper(prev_velocity, curr_velocity):
        return MotionState.ANGLE_MAX
    if hit_zero_crossing_from_lower(prev_velocity, curr_velocity):
        return MotionState.ANGLE_MIN
    return MotionState.INITIAL


def velocity_extrema_trigger(curr_angle: float, prev_angle: float) -> MotionState:
    """Check if velocity extrema is reached based on angle zero-crossing."""
    if hit_zero_crossing_from_lower(prev_angle, curr_angle):
        return MotionState.VELOCITY_MAX
    if hit_zero_crossing_from_upper(prev_angle, curr_angle):
        return MotionState.VELOCITY_MIN
    return MotionState.INITIAL


def check_state_validity(
    self,
    prev_state: MotionState,
    curr_state: MotionState,
) -> MotionState:
    """Check if the current state is valid.

    Returns True if:
    - There's no previous state (first transition), OR
    - Time threshold has been exceeded (allow any transition), OR
    - State follows expected cyclic order
    """
    if prev_state is None or prev_state == MotionState.INITIAL:
        return curr_state

    elif (prev_state.value + 1) % 4 == curr_state.value:
        return curr_state

    else:
        return prev_state
