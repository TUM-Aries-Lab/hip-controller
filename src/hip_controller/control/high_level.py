"""High-level control functions."""

from dataclasses import dataclass
from enum import Enum

from hip_controller.definitions import StateChangeTimeThreshold


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


def hit_zero_crossing_from_upper(curr: float, prev: float) -> bool:
    """Detect zero-crossing from upper to lower.

    Checks if a value transitions from non-negative to negative.

    :param curr: Current value.
    :param prev: Previous value.
    :return: True if zero-crossing from upper to lower detected, False otherwise.
    """
    return prev >= 0 and curr < 0


def hit_zero_crossing_from_lower(curr: float, prev: float) -> bool:
    """Detect zero-crossing from lower to upper.

    Checks if a value transitions from non-positive to positive.

    :param curr: Current value.
    :param prev: Previous value.
    :return: True if zero-crossing from lower to upper detected, False otherwise.
    """
    return prev <= 0 and curr > 0


def angle_max_trigger(curr_velocity: float, prev_velocity: float) -> bool:
    """Detect angle maximum based on velocity zero-crossing from positive to negative.

    :param curr_velocity: Current velocity value.
    :param prev_velocity: Previous velocity value.
    :return: True if angle maximum is detected, False otherwise.
    """
    return hit_zero_crossing_from_upper(curr=curr_velocity, prev=prev_velocity)


def angle_min_trigger(curr_velocity: float, prev_velocity: float) -> bool:
    """Detect angle minimum based on velocity zero-crossing from negative to positive.

    :param curr_velocity: Current velocity value.
    :param prev_velocity: Previous velocity value.
    :return: True if angle minimum is detected, False otherwise.
    """
    return hit_zero_crossing_from_lower(curr=curr_velocity, prev=prev_velocity)


def velocity_max_trigger(curr_angle: float, prev_angle: float) -> bool:
    """Detect velocity maximum based on angle zero-crossing from negative to positive.

    :param curr_angle: Current angle value.
    :param prev_angle: Previous angle value.
    :return: True if velocity maximum is detected, False otherwise.
    """
    return hit_zero_crossing_from_lower(curr=curr_angle, prev=prev_angle)


def velocity_min_trigger(curr_angle: float, prev_angle: float) -> bool:
    """Detect velocity minimum based on angle zero-crossing from positive to negative.

    :param curr_angle: Current angle value.
    :param prev_angle: Previous angle value.
    :return: True if velocity minimum is detected, False otherwise.
    """
    return hit_zero_crossing_from_upper(curr=curr_angle, prev=prev_angle)


def extrema_trigger(
    curr_angle: float,
    prev_angle: float,
    curr_velocity: float,
    prev_velocity: float,
) -> ExtremaTrigger:
    """Check validity for extrema triggers and return valid triggers.

    Validates all four extrema triggers (velocity max/min, angle max/min) based on
    zero-crossings and sign conditions.

    :param curr_angle: Current angle value.
    :param prev_angle: Previous angle value.
    :param curr_velocity: Current velocity value.
    :param prev_velocity: Previous velocity value.
    :return: ExtremaTrigger dataclass containing four boolean flags (vel_max, ang_max, vel_min, ang_min).
    """
    vel_max = velocity_max_trigger(curr_angle, prev_angle) and curr_velocity > 0
    ang_max = angle_max_trigger(curr_velocity, prev_velocity) and curr_angle > 0
    vel_min = velocity_min_trigger(curr_angle, prev_angle) and curr_velocity < 0
    ang_min = angle_min_trigger(curr_velocity, prev_velocity) and curr_angle < 0
    return ExtremaTrigger(vel_max, ang_max, vel_min, ang_min)


def _handle_initial_state(
    trigger: ExtremaTrigger,
) -> MotionState:
    """Handle INITIAL state transitions.

    Determines the first state to transition to from INITIAL based on which extrema
    trigger is detected, with priority: vel_max > ang_max > vel_min > ang_min.

    :param trigger: ExtremaTrigger dataclass containing active triggers.
    :return: New MotionState to transition to, or INITIAL if no valid trigger.
    """
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
    """Determine the next motion state based on current state and active triggers.

    Implements the state machine logic for cyclic transitions:
    INITIAL → VELOCITY_MAX → ANGLE_MAX → VELOCITY_MIN → ANGLE_MIN → (back to VELOCITY_MAX)

    If two triggers occur simultaneously, priority is given in order:
    vel_max > angle_max > vel_min > angle_min.

    :param current_state: The current motion state.
    :param trigger: ExtremaTrigger dataclass containing active triggers.
    :return: Next MotionState to transition to, or current state if no valid transition.
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
        """Initialize the high-level controller.

        Sets up initial state as INITIAL and initializes tracking variables for
        angle, velocity, extrema values, and timing information.
        """
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

    def _set_state(self, state: MotionState, timestamp: float | None) -> None:
        """Set the current state and record extrema values when applicable.

        Updates the internal state and stores the current extrema value (angle or velocity)
        based on the new state. Records the timestamp of state transition.

        :param state: The new MotionState to set.
        :param timestamp: Timestamp of the state transition, or None to clear timing.
        :return: None
        """
        self.state = state
        self.tick = timestamp

        if state == MotionState.INITIAL:
            return

        elif state == MotionState.ANGLE_MAX:
            self.angle_max = self.curr_angle

        elif state == MotionState.ANGLE_MIN:
            self.angle_min = self.curr_angle

        elif state == MotionState.VELOCITY_MAX:
            self.velocity_max = self.curr_velocity

        elif state == MotionState.VELOCITY_MIN:
            self.velocity_min = self.curr_velocity

    def update(self, curr_angle, curr_vel, timestamp) -> None:
        """Update controller state based on current angle, velocity, and timestamp.

        Checks for timeout conditions and detects state transitions based on extrema triggers.
        Updates internal tracking variables for next iteration.

        :param curr_angle: Current angle measurement.
        :param curr_vel: Current velocity measurement.
        :param timestamp: Current timestamp.
        :return: None
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
                self._set_state(state=new_state, timestamp=timestamp)

    def _check_timeout(self, timestamp: float) -> bool:
        """Check if a timeout has occurred and reset state if necessary.

        Returns True if in timeout period (update should be skipped). Resets to INITIAL
        state if timeout threshold is exceeded.

        :param timestamp: Current timestamp.
        :return: True if currently in timeout period, False otherwise.
        """
        if self.state == MotionState.INITIAL:
            return False

        if self.tick is None:
            return False

        dt = timestamp - self.tick

        # before: inclusive, after: exclusive
        if dt < StateChangeTimeThreshold.TMIN:
            return True

        elif dt >= StateChangeTimeThreshold.TMAX:
            self._set_state(MotionState.INITIAL, timestamp=None)
            return True

        return False
