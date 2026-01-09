"""High-level control functions."""

from dataclasses import dataclass
from enum import Enum

from hip_controller.definitions import MIN_TIME_THRESHOLD


@dataclass
class MotionData:
    """Dataclass to store motion sensor data."""

    timestamp: float
    angle: float
    velocity: float


class MotionState(Enum):
    """Enum for motion states."""

    VELOCITY_MAX = 0
    ANGLE_MAX = 1
    VELOCITY_MIN = 2
    ANGLE_MIN = 3


class MotionStateHandler:
    """Detect zero-crossing events, validate cyclic order, and check min and max values.

    The required cyclic order is: SPEED_MAX -> ANGLE_MAX -> SPEED_MIN -> ANGLE_MIN (circular).
    The first observed trigger can set the starting point of the cycle.
    """

    def __init__(self):
        self._previous_data: MotionData
        self._current_data: MotionData
        self._previous_state: MotionState | None = None
        self._current_state: MotionState | None = self.detect_state()
        self._last_state_change_time: float | None = None

    def update(self, new_data: MotionData) -> None:
        """Update the state handler with new motion data.

        :param MotionData new_data: The new motion data.
        """
        self._previous_data = self._current_data
        self._current_data = new_data
        if self._current_state is not None:
            self._previous_state = self._current_state

    def _has_time_threshold_exceeded(self) -> bool:
        """Check if time threshold has been exceeded since last state change."""
        if self._last_state_change_time is None or self._current_data is None:
            return False
        time_elapsed = self._current_data.timestamp - self._last_state_change_time
        return time_elapsed > MIN_TIME_THRESHOLD

    @staticmethod
    def hit_zero_crossing_from_upper(prev: float, curr: float) -> bool:
        """Detect zero-crossing from upper to lower."""
        return prev > 0 and curr <= 0

    @staticmethod
    def hit_zero_crossing_from_lower(prev: float, curr: float) -> bool:
        """Detect zero-crossing from lower to upper."""
        return prev < 0 and curr >= 0

    def detect_state(self) -> MotionState | None:
        """Detect the current motion state based on zero-crossings.

        :return: The detected motion state, or None if no state change is detected.
        """
        if self._previous_data is None or self._current_data is None:
            return None

        # Check for zero-crossings in velocity for angle states
        if self.hit_zero_crossing_from_upper(
            self._previous_data.velocity, self._current_data.velocity
        ):
            return MotionState.ANGLE_MAX

        elif self.hit_zero_crossing_from_lower(
            self._previous_data.velocity, self._current_data.velocity
        ):
            return MotionState.ANGLE_MIN

        # Check for zero-crossings in angle for velocity states
        if self.hit_zero_crossing_from_lower(
            self._previous_data.angle, self._current_data.angle
        ):
            return MotionState.VELOCITY_MAX

        elif self.hit_zero_crossing_from_upper(
            self._previous_data.angle, self._current_data.angle
        ):
            return MotionState.VELOCITY_MIN

        return None

    def check_state_validity(self, state: MotionState) -> bool:
        """Check if the current state is valid.

        Returns True if:
        - There's no previous state (first transition), OR
        - Time threshold has been exceeded (allow any transition), OR
        - State follows expected cyclic order
        """
        if self._previous_state is None:
            return True
        if self._has_time_threshold_exceeded():
            self._last_state_change_time = self._current_data.timestamp
            return True
        return (self._previous_state.value + 1) % 4 == state.value
