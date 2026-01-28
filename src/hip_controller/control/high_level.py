"""High-level control functions."""

import math
from enum import Enum

from hip_controller.definitions import (
    VALUE_NEAR_ZERO,
    PositionLimitation,
    StateChangeTimeThreshold,
)
from hip_controller.math_utils import (
    hit_zero_crossing_from_lower,
    hit_zero_crossing_from_upper,
)


class MotionState(Enum):
    """Enum for motion states."""

    INITIAL = -1
    VELOCITY_MAX = 0
    ANGLE_MAX = 1
    VELOCITY_MIN = 2
    ANGLE_MIN = 3


class ExtremaTrigger:
    """Class for motion extrema."""

    def __init__(self):
        """Initialize four extrema triggers."""
        self.vel_max: bool = False
        self.ang_max: bool = False
        self.vel_min: bool = False
        self.ang_min: bool = False

    def _angle_max_trigger(
        self, curr_velocity: float, prev_velocity: float, curr_angle: float
    ) -> bool:
        """Detect angle maximum based on velocity zero-crossing from positive to negative.

        :param curr_velocity: Current velocity value.
        :param prev_velocity: Previous velocity value.
        :return: True if angle maximum is detected, False otherwise.
        """
        return (
            hit_zero_crossing_from_upper(curr=curr_velocity, prev=prev_velocity)
            and curr_angle > 0
        )

    def _angle_min_trigger(
        self, curr_velocity: float, prev_velocity: float, curr_angle: float
    ) -> bool:
        """Detect angle minimum based on velocity zero-crossing from negative to positive.

        :param curr_velocity: Current velocity value.
        :param prev_velocity: Previous velocity value.
        :return: True if angle minimum is detected, False otherwise.
        """
        return (
            hit_zero_crossing_from_lower(curr=curr_velocity, prev=prev_velocity)
            and curr_angle < 0
        )

    def _velocity_max_trigger(
        self, curr_angle: float, prev_angle: float, curr_velocity: float
    ) -> bool:
        """Detect velocity maximum based on angle zero-crossing from negative to positive.

        :param curr_angle: Current angle value.
        :param prev_angle: Previous angle value.
        :return: True if velocity maximum is detected, False otherwise.
        """
        return (
            hit_zero_crossing_from_lower(curr=curr_angle, prev=prev_angle)
            and curr_velocity > 0
        )

    def _velocity_min_trigger(
        self, curr_angle: float, prev_angle: float, curr_velocity: float
    ) -> bool:
        """Detect velocity minimum based on angle zero-crossing from positive to negative.

        :param curr_angle: Current angle value.
        :param prev_angle: Previous angle value.
        :return: True if velocity minimum is detected, False otherwise.
        """
        return (
            hit_zero_crossing_from_upper(curr=curr_angle, prev=prev_angle)
            and curr_velocity < 0
        )

    def update_triggers(
        self,
        curr_angle: float,
        prev_angle: float,
        curr_velocity: float,
        prev_velocity: float,
    ) -> None:
        """Check validity for extrema triggers and return valid triggers.

        Validates all four extrema triggers (velocity max/min, angle max/min) based on
        zero-crossings and sign conditions.

        :param curr_angle: Current angle value.
        :param prev_angle: Previous angle value.
        :param curr_velocity: Current velocity value.
        :param prev_velocity: Previous velocity value.
        :return: ExtremaTrigger dataclass containing four boolean flags (vel_max, ang_max, vel_min, ang_min).
        """
        self.vel_max = self._velocity_max_trigger(
            curr_angle=curr_angle, prev_angle=prev_angle, curr_velocity=curr_velocity
        )
        self.ang_max = self._angle_max_trigger(
            curr_velocity=curr_velocity,
            prev_velocity=prev_velocity,
            curr_angle=curr_angle,
        )
        self.vel_min = self._velocity_min_trigger(
            curr_angle=curr_angle, prev_angle=prev_angle, curr_velocity=curr_velocity
        )
        self.ang_min = self._angle_min_trigger(
            curr_velocity=curr_velocity,
            prev_velocity=prev_velocity,
            curr_angle=curr_angle,
        )

    def _handle_initial_state(self) -> MotionState:
        """Handle INITIAL state transitions.

        Determines the first state to transition to from INITIAL based on which extrema
        trigger is detected, with priority: vel_max > ang_max > vel_min > ang_min.

        :param trigger: ExtremaTrigger dataclass containing active triggers.
        :return: New MotionState to transition to, or INITIAL if no valid trigger.
        """
        # The order is not important
        if self.vel_max:
            return MotionState.VELOCITY_MAX
        elif self.ang_max:
            return MotionState.ANGLE_MAX
        elif self.vel_min:
            return MotionState.VELOCITY_MIN
        elif self.ang_min:
            return MotionState.ANGLE_MIN
        return MotionState.INITIAL

    def detect_state(
        self,
        current_state: MotionState,
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
            return self._handle_initial_state()

        elif current_state == MotionState.ANGLE_MAX and self.vel_min:
            new_state = MotionState.VELOCITY_MIN

        elif current_state == MotionState.ANGLE_MIN and self.vel_max:
            new_state = MotionState.VELOCITY_MAX

        elif current_state == MotionState.VELOCITY_MAX and self.ang_max:
            new_state = MotionState.ANGLE_MAX

        elif current_state == MotionState.VELOCITY_MIN and self.ang_min:
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

        self.extrema_triggers: ExtremaTrigger = ExtremaTrigger()
        self.steady_state_tracker: SteadyStateTracker = SteadyStateTracker()

        self.timestamp_sec: float | None

    def _set_state(self, state: MotionState, timestamp: float | None) -> None:
        """Set the current state and record extrema values when applicable.

        Updates the internal state and stores the current extrema value (angle or velocity)
        based on the new state. Records the timestamp of state transition.

        :param state: The new MotionState to set.
        :param timestamp: Timestamp of the state transition, or None to clear timing.
        :return: None
        """
        self.state = state
        self.timestamp_sec = timestamp

        if state == MotionState.INITIAL:
            return

        elif state == MotionState.ANGLE_MAX:
            self.steady_state_tracker.angle_max = self.curr_angle

        elif state == MotionState.ANGLE_MIN:
            self.steady_state_tracker.angle_min = self.curr_angle

        elif state == MotionState.VELOCITY_MAX:
            self.steady_state_tracker.velocity_max = self.curr_velocity

        elif state == MotionState.VELOCITY_MIN:
            self.steady_state_tracker.velocity_min = self.curr_velocity

    def _check_timeout(self, timestamp: float) -> bool:
        """Check if a timeout has occurred and reset state if necessary.

        Returns True if in timeout period (update should be skipped). Resets to INITIAL
        state if timeout threshold is exceeded.

        :param timestamp: Current timestamp.
        :return: True if currently in timeout period, False otherwise.
        """
        if self.state == MotionState.INITIAL:
            return False

        if self.timestamp_sec is None:
            return False

        dt = timestamp - self.timestamp_sec

        # before: inclusive, after: exclusive
        if dt < StateChangeTimeThreshold.TMIN:
            return True

        elif dt >= StateChangeTimeThreshold.TMAX:
            self._set_state(MotionState.INITIAL, timestamp=None)
            return True

        return False

    def update(self, curr_angle: float, curr_vel: float, timestamp: float) -> None:
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
            pass

        else:
            self.extrema_triggers.update_triggers(
                curr_angle=self.curr_angle,
                prev_angle=self.prev_angle,
                curr_velocity=self.curr_velocity,
                prev_velocity=self.prev_velocity,
            )
            new_state = self.extrema_triggers.detect_state(self.state)
            if self.state != new_state:
                self._set_state(state=new_state, timestamp=timestamp)

        self.steady_state_tracker.update_steady_state(
            curr_angle=curr_angle, curr_velocity=curr_vel
        )


class SteadyStateTracker:
    """Tracker for steady state."""

    def __init__(self):
        """Initialize the tracker for steady state.

        Sets up initial state as INITIAL and initializes tracking variables for max and min values of angle and velocity. Provide information through centering and normalization for steady state.
        """
        self.angle_max: float = 0.0
        self.angle_min: float = 0.0
        self.velocity_max: float = 0.0
        self.velocity_min: float = 0.0

        self.vel_steady_state: float = 0.0
        self.z_t: float = 0.0
        self.pos_steady_state: float = 0.0

    @staticmethod
    def calculate_steady(val_max: float, val_min: float, val_curr: float) -> float:
        """Calculate the steady-state value relative to a bounded range.

        The steady value is computed by subtracting the midpoint of the
        provided maximum and minimum bounds from the current value.

        :param val_max: Upper bound of the value range.
        :param val_min: Lower bound of the value range.
        :param val_curr: Current value.
        :return: Steady-state value relative to the range midpoint.
        """
        return val_curr - ((val_max + val_min) / 2.0)

    def _calculate_vel_ss(self, curr_velocity: float) -> float:
        """Calculate steady state of velocity.

        :return: steady state of velocity.
        """
        return self.calculate_steady(
            val_max=self.velocity_max,
            val_min=self.velocity_min,
            val_curr=curr_velocity,
        )

    def _calculate_ang_ss(self, curr_angle: float) -> float:
        """Calculate steady state of angle.

        :return: steady state of angle.
        """
        return self.calculate_steady(
            val_max=self.angle_max, val_min=self.angle_min, val_curr=curr_angle
        )

    def _calculate_z_t(self) -> float:
        """Calculate value of position steady state, set z_t, pos_ss.

        :return: value of z_t
        """
        u_vel = abs(self.velocity_max - self.velocity_min)
        u_ang = abs(self.angle_max - self.angle_min)

        # Avoid division by zero
        if u_ang == 0.0:
            u_ang = VALUE_NEAR_ZERO

        return u_vel / u_ang

    def _calculate_pos_ss(self, curr_angle: float) -> float:
        """Calculate value of position steady state, set pos_ss.

        :return: value of pos_ss
        """
        # This has to happen after z_t is set
        return self.z_t * self._calculate_ang_ss(curr_angle=curr_angle)

    def calculate_gait_phase(self) -> float:
        """Calculate gait phase.

        :return: gait phase
        """
        if self.z_t == 0.0:
            return 0.0
        else:
            return math.atan2(self.vel_steady_state, -self.pos_steady_state)

    def update_steady_state(self, curr_angle: float, curr_velocity: float) -> None:
        """Update steady state variables.

        :return: None
        """
        self.vel_steady_state = self._calculate_vel_ss(curr_velocity=curr_velocity)

        z_t = self._calculate_z_t()
        if not math.isnan(z_t):
            self.z_t = z_t

        pos_ss = self._calculate_pos_ss(curr_angle=curr_angle)
        if PositionLimitation.LOWER <= pos_ss <= PositionLimitation.UPPER:
            self.pos_steady_state = pos_ss
