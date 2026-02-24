"""State machine for motion state transitions. Detects states in the order of Angle min → Velocity max → Angle max → Velocity min → Angle min → .. to make sure that no false small maxima or minima are identified. If at any point during the execution of the code, the expected extrema does not appear after a specified time (Tmax), the diagram is resynchronized and any incoming extrema is accepted as the new starting point. These final valid extrema triggers are then used to save the amplitude of the angle or velocity minima or maxima."""

from dataclasses import dataclass
from enum import Enum

from hip_controller.definitions import (
    SensorSignal,
    StateChangeTimeThreshold,
)
from hip_controller.utils.math_utils import (
    hit_crossing_from_lower,
    hit_crossing_from_upper,
)


class MotionState(Enum):
    """Enumeration of motion states in the gait cycle.

    Represents the four fundamental states of periodic motion as detected through
    extrema analysis of joint angle and angular velocity. The state machine cycles
    through these states in order: VELOCITY_MAX → ANGLE_MAX → VELOCITY_MIN
    → ANGLE_MIN → (back to INITIAL anytime).
    """

    INITIAL = 0
    VELOCITY_MAX = 1
    ANGLE_MAX = 2
    VELOCITY_MIN = 3
    ANGLE_MIN = 4


@dataclass
class ExtremaTrigger:
    """Boolean flags for motion extrema detection.

    Angular velocity is the first derivative of joint angle. Therefore, local maxima and minima of the angle occur at time instants where the angular velocity crosses zero with a change in sign. Angle maxima correspond to velocity zero-crossings from positive to negative, while angle minima correspond to zero-crossings from negative to positive.
    Each boolean flag indicates whether a specific extrema condition was met:

    Attributes
    ----------
    :vel_max: Velocity reaches maximum (zero-crossing from negative to positive in angle)
    :ang_max: Angle reaches maximum (zero-crossing from positive to negative in velocity)
    :vel_min: Velocity reaches minimum (zero-crossing from positive to negative in angle)
    :ang_min: Angle reaches minimum (zero-crossing from negative to positive in velocity)

    """

    vel_max: bool
    ang_max: bool
    vel_min: bool
    ang_min: bool

    def _angle_max_trigger(
        self, curr_velocity: float, prev_velocity: float, curr_angle: float
    ) -> bool:
        """Detect angle maximum based on velocity zero-crossing from positive to negative.

        :param curr_velocity: Current velocity value.
        :param prev_velocity: Previous velocity value.
        :return: True if angle maximum is detected, False otherwise.
        """
        return (
            hit_crossing_from_upper(curr=curr_velocity, prev=prev_velocity)
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
            hit_crossing_from_lower(curr=curr_velocity, prev=prev_velocity)
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
            hit_crossing_from_lower(curr=curr_angle, prev=prev_angle)
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
            hit_crossing_from_upper(curr=curr_angle, prev=prev_angle)
            and curr_velocity < 0
        )

    def set_triggers(self, curr: SensorSignal, prev: SensorSignal) -> None:
        """Evaluate and set all extrema triggers based on sensor signal transitions.

        Validates all four extrema triggers (velocity max/min, angle max/min) by
        evaluating zero-crossings and sign conditions on the current and previous
        sensor measurements. Updates the instance variables for each trigger flag.

        :param SensorSignal curr:
            Current sensor signal containing angle and velocity measurements.
        :param SensorSignal prev:
            Previous sensor signal containing angle and velocity measurements.
        :return:
            None. Updates instance variables: self.vel_max, self.ang_max, self.vel_min,
            self.ang_min to reflect the detected extrema.
        :rtype: None
        """
        self.vel_max = self._velocity_max_trigger(
            curr_angle=curr.angle_rad,
            prev_angle=prev.angle_rad,
            curr_velocity=curr.velocity_rad_per_sec,
        )
        self.ang_max = self._angle_max_trigger(
            curr_velocity=curr.velocity_rad_per_sec,
            prev_velocity=prev.velocity_rad_per_sec,
            curr_angle=curr.angle_rad,
        )
        self.vel_min = self._velocity_min_trigger(
            curr_angle=curr.angle_rad,
            prev_angle=prev.angle_rad,
            curr_velocity=curr.velocity_rad_per_sec,
        )
        self.ang_min = self._angle_min_trigger(
            curr_velocity=curr.velocity_rad_per_sec,
            prev_velocity=prev.velocity_rad_per_sec,
            curr_angle=curr.angle_rad,
        )


class MotionStateMachine:
    """Finite state machine for motion state transitions.

    Manages the state machine that cycles through motion states (VELOCITY_MAX
    → ANGLE_MAX → VELOCITY_MIN → ANGLE_MIN → back to VELOCITY_MAX) based on detected
    extrema triggers. Includes timeout detection and priority-based state resolution
    when multiple triggers occur simultaneously. The machine enforces minimum and maximum
    state dwell times to ensure physical validity of state transitions.
    """

    def __init__(self) -> None:
        """Initialize the motion state machine.

        Sets up the initial state as INITIAL, clears the timestamp, and initializes
        all extrema trigger flags to False. The machine is ready to receive sensor
        data and detect state transitions.

        Attributes
        ----------
        state : MotionState
            Current motion state.

        timestamp_sec : float or None
            Timestamp (in seconds) when the current non-initial state was entered.

            * ``float`` — A valid timestamp is stored when the state is not ``MotionState.INITIAL``.
            * ``None`` — No timestamp is tracked when the state is ``MotionState.INITIAL``.

        triggers : ExtremaTrigger
            Stores the results of extrema trigger detection for a single control cycle.

        :return: None

        """
        self.state: MotionState = MotionState.INITIAL
        self.timestamp_sec: float | None = None
        self.triggers: ExtremaTrigger = ExtremaTrigger(False, False, False, False)

    def _handle_initial_state(self) -> MotionState | None:
        """Determine next state from INITIAL based on active triggers.

        When in the INITIAL state, any active extrema trigger can initiate a transition.
        The order of evaluation is not enforced in INITIAL; the first active trigger
        encountered determines the next state. Typically, the first motion detection
        (vel_max, ang_max, vel_min, or ang_min) will drive the first transition.

        :return:
            Next MotionState to transition to (VELOCITY_MAX, ANGLE_MAX, VELOCITY_MIN,
            or ANGLE_MIN), or None if no valid trigger is active.
        :rtype: MotionState | None
        """
        # The order is not important
        if self.triggers.vel_max:
            return MotionState.VELOCITY_MAX
        elif self.triggers.ang_max:
            return MotionState.ANGLE_MAX
        elif self.triggers.vel_min:
            return MotionState.VELOCITY_MIN
        elif self.triggers.ang_min:
            return MotionState.ANGLE_MIN
        return None

    def _detect_state(self) -> MotionState | None:
        """Determine next state transition based on current state and active triggers.

        Implements the cyclic state machine logic where valid transitions depend on
        the current state and the active triggers. The machine enforces the following
        cycle:VELOCITY_MAX → ANGLE_MAX → VELOCITY_MIN → ANGLE_MIN → (back
        to VELOCITY_MAX). If multiple triggers occur simultaneously, priority order
        applies always on the next state in cycle.

        :return:
            Next MotionState to transition to based on current state and triggers,
            or None if no valid transition is possible.
        :rtype: MotionState | None
        """
        new_state = None
        # State machine transitions
        if self.state == MotionState.INITIAL:
            return self._handle_initial_state()

        elif self.state == MotionState.ANGLE_MAX and self.triggers.vel_min:
            new_state = MotionState.VELOCITY_MIN

        elif self.state == MotionState.ANGLE_MIN and self.triggers.vel_max:
            new_state = MotionState.VELOCITY_MAX

        elif self.state == MotionState.VELOCITY_MAX and self.triggers.ang_max:
            new_state = MotionState.ANGLE_MAX

        elif self.state == MotionState.VELOCITY_MIN and self.triggers.ang_min:
            new_state = MotionState.ANGLE_MIN

        return new_state

    def _is_timeout(self, timestamp: float) -> bool:
        """Detect timeout condition and reset state if necessary.

        Checks if the system is in a timeout period (where updates are skipped) based
        on the state change threshold timings. Returns True if in timeout, False if an
        update should proceed. If the maximum allowed state dwell time (TMAX) is exceeded,
        the state machine is reset to INITIAL.

        :param float timestamp:
            Current timestamp in seconds.
        :return:
            True if currently in timeout period and update should be skipped, False if
            state transition check should proceed. Reset state to INITIAL and timestamp_sec to None
            if TMAX is exceeded.
        :rtype: bool
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
            self.state = MotionState.INITIAL
            self.timestamp_sec = None
            return True

        return False

    def update_motion_state(
        self, prev: SensorSignal, curr: SensorSignal, timestamp: float
    ) -> MotionState | None:
        """Update the motion state machine based on sensor signals and timing.

        Evaluates timeout conditions, processes sensor signal transitions through
        extrema trigger detection, and attempts a state transition. Records the
        timestamp of any new state transition for timeout tracking.

        :param SensorSignal prev:
            Previous sensor signal containing prior angle and velocity measurements.
        :param SensorSignal curr:
            Current sensor signal containing current angle and velocity measurements.
        :param float timestamp:
            Current timestamp in seconds.
        :return:
            The new MotionState if a transition occurred, or None if no transition
            happened (either in timeout period or no valid trigger was active).
        :rtype: MotionState | None
        """
        if not self._is_timeout(timestamp=timestamp):
            self.triggers.set_triggers(curr=curr, prev=prev)
            new_state = self._detect_state()
            if new_state is not None:
                self.state = new_state
                self.timestamp_sec = timestamp
                return new_state
        return None
