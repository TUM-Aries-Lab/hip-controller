"""High-level control functions."""

import math
from dataclasses import dataclass
from enum import Enum

from hip_controller.definitions import (
    LAG_CORRECTION,
    VALUE_NEAR_ZERO,
    PositionLimitation,
    StateChangeTimeThreshold,
)
from hip_controller.math_utils import (
    hit_zero_crossing_from_lower,
    hit_zero_crossing_from_upper,
    normalize,
)


class MotionState(Enum):
    """Enumeration of motion states in the gait cycle.

    Represents the four fundamental states of periodic motion as detected through
    extrema analysis of joint angle and angular velocity. The state machine cycles
    through these states in order: VELOCITY_MAX → ANGLE_MAX → VELOCITY_MIN
    → ANGLE_MIN → (back to INITIAL anytime).
    """

    INITIAL = -1
    VELOCITY_MAX = 0
    ANGLE_MAX = 1
    VELOCITY_MIN = 2
    ANGLE_MIN = 3


@dataclass
class SensorSignal:
    """Container for angle and velocity measurements from the sensor.

    Represents a single snapshot of kinematic data (angle and velocity) read from
    the hip joint sensor at a specific point in time. Used throughout the control
    system to maintain consistent representation of joint state.
    """

    angle: float = 0.0
    velocity: float = 0.0


class HighLevelController:
    """High-level motion controller for gait analysis and state tracking.

    Manages the overall control logic by tracking sensor measurements, detecting
    motion extrema through a state machine, and computing steady-state gait phase
    parameters. Combines motion state detection with steady-state signal analysis
    to provide real-time gait phase information for downstream control modules.
    """

    def __init__(self):
        """Initialize the HighLevelController.

        Sets up initial sensor signals (angle and velocity) at zero, creates the
        motion state machine, initializes the steady-state tracker, and sets the
        initial gait phase to zero.

        :return: None
        """
        self.prev_signal: SensorSignal = SensorSignal(angle=0.0, velocity=0.0)
        self.curr_signal: SensorSignal = SensorSignal(angle=0.0, velocity=0.0)

        self.state_machine: MotionStateMachine = MotionStateMachine()
        self.steady_state_tracker: SteadyStateTracker = SteadyStateTracker()

        self.sinusoidal_behavior: float = 0.0

    def compute(self, curr_angle: float, curr_vel: float, timestamp: float) -> float:
        """Update controller state with latest sensor measurements - sensor signals, motion state, extrema values, and compute the sinosoidal-like behaviour of the hip joint in the sagittal plane with the normalized and centered gait phase value.

        Processes current angle and velocity measurements, shifts previous signal
        to storage, updates the motion state machine to detect extrema transitions,
        and computes the steady-state gait phase parameters. This should be called
        once per control cycle with the latest sensor data.

        :param float curr_angle:
            Current hip joint angle in radians.
        :param float curr_vel:
            Current hip joint angular velocity in radians per second.
        :param float timestamp:
            Current timestamp in seconds.
        :return:
            None.
        :rtype: None

        :return:
            Sinosoidal-like behaviour of the hip joint in the sagittal plane.
        :rtype: float
        """
        self.prev_signal = self.curr_signal
        self.curr_signal = SensorSignal(angle=curr_angle, velocity=curr_vel)

        state = self.state_machine.update_motion_state(
            prev=self.prev_signal, curr=self.curr_signal, timestamp=timestamp
        )
        if state is not None:
            self.steady_state_tracker.update_extrema(
                state=state, curr_signal=self.curr_signal
            )
        self.steady_state_tracker.update_steady_state(curr_signal=self.curr_signal)

        return self.center_and_transform_gaitphase(
            self.steady_state_tracker.calculate_gait_phase()
        )

    @staticmethod
    def center_and_transform_gaitphase(gait_phase: float) -> float:
        """Center and transform the gait phase into a sinusoidal control signal.

        Applies a phase offset and sinusoidal transformation to the
        computed gait phase, producing a normalized control signal
        suitable for downstream controllers.


        :param float gait_phase: Gait phase angle in radians.

        :return: Transformed sinusoidal signal derived from the gait phase.
        :rtype: float
        """
        return -math.sin(gait_phase + LAG_CORRECTION)


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
            curr_angle=curr.angle, prev_angle=prev.angle, curr_velocity=curr.velocity
        )
        self.ang_max = self._angle_max_trigger(
            curr_velocity=curr.velocity,
            prev_velocity=prev.velocity,
            curr_angle=curr.angle,
        )
        self.vel_min = self._velocity_min_trigger(
            curr_angle=curr.angle, prev_angle=prev.angle, curr_velocity=curr.velocity
        )
        self.ang_min = self._angle_min_trigger(
            curr_velocity=curr.velocity,
            prev_velocity=prev.velocity,
            curr_angle=curr.angle,
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


class SteadyStateTracker:
    """Tracker and calculator for steady-state gait phase parameters.

    Maintains extrema values (angle and velocity maxima and minima) and computes
    normalized steady-state values for both angle and velocity. Provides gait phase
    calculation through normalization of these steady-state values, transforming
    raw joint kinematics into a phase angle for control purposes.
    """

    def __init__(self):
        """Initialize the SteadyStateTracker.

        Sets up initial tracking variables for extrema (angle_max, angle_min,
        velocity_max, velocity_min) and steady-state parameters (velocity steady-state,
        position steady-state, and rescale factor) all set to zero. These are populated
        through the update methods as the system detects motion extrema.

        :return: None
        """
        self.angle_max: float = 0.0
        self.angle_min: float = 0.0
        self.velocity_max: float = 0.0
        self.velocity_min: float = 0.0

        self.vel_steady_state: float = 0.0
        self.rescale_factor: float = 0.0
        self.pos_steady_state: float = 0.0

    def _calculate_vel_ss(self, curr_velocity: float) -> float:
        """Calculate normalized steady-state velocity.

        Computes the steady-state velocity by centering the current velocity around
        the midpoint of the velocity extrema (velocity_max and velocity_min). The
        result is zero when current velocity equals the midpoint.

        :param float curr_velocity:
            Current velocity value.
        :return:
            Normalized steady-state velocity centered at zero.
        :rtype: float
        """
        return normalize(
            val_max=self.velocity_max,
            val_min=self.velocity_min,
            val_curr=curr_velocity,
        )

    def _calculate_ang_ss(self, curr_angle: float) -> float:
        """Calculate normalized steady-state angle.

        Computes the steady-state angle by centering the current angle around the
        midpoint of the angle extrema (angle_max and angle_min). The result is zero
        when current angle equals the midpoint.

        :param float curr_angle:
            Current angle value.
        :return:
            Normalized steady-state angle centered at zero.
        :rtype: float
        """
        return normalize(
            val_max=self.angle_max, val_min=self.angle_min, val_curr=curr_angle
        )

    def _calculate_rescale_factor(self) -> float:
        """Calculate the velocity-to-angle scaling factor.

        Computes the ratio of velocity range to angle range for normalizing the
        position steady-state value. This factor scales angle measurements to match
        velocity magnitude for proper phase plane representation. Handles the case
        of zero angle range to avoid division by zero.

        :return:
            Scaling factor (velocity_range / angle_range). If angle_range is zero,
            uses a near-zero value to prevent division by zero.
        :rtype: float
        """
        u_vel = abs(self.velocity_max - self.velocity_min)
        u_ang = abs(self.angle_max - self.angle_min)

        # Avoid division by zero
        if u_ang == 0.0:
            u_ang = VALUE_NEAR_ZERO

        return u_vel / u_ang

    def _calculate_pos_ss(self, curr_angle: float) -> float:
        """Calculate normalized position (angle) steady-state scaled by velocity range.

        Computes the position steady-state by scaling the normalized angle by the
        velocity-to-angle rescale factor. This creates a phase plane representation
        where angle is normalized to match velocity magnitude, enabling proper
        gait phase calculation via arctangent in the velocity-angle plane.

        :param float curr_angle:
            Current angle value.
        :return:
            Scaled position steady-state value for phase plane representation.
        :rtype: float
        """
        # This has to happen after z_t is set
        return self.rescale_factor * self._calculate_ang_ss(curr_angle=curr_angle)

    def calculate_gait_phase(self) -> float:
        """Calculate the current gait phase as an angle in the phase plane.

        Computes the gait phase using arctangent of normalized velocity and scaled
        position steady-state values. Returns an angle in the phase plane that
        represents the current position in the gait cycle. Returns 0.0 if the
        rescale factor is zero (uninitialized state).

        :return:
            Gait phase angle in radians, typically in the range [-π, π].
            Computed as atan2(vel_steady_state, -pos_steady_state).
        :rtype: float
        """
        if self.rescale_factor == 0.0:
            return 0.0
        else:
            return math.atan2(self.vel_steady_state, -self.pos_steady_state)

    def update_steady_state(self, curr_signal: SensorSignal) -> None:
        """Update steady-state parameters based on current sensor signal.

        Recomputes velocity steady-state, rescale factor, and position steady-state
        from the current sensor measurements. Applies validation: rescale factor
        must be a valid number, and position steady-state must be within the
        specified position limitation bounds. Invalid values are rejected to maintain
        state consistency.

        :param SensorSignal curr_signal:
            Current sensor signal containing angle and velocity.
        :return:
            None. Updates instance variables: vel_steady_state, rescale_factor,
            and pos_steady_state (only if values pass validation).
        :rtype: None
        """
        self.vel_steady_state = self._calculate_vel_ss(
            curr_velocity=curr_signal.velocity
        )

        rescale_factor = self._calculate_rescale_factor()
        if not math.isnan(rescale_factor):
            self.rescale_factor = rescale_factor

        pos_ss = self._calculate_pos_ss(curr_angle=curr_signal.angle)
        if PositionLimitation.LOWER <= pos_ss <= PositionLimitation.UPPER:
            self.pos_steady_state = pos_ss

    def update_extrema(self, state: MotionState, curr_signal: SensorSignal) -> None:
        """Update extrema values when new motion state is detected.

        Records the current angle or velocity measurement as an extremum value based
        on the detected motion state. This is called when a new state is reached to
        capture the extrema values (angle_max, angle_min, velocity_max, velocity_min)
        that define the motion bounds for steady-state normalization.

        :param MotionState state:
            The newly detected motion state indicating which extremum was reached.
        :param SensorSignal curr_signal:
            Current sensor signal containing angle and velocity to be recorded.
        :return:
            None. Updates one of: angle_max, angle_min, velocity_max, or velocity_min
            based on the provided state.
        :rtype: None
        """
        if state == MotionState.ANGLE_MAX:
            self.angle_max = curr_signal.angle

        elif state == MotionState.ANGLE_MIN:
            self.angle_min = curr_signal.angle

        elif state == MotionState.VELOCITY_MAX:
            self.velocity_max = curr_signal.velocity

        elif state == MotionState.VELOCITY_MIN:
            self.velocity_min = curr_signal.velocity
