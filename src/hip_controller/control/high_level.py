"""High-level control functions."""

from math import atan2, isnan

from hip_controller.control.motion_state_machine import MotionState, MotionStateMachine
from hip_controller.definitions import (
    VALUE_NEAR_ZERO,
    PositionLimitation,
    SensorSignal,
)
from hip_controller.utils.math_utils import center, normalize


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
        self.prev_signal: SensorSignal = SensorSignal(
            angle_rad=0.0, velocity_rad_per_sec=0.0
        )
        self.curr_signal: SensorSignal = SensorSignal(
            angle_rad=0.0, velocity_rad_per_sec=0.0
        )

        self.state_machine: MotionStateMachine = MotionStateMachine()
        self.steady_state_tracker: SteadyStateTracker = SteadyStateTracker()

    def update_and_compute(self, curr_signal: SensorSignal, timestamp: float) -> float:
        """Update controller state with latest sensor data.

        :param SensorSignal curr_signal: Current hip joint angle in radians and current hip joint angular velocity in radians per second.
        :param float timestamp:  Current timestamp in seconds.

        :return:
            Gait phase of the hip joint in the sagittal plane.
        :rtype: float
        """
        # Processes current angle and velocity measurements, shifts previous signal to storage
        self.prev_signal = self.curr_signal
        self.curr_signal = curr_signal

        # Updates the motion state machine to detect extrema transitions
        state = self.state_machine.update_motion_state(
            prev=self.prev_signal, curr=self.curr_signal, timestamp=timestamp
        )
        if state is not None:
            self.steady_state_tracker.update_extrema(
                state=state, curr_signal=self.curr_signal
            )

        # Computes the steady-state gait phase parameters
        self.steady_state_tracker.update_steady_state(curr_signal=self.curr_signal)

        return self.steady_state_tracker.calculate_gait_phase()

    def get_signal_steady_state(self) -> SensorSignal:
        """Get normalized value of angle and velocity after the compute function was called.

        :return: Normalized velocity and rescaled angle.
        :rtype: SensorSignal
        """
        return SensorSignal(
            angle_rad=self.steady_state_tracker.pos_steady_state,
            velocity_rad_per_sec=self.steady_state_tracker.vel_steady_state,
        )


class StrideEventDetector:  # pragma no cover
    """Detector for a new stride event before the very first step.

    To smoothen the controller at the very beginning, the calculated gait phase is only returned after the first stride detection has occured. Before that, the gait phase is set to 0.
    """

    # TODO stride event detector for recalculation of the centered values within last 31ms
    def __init__(self):
        """Initialize the StrideEventDetector."""


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

    def _calculate_velocity_steady_state(self, curr_velocity: float) -> float:
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
            center_val=center(val_max=self.velocity_max, val_min=self.velocity_min),
            val_curr=curr_velocity,
        )

    def _calculate_centered_angle(self, curr_angle: float) -> float:
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
            center_val=center(val_max=self.angle_max, val_min=self.angle_min),
            val_curr=curr_angle,
        )

    def _calculate_rescale_factor(self) -> float:
        """Calculate gamma - the velocity-to-angle scaling factor.

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

    def _calculate_angle_steady_state(self, curr_angle: float) -> float:
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
        return self.rescale_factor * self._calculate_centered_angle(
            curr_angle=curr_angle
        )

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
            return atan2(self.vel_steady_state, -self.pos_steady_state)

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
        self.vel_steady_state = self._calculate_velocity_steady_state(
            curr_velocity=curr_signal.velocity_rad_per_sec
        )

        rescale_factor = self._calculate_rescale_factor()
        if not isnan(rescale_factor):
            self.rescale_factor = rescale_factor

        ang_ss = self._calculate_angle_steady_state(curr_angle=curr_signal.angle_rad)
        if PositionLimitation.LOWER <= ang_ss <= PositionLimitation.UPPER:
            self.pos_steady_state = ang_ss

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
            self.angle_max = curr_signal.angle_rad

        elif state == MotionState.ANGLE_MIN:
            self.angle_min = curr_signal.angle_rad

        elif state == MotionState.VELOCITY_MAX:
            self.velocity_max = curr_signal.velocity_rad_per_sec

        elif state == MotionState.VELOCITY_MIN:
            self.velocity_min = curr_signal.velocity_rad_per_sec
