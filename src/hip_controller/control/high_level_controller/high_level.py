"""High-level control functions transforming raw joint kinematics into a phase angle for control purposes."""

from math import atan2

from hip_controller.control.high_level_controller.motion_state_machine import (
    MotionState,
    MotionStateMachine,
)
from hip_controller.control.high_level_controller.steady_state_tracker import (
    SteadyStateTracker,
)
from hip_controller.control.high_level_controller.stride_event_detector import (
    StrideEventDetector,
)
from hip_controller.definitions import SensorSignal


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
        self.last_timestamp: float | None = None
        self.curr_timestamp: float | None = None

        self.prev_signal: SensorSignal = SensorSignal(
            angle_rad=0.0, velocity_rad_per_sec=0.0
        )
        self.curr_signal: SensorSignal = SensorSignal(
            angle_rad=0.0, velocity_rad_per_sec=0.0
        )

        self.stride_event_detector: StrideEventDetector = StrideEventDetector()
        self.state_machine: MotionStateMachine = MotionStateMachine()
        self.steady_state_tracker: SteadyStateTracker = SteadyStateTracker()

        self.controller_initialized: bool = False
        self.last_detection: bool = False

    def update_and_compute(self, curr_signal: SensorSignal, timestamp: float) -> float:
        """Update controller state with latest sensor data.

        :param SensorSignal curr_signal: Current hip joint angle in radians and current hip joint angular velocity in radians per second.
        :param float timestamp:  Current timestamp in seconds.

        :return: Gait phase of the hip joint in the sagittal plane.
        :rtype: float
        """
        # Processes current angle and velocity measurements, shifts previous signal to storage
        self.prev_signal = self.curr_signal
        self.curr_signal = curr_signal

        self.last_timestamp = self.curr_timestamp
        self.curr_timestamp = timestamp

        if self.last_timestamp is None:
            return 0

        # Updates the motion state machine to detect extrema transitions
        state = self.state_machine.update_motion_state(
            prev=self.prev_signal, curr=self.curr_signal, timestamp=timestamp
        )

        if state is not None:
            self.steady_state_tracker.update_extrema(
                state=state, curr_signal=self.curr_signal
            )

        # stride event detector
        curr_detector = self.stride_event_detector.stride_event(
            time_difference=timestamp - self.last_timestamp,
            valid_ang_max=(state == MotionState.ANGLE_MAX),
            prev_vel=self.prev_signal.velocity_rad_per_sec,
            curr_vel=self.curr_signal.velocity_rad_per_sec,
        )

        # Rising pattern of the stride event detector
        if not self.last_detection and curr_detector:
            self.steady_state_tracker.recenter()

        self.last_detection = curr_detector

        # Computes the steady-state gait phase parameters
        self.steady_state_tracker.update_steady_state(curr_signal=self.curr_signal)

        if self.stride_event_detector.valid_stride:
            self.controller_initialized = True

        if not self.controller_initialized:
            return 0
        else:
            return self.calculate_gait_phase()

    def get_signal_steady_state(self) -> SensorSignal:
        """Get normalized value of angle and velocity after the compute function was called.

        :return: Normalized velocity and rescaled angle.
        :rtype: SensorSignal
        """
        return SensorSignal(
            angle_rad=self.steady_state_tracker.ang_steady_state,
            velocity_rad_per_sec=self.steady_state_tracker.vel_steady_state,
        )

    def calculate_gait_phase(self) -> float:
        """Calculate the current gait phase as an angle in the phase plane that represents the current position in the gait cycle.

        :return: Gait phase angle in radians.
        :rtype: float
        """
        return atan2(
            self.steady_state_tracker.vel_steady_state,
            self.steady_state_tracker.ang_steady_state,
        )
