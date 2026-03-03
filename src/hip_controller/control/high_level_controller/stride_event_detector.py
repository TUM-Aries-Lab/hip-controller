"""Detector for a new stride event to update the angle and velocity values."""

from hip_controller.definitions import STRIDE_EVENT_COUNTER_TIME
from hip_controller.utils.math_utils import hit_crossing_falling


@staticmethod
def is_stride_detected(prev_vel: float, curr_vel: float) -> bool:
    """Check if a stride event is detected based on velocity crossing with an offset of -0.1.

    :param prev_vel: Previous velocity value.
    :param curr_vel: Current velocity value.
    :return: True if stride is detected. False otherwise.
    """
    return hit_crossing_falling(curr=curr_vel, prev=prev_vel, offset=-0.1)


class StrideEventDetector:
    """Detector for a new stride event before the very first step.

    To smoothen the controller at the very beginning, the calculated gait phase is only returned after the first stride detection has occured. Before that, the gait phase is set to 0.
    """

    def __init__(self) -> None:
        """Initialize the StrideEventDetector for recalculation of the centered values and valid angle max triggers within last 31ms.

        :Return: None
        """
        # Valid stride detector
        self.enable_detector: bool = False
        self.valid_stride: bool = False
        self.detector_count = True
        self.detector_counter_time = 0.01  # Reset value is zero. It is only 0.01 while initializing because there is no delay time when the detector is enabled for the first time.

        # Valid trigger detector
        self.enable_trigger = True
        self.trigger_counter_time = 0.0

        # Previous timestamp to calculate timestamp difference
        self.prev_timestamp = None

    def stride_event(
        self, dt: float, valid_ang_max: bool, prev_vel: float, curr_vel: float
    ) -> bool:
        """Detect stride event and trigger the update of centered values when valid angle maximum is reached.

        :param dt: Time difference between current and previous timestamp.
        :param valid_ang_max: Whether the current angle is a valid maximum for triggering the update
        :param prev_vel: Previous velocity value.
        :param curr_vel: Current velocity value.

        :return: True if stride event is detected and valid angle maximum is reached, False otherwise
        """
        self._detect(dt=dt, prev_vel=prev_vel, curr_vel=curr_vel)
        self._refract(dt=dt, valid_ang_max=valid_ang_max)

        return self.valid_stride and self.enable_trigger

    def _detect(self, dt: float, prev_vel: float, curr_vel: float) -> None:
        """Detect stride and proceed the countdown steps when necessary.

        :param dt: Time difference between current and previous timestamp.
        :param prev_vel: Previous velocity value.
        :param curr_vel: Current velocity value.
        :return: None
        """
        if self.detector_count and self._countdown_detector(dt):
            self._reset()

        elif self.enable_detector:
            if is_stride_detected(prev_vel=prev_vel, curr_vel=curr_vel):
                self.valid_stride = True
                self.detector_count = True

    def _refract(self, dt: float, valid_ang_max: bool) -> None:
        """Trigger the update of centered values when valid angle maximum is reached and proceed the countdown steps when necessary.

        :param dt: Time difference between current and previous timestamp.
        :param valid_ang_max: Whether the current angle is a valid maximum for triggering the update
        :return: None
        """
        if valid_ang_max:
            self.enable_trigger = True

        elif self.enable_trigger and self._countdown_trigger(dt):
            self.enable_trigger = False
            self.trigger_counter_time = 0

    def _countdown_trigger(self, dt: float) -> bool:
        """Check if the trigger countdown has reached the specified time threshold.

        :param dt: Time difference between current and previous timestamp.
        :return: True if the stride event detector countdown has reached the specified time threshold, False otherwise.
        """
        self.trigger_counter_time += dt
        self.enable_trigger = True
        return self.trigger_counter_time >= STRIDE_EVENT_COUNTER_TIME

    def _countdown_detector(self, dt: float) -> bool:
        """Check if the stride event detector countdown has reached the specified time threshold.

        :param dt: Time difference between current and previous timestamp.

        :return: True if the stride event detector countdown has reached the specified time threshold, False otherwise.
        """
        self.detector_counter_time += dt
        self.enable_detector = False
        return self.detector_counter_time >= STRIDE_EVENT_COUNTER_TIME

    def _reset(self) -> None:
        """Reset the stride event detector to the initial state.

        :return: None
        """
        self.detector_counter_time = 0.0
        self.detector_count = False
        self.valid_stride = False
        self.enable_detector = True
