"""Detector for a new stride event to update the angle and velocity values."""

from hip_controller.definitions import (
    STRIDE_EVENT_COUNTER_TIME,
    STRIDE_EVENT_HIT_CROSSING_OFFSET,
)
from hip_controller.utils.math_utils import hit_crossing_falling


class StrideEventDetector:
    """Detector for a new stride event before the very first step.

    To smoothen the controller at the very beginning, the calculated gait phase is only returned after the first stride detection has occured. Before that, the gait phase is set to 0.
    """

    def __init__(self) -> None:
        """Initialize the StrideEventDetector for recalculation of the centered values and valid angle max triggers within last 31ms.

        :Return: None
        """
        # Valid stride detector
        self.valid_stride: bool = False
        self._enable_detector: bool = False
        self._detector_count = True
        self._detector_counter_time = 0.01  # Reset value is zero. It is only 0.01 while initializing because there is no delay time when the detector is enabled for the first time.

        # Valid trigger detector
        self._enable_trigger = True
        self._trigger_counter_time = 0.0

        # Previous timestamp to calculate timestamp difference
        self._prev_timestamp = None

    def is_valid_stride_event(
        self,
        time_difference: float,
        valid_ang_max: bool,
        prev_vel: float,
        curr_vel: float,
    ) -> bool:
        """Detect stride event and trigger the update of centered values when valid angle maximum is reached.

        :param dt: Time difference between current and previous timestamp.
        :param valid_ang_max: Whether the current angle is a valid maximum for triggering the update
        :param prev_vel: Previous velocity value.
        :param curr_vel: Current velocity value.

        :return: True if stride event is detected and valid angle maximum is reached, False otherwise
        """
        self._detect_new_stride(
            time_difference=time_difference, prev_vel=prev_vel, curr_vel=curr_vel
        )
        self._detect_new_gait(
            time_difference=time_difference, valid_ang_max=valid_ang_max
        )

        return self.valid_stride and self._enable_trigger

    def _detect_new_stride(
        self, time_difference: float, prev_vel: float, curr_vel: float
    ) -> None:
        """Detect stride and proceed the countdown steps when necessary.

        :param time_difference: Time difference between current and previous timestamp.
        :param prev_vel: Previous velocity value.
        :param curr_vel: Current velocity value.
        :return: None
        """
        if self._detector_count and self._countdown_detected(time_difference):
            self._reset()

        elif self._enable_detector:
            # Check if a stride event is detected based on velocity crossing with an offset
            if hit_crossing_falling(
                curr=curr_vel, prev=prev_vel, offset=STRIDE_EVENT_HIT_CROSSING_OFFSET
            ):
                self.valid_stride = True
                self._detector_count = True

    def _detect_new_gait(self, time_difference: float, valid_ang_max: bool) -> None:
        """Trigger the update of centered values when valid angle maximum is reached and proceed the countdown steps when necessary.

        :param dt: Time difference between current and previous timestamp.
        :param valid_ang_max: Whether the current angle is a valid maximum for triggering the update
        :return: None
        """
        if valid_ang_max:
            self._enable_trigger = True

        elif self._enable_trigger and self._countdown_triggered(time_difference):
            self._enable_trigger = False
            self._trigger_counter_time = 0

    def _countdown_triggered(self, time_difference: float) -> bool:
        """Check if the trigger countdown has reached the specified time threshold.

        :param dt: Time difference between current and previous timestamp.
        :return: True if the stride event detector countdown has reached the specified time threshold, False otherwise.
        """
        self._trigger_counter_time += time_difference
        self._enable_trigger = True
        return self._trigger_counter_time >= STRIDE_EVENT_COUNTER_TIME

    def _countdown_detected(self, time_difference: float) -> bool:
        """Check if the stride event detector countdown has reached the specified time threshold.

        :param dt: Time difference between current and previous timestamp.

        :return: True if the stride event detector countdown has reached the specified time threshold, False otherwise.
        """
        self._detector_counter_time += time_difference
        self._enable_detector = False
        return self._detector_counter_time >= STRIDE_EVENT_COUNTER_TIME

    def _reset(self) -> None:
        """Reset the stride event detector to the initial state.

        :return: None
        """
        self._detector_counter_time = 0.0
        self._detector_count = False
        self.valid_stride = False
        self._enable_detector = True
