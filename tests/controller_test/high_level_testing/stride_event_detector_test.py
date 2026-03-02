"""Tests for the stride event detector."""

from pandas import read_csv

from hip_controller.control.stride_event_detector import StrideEventDetector
from tests.conftest import HighLevelData, KinematicsDataColumnName


def test_stride_event_detector() -> None:
    """Test the stride event detector.

    :return: None
    """
    df = read_csv(filepath_or_buffer=HighLevelData.DATA_STRIDE_EVENT_DETECTOR)

    sed = StrideEventDetector()

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        curr_velocity = curr[KinematicsDataColumnName.VELOCITY]
        prev_velocity = prev[KinematicsDataColumnName.VELOCITY]
        curr_timestamp = curr[KinematicsDataColumnName.TIMESTAMP]
        prev_timestamp = prev[KinematicsDataColumnName.TIMESTAMP]

        dt = curr_timestamp - prev_timestamp

        sed._detect(dt=dt, prev_vel=prev_velocity, curr_vel=curr_velocity)

        expected_enable_detector = curr[KinematicsDataColumnName.ENABLE_DETECTOR]
        expected_valid_stride = curr[KinematicsDataColumnName.VALID_STRIDE]

        assert sed.valid_stride == expected_valid_stride, (
            f"Row {i}, time {sed.detector_counter_time}"
        )
        assert sed.enable_detector == expected_enable_detector, (
            f"Row {i}, time {sed.detector_counter_time}"
        )


def test_stride_event_trigger() -> None:
    """Test the stride event trigger and the sum of the stride event detector and trigger.

    :return: None
    """
    df = read_csv(filepath_or_buffer=HighLevelData.DATA_STRIDE_EVENT_DETECTOR)

    sed = StrideEventDetector()

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        valid_ang_max = curr[KinematicsDataColumnName.VALID_TRIGG_ANG_MAX]
        curr_velocity = curr[KinematicsDataColumnName.VELOCITY]
        prev_velocity = prev[KinematicsDataColumnName.VELOCITY]
        curr_timestamp = curr[KinematicsDataColumnName.TIMESTAMP]
        prev_timestamp = prev[KinematicsDataColumnName.TIMESTAMP]

        dt = curr_timestamp - prev_timestamp

        stride_event = sed.stride_event(
            dt=dt,
            curr_vel=curr_velocity,
            prev_vel=prev_velocity,
            valid_ang_max=valid_ang_max,
        )

        expected_enable_trigger = curr[KinematicsDataColumnName.ENABLE_TRIGGER]
        expected_stride_event = curr[KinematicsDataColumnName.STRIDE_EVENT]

        assert sed.enable_trigger == expected_enable_trigger, (
            f"Row {i}, time {sed.trigger_counter_time}, angle_max_trigger {valid_ang_max}"
        )
        assert stride_event == expected_stride_event, (
            f"Row {i}, time {sed.trigger_counter_time}"
        )
