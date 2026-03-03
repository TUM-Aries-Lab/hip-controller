"""Tests for the motion state machine."""

from pandas import read_csv

from hip_controller.control.high_level_controller.high_level import (
    MotionState,
    MotionStateMachine,
    SensorSignal,
)
from hip_controller.utils.math_utils import (
    hit_crossing_falling,
    hit_crossing_rising,
)
from tests.conftest import HighLevelData, KinematicsDataColumnName


def test_extrema_trigger() -> None:
    """Test angle extrema detection based on velocity zero-crossings.

    :return: None
    """
    df = read_csv(filepath_or_buffer=HighLevelData.DATA_ZERO_CROSSING)

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        curr_velocity = curr[KinematicsDataColumnName.VELOCITY]
        prev_velocity = prev[KinematicsDataColumnName.VELOCITY]
        curr_angle = curr[KinematicsDataColumnName.ANGLE]
        prev_angle = prev[KinematicsDataColumnName.ANGLE]

        angle_max = hit_crossing_falling(curr=curr_velocity, prev=prev_velocity)
        angle_min = hit_crossing_rising(curr=curr_velocity, prev=prev_velocity)
        velocity_max = hit_crossing_rising(curr=curr_angle, prev=prev_angle)
        velocity_min = hit_crossing_falling(curr=curr_angle, prev=prev_angle)

        expected_vel_max = curr[KinematicsDataColumnName.TRIGG_VEL_MAX]
        expected_ang_max = curr[KinematicsDataColumnName.TRIGG_ANG_MAX]
        expected_vel_min = curr[KinematicsDataColumnName.TRIGG_VEL_MIN]
        expected_ang_min = curr[KinematicsDataColumnName.TRIGG_ANG_MIN]

        assert velocity_max == expected_vel_max, f"Row {i}"
        assert angle_max == expected_ang_max, f"Row {i}"
        assert velocity_min == expected_vel_min, f"Row {i}"
        assert angle_min == expected_ang_min, f"Row {i}"


def test_valid_trigger() -> None:
    """Test angle extrema detection based on velocity zero-crossings.

    :return: None
    """
    df = read_csv(filepath_or_buffer=HighLevelData.DATA_VALID_TRIGGER)
    state_machine = MotionStateMachine()

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        timestamp = curr[KinematicsDataColumnName.TIMESTAMP]
        prev_signal = SensorSignal(
            angle_rad=prev[KinematicsDataColumnName.ANGLE],
            velocity_rad_per_sec=prev[KinematicsDataColumnName.VELOCITY],
        )
        curr_signal = SensorSignal(
            angle_rad=curr[KinematicsDataColumnName.ANGLE],
            velocity_rad_per_sec=curr[KinematicsDataColumnName.VELOCITY],
        )

        state_machine.update_motion_state(
            curr=curr_signal, prev=prev_signal, timestamp=timestamp
        )

        vel_max = curr[KinematicsDataColumnName.VALID_TRIGG_VEL_MAX]
        ang_max = curr[KinematicsDataColumnName.VALID_TRIGG_ANG_MAX]
        vel_min = curr[KinematicsDataColumnName.VALID_TRIGG_VEL_MIN]
        ang_min = curr[KinematicsDataColumnName.VALID_TRIGG_ANG_MIN]

        if vel_max:
            assert state_machine.state == MotionState.VELOCITY_MAX, (
                f"Row {i}, vel_max {vel_max}, angle_max {ang_max}, vel_min {vel_min}, ang_min {ang_min}"
            )
        if ang_max:
            assert state_machine.state == MotionState.ANGLE_MAX, (
                f"Row {i}, vel_max {vel_max}, angle_max {ang_max}, vel_min {vel_min}, ang_min {ang_min}"
            )
        if vel_min:
            assert state_machine.state == MotionState.VELOCITY_MIN, (
                f"Row {i}, vel_max {vel_max}, angle_max {ang_max}, vel_min {vel_min}, ang_min {ang_min}"
            )
        if ang_min:
            assert state_machine.state == MotionState.ANGLE_MIN, (
                f"Row {i}, vel_max {vel_max}, angle_max {ang_max}, vel_min {vel_min}, ang_min {ang_min}"
            )
