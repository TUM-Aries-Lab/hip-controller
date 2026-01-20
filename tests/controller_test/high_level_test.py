"""Tests for high-level control functions. Unit tests in arrange-act-assert format."""

import pandas as pd

from hip_controller.control.high_level import (
    HighLevelController,
    MotionState,
    angle_max_trigger,
    angle_min_trigger,
    hit_zero_crossing_from_lower,
    hit_zero_crossing_from_upper,
    velocity_max_trigger,
    velocity_min_trigger,
)
from hip_controller.definitions import ColumnName, HighLevelDataDir


def test_hit_zero_crossing_from_upper() -> None:
    """Test zero-crossing detection from upper to lower.

    :return: None
    """
    assert hit_zero_crossing_from_upper(prev=0.1, curr=-0.1)
    assert hit_zero_crossing_from_upper(prev=0.0, curr=-0.1)

    assert not hit_zero_crossing_from_upper(prev=0.0, curr=0.0)
    assert not hit_zero_crossing_from_upper(prev=-1.0, curr=0.0)
    assert not hit_zero_crossing_from_upper(prev=2.0, curr=1.0)
    assert not hit_zero_crossing_from_upper(prev=1.0, curr=0.0)


def test_hit_zero_crossing_from_lower() -> None:
    """Test zero-crossing detection from lower to upper.

    :return: None
    """
    assert hit_zero_crossing_from_lower(prev=-0.1, curr=0.1)
    assert hit_zero_crossing_from_lower(prev=0.0, curr=0.1)

    assert not hit_zero_crossing_from_lower(prev=0.0, curr=0.0)
    assert not hit_zero_crossing_from_lower(prev=1.0, curr=2.0)
    assert not hit_zero_crossing_from_lower(prev=-1.0, curr=-1.0)
    assert not hit_zero_crossing_from_lower(prev=-1.0, curr=0.0)


def test_extrema_trigger() -> None:
    """Test angle extrema detection based on velocity zero-crossings.

    :return: None
    """
    df = pd.read_csv(HighLevelDataDir.DATA_ZERO_CROSSING)

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        curr_velocity = curr[ColumnName.VELOCITY]
        prev_velocity = prev[ColumnName.VELOCITY]
        curr_angle = curr[ColumnName.ANGLE]
        prev_angle = prev[ColumnName.ANGLE]

        angle_max = angle_max_trigger(
            curr_velocity=curr_velocity, prev_velocity=prev_velocity
        )

        angle_min = angle_min_trigger(
            curr_velocity=curr_velocity, prev_velocity=prev_velocity
        )

        velocity_max = velocity_max_trigger(
            curr_angle=curr_angle, prev_angle=prev_angle
        )
        velocity_min = velocity_min_trigger(
            curr_angle=curr_angle, prev_angle=prev_angle
        )

        expected_vel_max = curr[ColumnName.TRIGG_VEL_MAX]
        expected_ang_max = curr[ColumnName.TRIGG_ANG_MAX]
        expected_vel_min = curr[ColumnName.TRIGG_VEL_MIN]
        expected_ang_min = curr[ColumnName.TRIGG_ANG_MIN]

        assert velocity_max == expected_vel_max, f"Row {i}"
        assert angle_max == expected_ang_max, f"Row {i}"
        assert velocity_min == expected_vel_min, f"Row {i}"
        assert angle_min == expected_ang_min, f"Row {i}"


def test_valid_trigger() -> None:
    """Test angle extrema detection based on velocity zero-crossings.

    :return: None
    """
    df = pd.read_csv(HighLevelDataDir.DATA_VALID_TRIGGER)
    controller = HighLevelController()

    for i in range(0, len(df)):
        curr = df.iloc[i]

        timestamp = curr[ColumnName.TIMESTAMP]
        curr_velocity = curr[ColumnName.VELOCITY]
        curr_angle = curr[ColumnName.ANGLE]

        controller.update(
            curr_angle=curr_angle, curr_vel=curr_velocity, timestamp=timestamp
        )

        vel_max = curr[ColumnName.VALID_TRIGG_VEL_MAX]
        ang_max = curr[ColumnName.VALID_TRIGG_ANG_MAX]
        vel_min = curr[ColumnName.VALID_TRIGG_VEL_MIN]
        ang_min = curr[ColumnName.VALID_TRIGG_ANG_MIN]

        if vel_max:
            assert controller.state == MotionState.VELOCITY_MAX, (
                f"Row {i}, vel_max {vel_max}, angle_max {ang_max}, vel_min {vel_min}, ang_min {ang_min}"
            )
        if ang_max:
            assert controller.state == MotionState.ANGLE_MAX, (
                f"Row {i}, vel_max {vel_max}, angle_max {ang_max}, vel_min {vel_min}, ang_min {ang_min}"
            )
        if vel_min:
            assert controller.state == MotionState.VELOCITY_MIN, (
                f"Row {i}, vel_max {vel_max}, angle_max {ang_max}, vel_min {vel_min}, ang_min {ang_min}"
            )
        if ang_min:
            assert controller.state == MotionState.ANGLE_MIN, (
                f"Row {i}, vel_max {vel_max}, angle_max {ang_max}, vel_min {vel_min}, ang_min {ang_min}"
            )
