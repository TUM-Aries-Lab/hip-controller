"""Tests for high-level control functions. Unit tests in arrange-act-assert format."""

import pandas as pd
import pytest

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


@pytest.fixture(scope="module")
def zero_crossing_data():
    """Load motion data from CSV file."""
    df = pd.read_csv(
        "data/sensor_data/high_level_testing/zero_crossing_left_2026_01_09.csv"
    )
    return df


@pytest.fixture(scope="module")
def valid_trigger_data():
    """Load motion data from CSV file."""
    df = pd.read_csv(
        "data/sensor_data/high_level_testing/valid_trigger_left_2026_01_15.csv"
    )
    return df


def test_hit_zero_crossing_from_upper() -> None:
    """Test zero-crossing detection from upper to lower."""
    assert hit_zero_crossing_from_upper(prev=0.1, curr=-0.1)
    assert hit_zero_crossing_from_upper(prev=0.0, curr=-0.1)

    assert not hit_zero_crossing_from_upper(prev=0.0, curr=0.0)
    assert not hit_zero_crossing_from_upper(prev=-1.0, curr=0.0)
    assert not hit_zero_crossing_from_upper(prev=2.0, curr=1.0)
    assert not hit_zero_crossing_from_upper(prev=1.0, curr=0.0)


def test_hit_zero_crossing_from_lower() -> None:
    """Test zero-crossing detection from lower to upper."""
    assert hit_zero_crossing_from_lower(prev=-0.1, curr=0.1)
    assert hit_zero_crossing_from_lower(prev=0.0, curr=0.1)

    assert not hit_zero_crossing_from_lower(prev=0.0, curr=0.0)
    assert not hit_zero_crossing_from_lower(prev=1.0, curr=2.0)
    assert not hit_zero_crossing_from_lower(prev=-1.0, curr=-1.0)
    assert not hit_zero_crossing_from_lower(prev=-1.0, curr=0.0)


def test_extrema_trigger(zero_crossing_data) -> None:
    """Test angle extrema detection based on velocity zero-crossings."""
    df = zero_crossing_data

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        curr_velocity = curr["vel_left (rad/s)"]
        prev_velocity = prev["vel_left (rad/s)"]
        curr_angle = curr["angle_left (rad)"]
        prev_angle = prev["angle_left (rad)"]

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

        expected_vel_max = curr["vel_max_trigg_left"]
        expected_ang_max = curr["ang_max_trigg_left"]
        expected_vel_min = curr["vel_min_trigg_left"]
        expected_ang_min = curr["ang_min_trigg_left"]

        assert velocity_max == expected_vel_max, f"Row {i}"
        assert angle_max == expected_ang_max, f"Row {i}"
        assert velocity_min == expected_vel_min, f"Row {i}"
        assert angle_min == expected_ang_min, f"Row {i}"


def test_valid_trigger(valid_trigger_data) -> None:
    """Test angle extrema detection based on velocity zero-crossings."""
    df = valid_trigger_data
    controller = HighLevelController()

    for i in range(0, len(df)):
        curr = df.iloc[i]

        timestamp = curr["time (s)"]
        curr_velocity = curr["vel_left (rad/s)"]
        curr_angle = curr["angle_left (rad)"]

        controller.update(
            curr_angle=curr_angle, curr_vel=curr_velocity, timestamp=timestamp
        )

        vel_max = curr["valid_vel_max_left"]
        ang_max = curr["valid_ang_max_left"]
        vel_min = curr["valid_vel_min_left"]
        ang_min = curr["valid_ang_min_left"]

        if vel_max:
            assert controller.state == MotionState.VELOCITY_MAX, (
                f"Row {i}, vel_max {vel_max}"
            )
        if ang_max:
            assert controller.state == MotionState.ANGLE_MAX, f"Row {i}"
        if vel_min:
            assert controller.state == MotionState.VELOCITY_MIN, f"Row {i}"
        if ang_min:
            assert controller.state == MotionState.ANGLE_MIN, f"Row {i}"
