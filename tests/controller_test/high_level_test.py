"""Tests for high-level control functions. Unit tests in arrange-act-assert format."""

import pandas as pd
import pytest

from hip_controller.control.high_level import (
    MotionState,
    angle_extrema_trigger,
    angle_max_trigger,
    angle_min_trigger,
    check_state_validity,
    hit_zero_crossing_from_lower,
    hit_zero_crossing_from_upper,
    velocity_extrema_trigger,
    velocity_max_trigger,
    velocity_min_trigger,
)


@pytest.fixture(scope="module")
def zero_crossing_data():
    """Load motion data from CSV file."""
    df = pd.read_csv("data/sensor_data/high_level_testing/zero_crossing_trigger_L.csv")
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


def test_angle_extrema_trigger(zero_crossing_data) -> None:
    """Test angle extrema detection based on velocity zero-crossings."""
    df = zero_crossing_data

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        detected = angle_extrema_trigger(
            curr_velocity=curr["vel_L_filtered"],
            prev_velocity=prev["vel_L_filtered"],
        )

        if curr["ang_max_trigg_L"]:
            assert detected == MotionState.ANGLE_MAX, f"Row {i}"

        elif curr["ang_min_trigg_L"]:
            assert detected == MotionState.ANGLE_MIN, f"Row {i}"

        else:
            assert detected == MotionState.INITIAL


def test_velocity_extrema_trigger(zero_crossing_data):
    """Test motion state detection using data from CSV file."""
    df = zero_crossing_data

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        detected = velocity_extrema_trigger(
            curr_angle=curr["angle_L_filtered"],
            prev_angle=prev["angle_L_filtered"],
        )

        # VELOCITY extrema (angle zero-crossings)
        if curr["speed_max_trigg_L"]:
            assert detected == MotionState.VELOCITY_MAX, f"Row {i}"

        elif curr["speed_min_trigg_L"]:
            assert detected == MotionState.VELOCITY_MIN, f"Row {i}"

        else:
            assert detected == MotionState.INITIAL, f"Row {i}"


def test_extrema_trigger(zero_crossing_data) -> None:
    """Test angle extrema detection based on velocity zero-crossings."""
    df = zero_crossing_data

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        angle_max = angle_max_trigger(
            curr_velocity=curr["vel_L_filtered"],
            prev_velocity=prev["vel_L_filtered"],
        )

        angle_min = angle_min_trigger(
            curr_velocity=curr["vel_L_filtered"],
            prev_velocity=prev["vel_L_filtered"],
        )
        velocity_max = velocity_max_trigger(
            curr_angle=curr["angle_L_filtered"],
            prev_angle=prev["angle_L_filtered"],
        )
        velocity_min = velocity_min_trigger(
            curr_angle=curr["angle_L_filtered"],
            prev_angle=prev["angle_L_filtered"],
        )

        assert angle_max == curr["ang_max_trigg_L"], f"Row {i}"

        assert angle_min == curr["ang_min_trigg_L"], f"Row {i}"

        assert velocity_max == curr["speed_max_trigg_L"], f"Row {i}"

        assert velocity_min == curr["speed_min_trigg_L"], f"Row {i}"


def test_check_state_validity():
    """Test the state validity checking function."""
    # prev_state INITIAL -> accept any curr_state
    assert (
        check_state_validity(None, MotionState.INITIAL, MotionState.VELOCITY_MIN)
        == MotionState.VELOCITY_MIN
    )

    # valid cyclic transitions
    assert (
        check_state_validity(None, MotionState.VELOCITY_MAX, MotionState.ANGLE_MAX)
        == MotionState.ANGLE_MAX
    )
    # wrap-around valid transition: ANGLE_MIN -> VELOCITY_MAX
    assert (
        check_state_validity(None, MotionState.ANGLE_MIN, MotionState.VELOCITY_MAX)
        == MotionState.VELOCITY_MAX
    )

    # invalid transition should return previous state
    assert (
        check_state_validity(None, MotionState.VELOCITY_MAX, MotionState.VELOCITY_MIN)
        == MotionState.VELOCITY_MAX
    )
