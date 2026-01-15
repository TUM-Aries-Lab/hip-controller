"""Tests for high-level control functions. Unit tests in arrange-act-assert format."""

import pandas as pd
import pytest

from hip_controller.control.high_level import (
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
