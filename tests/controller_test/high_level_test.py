"""Tests for high-level control functions. Unit tests in arrange-act-assert format.

=================
Note on numerical precision:

Reference outputs used in these tests were generated in MATLAB.
Due to implementation-dependent floating-point arithmetic and
numerical backend differences between MATLAB and Python, exact
equality comparisons are not reliable.

Therefore, floating-point values are compared using ``math.isclose``
with a relative tolerance. The tolerance is chosen as the smallest
value for which the test passes, ensuring strict yet numerically
robust validation.
"""

from math import isclose

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
from hip_controller.definitions import ColumnName, HighLevelData


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
    df = pd.read_csv(HighLevelData.DATA_ZERO_CROSSING)

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
    df = pd.read_csv(HighLevelData.DATA_VALID_TRIGGER)
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


def test_set_state() -> None:
    """Test angle extrema detection based on velocity zero-crossings.

    :return: None
    """
    df = pd.read_csv(HighLevelData.DATA_EXTREMA_VALUES)
    controller = HighLevelController()

    for i in range(0, len(df)):
        curr = df.iloc[i]

        timestamp = curr[ColumnName.TIMESTAMP]
        curr_velocity = curr[ColumnName.VELOCITY]
        curr_angle = curr[ColumnName.ANGLE]

        controller.update(
            curr_angle=curr_angle, curr_vel=curr_velocity, timestamp=timestamp
        )

        vel_max = curr[ColumnName.VALUE_VEL_MAX]
        ang_max = curr[ColumnName.VALUE_ANG_MAX]
        vel_min = curr[ColumnName.VALUE_VEL_MIN]
        ang_min = curr[ColumnName.VALUE_ANG_MIN]

        assert controller.velocity_max == vel_max
        assert controller.angle_max == ang_max
        assert controller.velocity_min == vel_min
        assert controller.angle_min == ang_min


def test_calculate_vel_ss() -> None:
    """Test the calculation of vel_ss.

    :return: None
    """
    df = pd.read_csv(HighLevelData.DATA_VEL_SS)
    controller = HighLevelController()

    for i in range(0, len(df)):
        curr = df.iloc[i]

        # Arrange
        controller.curr_velocity = curr[ColumnName.VELOCITY]
        controller.velocity_max = curr[ColumnName.VALUE_VEL_MAX]
        controller.velocity_min = curr[ColumnName.VALUE_VEL_MIN]

        # Act
        sum = controller.velocity_max + controller.velocity_min
        gamma_t = -(controller.velocity_max + controller.velocity_min) / 2.0
        vel_ss = controller._calculate_vel_ss()

        # Assert
        expected_sum = curr[ColumnName.VEL_SUM_MINMAX]
        expected_gamma_t = curr[ColumnName.VEL_GAMMA_T]
        expected_vel_ss = curr[ColumnName.VEL_SS]

        # Due to floating-point round-off/precision differences between MATLAB and Python numerical backends, exact equality comparisons seem to be not reliable.
        assert isclose(sum, expected_sum, rel_tol=1e-13)
        assert isclose(gamma_t, expected_gamma_t, rel_tol=1e-13)
        assert isclose(vel_ss, expected_vel_ss, rel_tol=1e-12)


def test_calculate_ang_ss() -> None:
    """Test the calculation of ang_ss.

    :return: None
    """
    df = pd.read_csv(HighLevelData.DATA_ANG_SS)
    controller = HighLevelController()

    for i in range(0, len(df)):
        curr = df.iloc[i]

        # Arrange
        controller.curr_angle = curr[ColumnName.ANGLE]
        controller.angle_max = curr[ColumnName.VALUE_ANG_MAX]
        controller.angle_min = curr[ColumnName.VALUE_ANG_MIN]

        # Act
        gamma_t = -(controller.angle_max + controller.angle_min) / 2.0
        ang_ss = controller._calculate_ang_ss()

        # Assert
        expected_gamma_t = curr[ColumnName.ANG_GAMMA_T]
        expected_ang_ss = curr[ColumnName.ANG_SS]

        # Due to floating-point round-off/precision differences between MATLAB and Python numerical backends, exact equality comparisons seem to be not reliable.
        assert isclose(gamma_t, expected_gamma_t, rel_tol=1e-12)
        assert isclose(ang_ss, expected_ang_ss, rel_tol=1e-11)


def test_z_t_and_pos_ss() -> None:
    """Test z(t) and pos_ss if these values are correctly set through the update method.

    :return: None
    """
    df = pd.read_csv(HighLevelData.DATA_GAIT_PHASE)
    controller = HighLevelController()

    for i in range(0, len(df)):
        curr = df.iloc[i]

        timestamp = curr[ColumnName.TIMESTAMP]
        curr_velocity = curr[ColumnName.VELOCITY]
        curr_angle = curr[ColumnName.ANGLE]

        controller.update(
            curr_angle=curr_angle, curr_vel=curr_velocity, timestamp=timestamp
        )

        expected_z_t = curr[ColumnName.Z_T]
        expected_pos_ss = curr[ColumnName.POS_SS]

        assert isclose(controller.z_t, expected_z_t, rel_tol=1e-12), f"Row {i}"
        assert isclose(controller.pos_ss, expected_pos_ss, rel_tol=1e-8), (
            f"Row {i}, expected_z_t{expected_z_t}, current_z_t{controller.z_t}, "
            f"ang_ss{controller._calculate_ang_ss()}, multiplication{controller.z_t * controller._calculate_ang_ss()}; "
            f"pos_ss{controller.pos_ss}"
        )
