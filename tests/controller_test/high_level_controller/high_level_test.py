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

import math
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
from tests.conftest import CSVColumnName, HighLevelData


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
    df = pd.read_csv(filepath_or_buffer=HighLevelData.DATA_ZERO_CROSSING)

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        curr_velocity = curr[CSVColumnName.VELOCITY]
        prev_velocity = prev[CSVColumnName.VELOCITY]
        curr_angle = curr[CSVColumnName.ANGLE]
        prev_angle = prev[CSVColumnName.ANGLE]

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

        expected_vel_max = curr[CSVColumnName.TRIGG_VEL_MAX]
        expected_ang_max = curr[CSVColumnName.TRIGG_ANG_MAX]
        expected_vel_min = curr[CSVColumnName.TRIGG_VEL_MIN]
        expected_ang_min = curr[CSVColumnName.TRIGG_ANG_MIN]

        assert velocity_max == expected_vel_max, f"Row {i}"
        assert angle_max == expected_ang_max, f"Row {i}"
        assert velocity_min == expected_vel_min, f"Row {i}"
        assert angle_min == expected_ang_min, f"Row {i}"


def test_valid_trigger() -> None:
    """Test angle extrema detection based on velocity zero-crossings.

    :return: None
    """
    df = pd.read_csv(filepath_or_buffer=HighLevelData.DATA_VALID_TRIGGER)
    controller = HighLevelController()

    for i in range(0, len(df)):
        curr = df.iloc[i]

        timestamp = curr[CSVColumnName.TIMESTAMP]
        curr_velocity = curr[CSVColumnName.VELOCITY]
        curr_angle = curr[CSVColumnName.ANGLE]

        controller.update(
            curr_angle=curr_angle, curr_vel=curr_velocity, timestamp=timestamp
        )

        vel_max = curr[CSVColumnName.VALID_TRIGG_VEL_MAX]
        ang_max = curr[CSVColumnName.VALID_TRIGG_ANG_MAX]
        vel_min = curr[CSVColumnName.VALID_TRIGG_VEL_MIN]
        ang_min = curr[CSVColumnName.VALID_TRIGG_ANG_MIN]

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
    df = pd.read_csv(filepath_or_buffer=HighLevelData.DATA_EXTREMA_VALUES)
    controller = HighLevelController()

    for i in range(0, len(df)):
        curr = df.iloc[i]

        timestamp = curr[CSVColumnName.TIMESTAMP]
        curr_velocity = curr[CSVColumnName.VELOCITY]
        curr_angle = curr[CSVColumnName.ANGLE]

        controller.update(
            curr_angle=curr_angle, curr_vel=curr_velocity, timestamp=timestamp
        )

        vel_max = curr[CSVColumnName.VALUE_VEL_MAX]
        ang_max = curr[CSVColumnName.VALUE_ANG_MAX]
        vel_min = curr[CSVColumnName.VALUE_VEL_MIN]
        ang_min = curr[CSVColumnName.VALUE_ANG_MIN]

        assert controller.steady_state_tracker.velocity_max == vel_max
        assert controller.steady_state_tracker.angle_max == ang_max
        assert controller.steady_state_tracker.velocity_min == vel_min
        assert controller.steady_state_tracker.angle_min == ang_min


def test_calculate_vel_ss() -> None:
    """Test the calculation of vel_ss.

    :return: None
    """
    df = pd.read_csv(filepath_or_buffer=HighLevelData.DATA_VEL_SS)
    controller = HighLevelController()

    for i in range(0, len(df)):
        curr = df.iloc[i]

        # Arrange
        controller.curr_velocity = curr[CSVColumnName.VELOCITY]
        controller.steady_state_tracker.velocity_max = curr[CSVColumnName.VALUE_VEL_MAX]
        controller.steady_state_tracker.velocity_min = curr[CSVColumnName.VALUE_VEL_MIN]

        # Act
        sum = (
            controller.steady_state_tracker.velocity_max
            + controller.steady_state_tracker.velocity_min
        )
        gamma_t = (
            -(
                controller.steady_state_tracker.velocity_max
                + controller.steady_state_tracker.velocity_min
            )
            / 2.0
        )
        vel_ss = controller.steady_state_tracker._calculate_vel_ss(
            curr_velocity=controller.curr_velocity
        )

        # Assert
        expected_sum = curr[CSVColumnName.VEL_SUM_MINMAX]
        expected_gamma_t = curr[CSVColumnName.VEL_GAMMA_T]
        expected_vel_ss = curr[CSVColumnName.VEL_STEADY_STATE]

        # Due to floating-point round-off/precision differences between MATLAB and Python numerical backends, exact equality comparisons seem to be not reliable.
        assert isclose(sum, expected_sum, rel_tol=1e-13)
        assert isclose(gamma_t, expected_gamma_t, rel_tol=1e-13)
        assert isclose(vel_ss, expected_vel_ss, rel_tol=1e-12)


def test_calculate_ang_ss() -> None:
    """Test the calculation of ang_ss.

    :return: None
    """
    df = pd.read_csv(filepath_or_buffer=HighLevelData.DATA_ANG_SS)
    controller = HighLevelController()

    for i in range(0, len(df)):
        curr = df.iloc[i]

        # Arrange
        controller.curr_angle = curr[CSVColumnName.ANGLE]
        controller.steady_state_tracker.angle_max = curr[CSVColumnName.VALUE_ANG_MAX]
        controller.steady_state_tracker.angle_min = curr[CSVColumnName.VALUE_ANG_MIN]

        # Act
        gamma_t = (
            -(
                controller.steady_state_tracker.angle_max
                + controller.steady_state_tracker.angle_min
            )
            / 2.0
        )
        ang_ss = controller.steady_state_tracker._calculate_ang_ss(
            curr_angle=controller.curr_angle
        )

        # Assert
        expected_gamma_t = curr[CSVColumnName.ANG_GAMMA_T]
        expected_ang_ss = curr[CSVColumnName.ANG_STEADY_STATE]

        # Due to floating-point round-off/precision differences between MATLAB and Python numerical backends, exact equality comparisons seem to be not reliable.
        assert isclose(gamma_t, expected_gamma_t, rel_tol=1e-12)
        assert isclose(ang_ss, expected_ang_ss, rel_tol=1e-11)


def test_z_t_and_pos_ss() -> None:
    """Test z(t) and pos_ss if these values are correctly set through the update method.

    :return: None
    """
    df = pd.read_csv(filepath_or_buffer=HighLevelData.DATA_GAIT_PHASE)
    controller = HighLevelController()

    for i in range(0, len(df)):
        curr = df.iloc[i]

        # Arrange
        timestamp = curr[CSVColumnName.TIMESTAMP]
        curr_velocity = curr[CSVColumnName.VELOCITY]
        curr_angle = curr[CSVColumnName.ANGLE]

        # Act
        controller.update(
            curr_angle=curr_angle, curr_vel=curr_velocity, timestamp=timestamp
        )

        # Assert
        expected_z_t = curr[CSVColumnName.Z_T]
        expected_pos_ss = curr[CSVColumnName.POSTION_STEADY_STATE]

        assert isclose(
            controller.steady_state_tracker.z_t, expected_z_t, rel_tol=1e-12
        ), f"Row {i}"
        assert isclose(
            controller.steady_state_tracker.pos_steady_state,
            expected_pos_ss,
            rel_tol=1e-11,
        ), (
            f"Row {i}, expected_z_t{expected_z_t}, current_z_t{controller.steady_state_tracker.z_t}, "
            f"ang_ss{controller.steady_state_tracker._calculate_ang_ss(controller.curr_angle)}, multiplication{controller.steady_state_tracker.z_t * controller.steady_state_tracker._calculate_ang_ss(curr_angle=controller.curr_angle)}; "
            f"pos_ss{controller.steady_state_tracker.pos_steady_state}"
        )


def test_gait_phase() -> None:
    """Test the calculation of gait phase.

    :return: None
    """
    df = pd.read_csv(filepath_or_buffer=HighLevelData.DATA_GAIT_PHASE)
    controller = HighLevelController()

    for i in range(0, len(df)):
        curr = df.iloc[i]

        # Arrange
        controller.steady_state_tracker.vel_steady_state = curr[
            CSVColumnName.VEL_STEADY_STATE
        ]
        controller.steady_state_tracker.z_t = curr[CSVColumnName.Z_T]
        controller.steady_state_tracker.pos_steady_state = curr[
            CSVColumnName.POSTION_STEADY_STATE
        ]

        # Act
        gait_phase = controller.steady_state_tracker.calculate_gait_phase()

        # Assert
        expected_gait_phase = curr[CSVColumnName.GAIT_PHASE]

        assert isclose(gait_phase, expected_gait_phase, rel_tol=1e-12), (
            f"Row {i}, current_z_t{controller.steady_state_tracker.z_t}, "
            f"calculated_gait_phase{math.atan2(controller.steady_state_tracker.vel_steady_state, -controller.steady_state_tracker.pos_steady_state)}, "
        )
