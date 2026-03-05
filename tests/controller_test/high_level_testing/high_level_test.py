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

# Due to floating-point round-off/precision differences between MATLAB and Python numerical backends, exact equality comparisons seem to be not reliable.

from math import isclose

from pandas import read_csv

from hip_controller.control.high_level_controller.high_level import (
    HighLevelController,
    SensorSignal,
)
from tests.conftest import REL_TOL, HighLevelData, KinematicsDataColumnName


def test_extrema_values() -> None:
    """Test angle and velocity extrema are updated correctly each step based on the given timestamp angle and velocity.

    :return: None
    """
    df = read_csv(filepath_or_buffer=HighLevelData.DATA_EXTREMA_VALUES)
    controller = HighLevelController()

    for i in range(0, len(df)):
        curr = df.iloc[i]

        # Arrange
        timestamp = curr[KinematicsDataColumnName.TIMESTAMP]
        curr_velocity = curr[KinematicsDataColumnName.VELOCITY]
        curr_angle = curr[KinematicsDataColumnName.ANGLE]

        # Act
        controller.update_and_compute(
            curr_signal=SensorSignal(
                angle_rad=curr_angle, velocity_rad_per_sec=curr_velocity
            ),
            timestamp=timestamp,
        )

        # Assert
        vel_max = curr[KinematicsDataColumnName.VALUE_VEL_MAX]
        ang_max = curr[KinematicsDataColumnName.VALUE_ANG_MAX]
        vel_min = curr[KinematicsDataColumnName.VALUE_VEL_MIN]
        ang_min = curr[KinematicsDataColumnName.VALUE_ANG_MIN]

        assert controller.steady_state_tracker._velocity_max == vel_max
        assert controller.steady_state_tracker._angle_max == ang_max
        assert controller.steady_state_tracker._velocity_min == vel_min
        assert controller.steady_state_tracker._angle_min == ang_min


def test_high_level() -> None:
    """Test all the functions combined with compute_and_update function.

    :return: None
    """
    df = read_csv(filepath_or_buffer=HighLevelData.DATA_HIGH_LEVEL)
    controller = HighLevelController()

    for i in range(0, len(df)):
        curr = df.iloc[i]

        # Arrange
        timestamp = curr[KinematicsDataColumnName.TIMESTAMP]
        curr_velocity = curr[KinematicsDataColumnName.VELOCITY]
        curr_angle = curr[KinematicsDataColumnName.ANGLE]

        # Act
        gait_phase = controller.update_and_compute(
            curr_signal=SensorSignal(
                angle_rad=curr_angle, velocity_rad_per_sec=curr_velocity
            ),
            timestamp=timestamp,
        )

        signal = controller.get_signal_steady_state()

        # boolean initialization
        expected_not_initialized = curr[KinematicsDataColumnName.NOT_INITIALIZED]

        # center_vel, center_ang, rescale_factor in steadyStateDetector which should be updated with the correct timing through stride event detector
        expected_center_vel = curr[KinematicsDataColumnName.CENTER_VEL]
        expected_center_ang = curr[KinematicsDataColumnName.CENTER_ANG]
        expected_rescale_factor = curr[KinematicsDataColumnName.RESCALE_FACTOR]

        # steady state of velocity and angle
        expected_vel_steady_state = curr[KinematicsDataColumnName.VEL_STEADY_STATE]
        expected_ang_steady_state = curr[KinematicsDataColumnName.ANG_STEADY_STATE]

        # gait phase value
        expected_gait_phase = curr[KinematicsDataColumnName.GAIT_PHASE]

        # Assert
        assert (not controller.initialized) == expected_not_initialized, f"Row {i}"

        assert isclose(
            controller.steady_state_tracker._center_vel,
            expected_center_vel,
            rel_tol=REL_TOL,
        ), f"Row {i}"
        assert isclose(
            controller.steady_state_tracker._center_ang,
            expected_center_ang,
            rel_tol=REL_TOL,
        ), f"Row {i}"
        assert isclose(
            controller.steady_state_tracker._rescale_factor,
            expected_rescale_factor,
            rel_tol=REL_TOL,
        ), f"Row {i}"

        assert isclose(
            signal.velocity_rad_per_sec, expected_vel_steady_state, rel_tol=REL_TOL
        ), f"Row {i}"
        assert isclose(signal.angle_rad, expected_ang_steady_state, rel_tol=REL_TOL), (
            f"Row {i}"
        )

        assert isclose(gait_phase, expected_gait_phase, rel_tol=REL_TOL), f"Row {i}"
