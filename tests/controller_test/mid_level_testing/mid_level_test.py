"""Tests for mid-level control functions."""

from math import isclose
from unittest.mock import patch

import numpy as np
from pandas import read_csv

from hip_controller.control.mid_level_controller.mid_level import (
    MidLevelController,
    MotionMapping,
    transform_to_cyclic,
)
from tests.conftest import REL_TOL, ControllerDataPath, KinematicsDataColumnName


def test_transform_gait_phase() -> None:
    """Test the calculation of sinusoidal behavior.

    :return: None
    """
    df = read_csv(filepath_or_buffer=ControllerDataPath.DATA_REFERENCE_MOTION_LEFT)

    for i in range(0, len(df)):
        curr = df.iloc[i]

        # Arrange
        gait_phase = curr[KinematicsDataColumnName.GAIT_PHASE_LEFT]
        # Act
        sinusoidal_behavior = transform_to_cyclic(val=gait_phase)

        # Assert
        expected_sinusoidal_behavior = curr[KinematicsDataColumnName.SIN_WAVE_LEFT]

        assert isclose(
            sinusoidal_behavior, expected_sinusoidal_behavior, rel_tol=REL_TOL
        ), f"Row {i}"


def test_cubic_spline_interpolation() -> None:
    """Test the calculation of cubic spline interpolation.

    :return: None
    """
    df = read_csv(filepath_or_buffer=ControllerDataPath.DATA_REFERENCE_MOTION_LEFT)
    lookup = MotionMapping()

    values = []
    expected_values = []

    n = len(df)
    for i in range(0, n):
        curr = df.iloc[i]

        # Arrange
        key = curr[KinematicsDataColumnName.MAPPING_KEY]

        # Act
        value = lookup.spline(key)
        values.append(value)

        # Assert
        expected_value = curr[KinematicsDataColumnName.MAPPING_VALUE]
        expected_values.append(expected_value)

    # The interpolation works slightly different with matlab simulink but the shape of the curve is the same
    np.testing.assert_array_almost_equal(values, expected_values, decimal=3)


def test_motor_command() -> None:
    """Test the calculation of reference motion.

    :return: None
    """
    df = read_csv(filepath_or_buffer=ControllerDataPath.DATA_REFERENCE_MOTION_LEFT)
    controller = MidLevelController()

    for i in range(0, len(df)):
        curr = df.iloc[i]

        # Arrange
        gait_phase = curr[KinematicsDataColumnName.GAIT_PHASE_LEFT]
        amplitude = curr[KinematicsDataColumnName.AMPLITUDE_LEFT]
        mapping_value = curr[KinematicsDataColumnName.MAPPING_VALUE]

        # Act
        with patch.object(
            controller.motion_mapping, "spline", return_value=mapping_value
        ):
            reference_motion = controller.compute_motor_command(
                gait_phase=gait_phase, amplitude=amplitude
            )

        # Assert
        expected_reference_motion = curr[KinematicsDataColumnName.REF_MOT_LEFT]

        assert isclose(reference_motion, expected_reference_motion, rel_tol=REL_TOL), (
            f"Row {i}, \n mapping_value {mapping_value}, \n amplitude {amplitude}, \n reference motion {reference_motion}, \n expected {expected_reference_motion}"
        )
