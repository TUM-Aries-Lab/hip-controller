"""Tests for motion mapping cubic spline interpolation functions."""

import numpy as np
from pandas import read_csv

from hip_controller.control.mid_level_controller.cubic_spline_interpolation import (
    CubicSplineInterpolation,
)
from tests.conftest import KinematicsDataColumnName, MidLevelData


def test_cubic_spline_interpolation() -> None:
    """Test the calculation of cubic spline interpolation.

    :return: None
    """
    df = read_csv(filepath_or_buffer=MidLevelData.DATA_MOTION_MAPPING)
    lookup = CubicSplineInterpolation()

    values = []
    expected_values = []

    n = len(df)
    for i in range(0, n):
        curr = df.iloc[i]

        # Arrange
        key = curr[KinematicsDataColumnName.MAPPING_KEY]

        # Act
        value = lookup.step(key)
        values.append(value)

        # Assert
        expected_value = curr[KinematicsDataColumnName.MAPPING_VALUE]
        expected_values.append(expected_value)

    np.testing.assert_array_almost_equal(values, expected_values, decimal=3)
