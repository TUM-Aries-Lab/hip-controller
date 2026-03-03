"""Tests for mid-level control functions."""

from math import isclose

from pandas import read_csv

from hip_controller.control.mid_level_controller.mid_level import (
    center_and_transform_gait_phase,
)
from tests.conftest import KinematicsDataColumnName, MidLevelData


def test_transform_gait_phase() -> None:
    """Test the calculation of sinusoidal behavior.

    :return: None
    """
    df = read_csv(filepath_or_buffer=MidLevelData.DATA_SINUSOIDAL_BEHAVIOR)

    for i in range(0, len(df)):
        curr = df.iloc[i]

        # Arrange
        gait_phase = curr[KinematicsDataColumnName.GAIT_PHASE]

        # Act
        sinusoidal_behavior = center_and_transform_gait_phase(gait_phase=gait_phase)

        # Assert
        expected_sinusoidal_behavior = curr[
            KinematicsDataColumnName.SINUSOIDAL_BEHAVIOR
        ]

        assert isclose(
            sinusoidal_behavior, expected_sinusoidal_behavior, rel_tol=1e-12
        ), f"Row {i}"
