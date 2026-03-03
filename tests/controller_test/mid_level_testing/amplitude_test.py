"""Testing the amplitude modulation."""

from math import isclose

from pandas import read_csv

from hip_controller.control.mid_level_controller.amplitude_modulation import (
    AmplitudeModulation,
)
from hip_controller.definitions import SIGMOID_POWER, SensorSignal
from tests.conftest import KinematicsDataColumnName, MidLevelData


def test_sigmoid_scaling() -> None:
    """Test the calculation of cubic spline interpolation.

    :return: None
    """
    df = read_csv(filepath_or_buffer=MidLevelData.DATA_AMPLITUDE_MODULATION)

    mode = AmplitudeModulation()
    power = SIGMOID_POWER

    n = len(df)
    for i in range(0, n):
        curr = df.iloc[i]

        # Arrange

        value = curr[KinematicsDataColumnName.SCALED_RADIUS]

        # Act
        sigmoid = mode.apply_sigmoid_scaling(value=value, power=power)

        # Assert
        expected_sigmoid = curr[KinematicsDataColumnName.SIGMOID_RADIUS]

        assert isclose(sigmoid, expected_sigmoid, rel_tol=1e-11), f"Row {i}"


def test_amplitude() -> None:
    """Test the calculation of amplitude.

    :return: None
    """
    df = read_csv(filepath_or_buffer=MidLevelData.DATA_AMPLITUDE_MODULATION)

    mode = AmplitudeModulation()

    n = len(df)
    for i in range(0, n):
        curr = df.iloc[i]

        # Arrange

        vel = curr[KinematicsDataColumnName.VELOCITY]
        ang = curr[KinematicsDataColumnName.ANGLE]

        # Act
        amplitude = mode.compute_amplitude(
            SensorSignal(angle_rad=ang, velocity_rad_per_sec=vel)
        )

        # Assert
        expected_amplitude = curr[KinematicsDataColumnName.AMPLITUDE]

        assert isclose(amplitude, expected_amplitude, rel_tol=1e-11), f"Row {i}"
