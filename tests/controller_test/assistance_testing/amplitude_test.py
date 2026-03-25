"""Testing the amplitude modulation."""

from math import isclose

from pandas import read_csv

from hip_controller.control.assistance_control.amplitude_modulation import (
    AmplitudeModulation,
)
from hip_controller.definitions import SIGMOID_POWER, SensorSignal
from hip_controller.utils.math_utils import apply_sigmoid_scaling
from tests.conftest import DATA_AMPLITUDE_MODULATION, REL_TOL, KinematicsDataColumnName


def test_sigmoid_scaling() -> None:
    """Test the calculation of cubic spline interpolation.

    :return: None
    """
    df = read_csv(filepath_or_buffer=DATA_AMPLITUDE_MODULATION)

    power = SIGMOID_POWER

    n = len(df)
    for i in range(0, n):
        curr = df.iloc[i]

        # Arrange

        value = curr[KinematicsDataColumnName.SCALED_RADIUS]

        # Act
        sigmoid = apply_sigmoid_scaling(value=value, power=power)

        # Assert
        expected_sigmoid = curr[KinematicsDataColumnName.SIGMOID_RADIUS]

        assert isclose(sigmoid, expected_sigmoid, rel_tol=REL_TOL), f"Row {i}"


def test_amplitude() -> None:
    """Test the calculation of amplitude.

    :return: None
    """
    df = read_csv(filepath_or_buffer=DATA_AMPLITUDE_MODULATION)

    mode = AmplitudeModulation(reverse=False)

    n = len(df)
    for i in range(0, n):
        curr = df.iloc[i]

        # Arrange
        timestamp = curr[KinematicsDataColumnName.TIMESTAMP]
        vel = curr[KinematicsDataColumnName.VELOCITY_LEFT]
        ang = curr[KinematicsDataColumnName.ANGLE_LEFT]

        # Act
        amplitude = mode.compute_amplitude(
            SensorSignal(timestamp=timestamp, angle_rad=ang, velocity_rad_per_sec=vel)
        )

        # Assert
        expected_amplitude = curr[KinematicsDataColumnName.AMPLITUDE_LEFT]

        assert isclose(amplitude, expected_amplitude, rel_tol=REL_TOL), f"Row {i}"
