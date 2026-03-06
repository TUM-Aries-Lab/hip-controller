"""Test the main program."""

from math import isclose

from pandas import read_csv

from hip_controller.control.app import (
    AmplitudeModulation,
    SensorSignal,
    WalkOnController,
)
from tests.conftest import REL_TOL, ControllerDataPath, KinematicsDataColumnName


def test_controller_right():
    """Test the main function with the right lower limb data."""
    controller = WalkOnController(reverse=True, plot=False)

    df = read_csv(filepath_or_buffer=ControllerDataPath.DATA_REFERENCE_MOTION_RIGHT)

    n = len(df)
    for i in range(0, n):
        curr = df.iloc[i]

        # Arrange
        timestamp = curr[KinematicsDataColumnName.TIMESTAMP]
        vel = curr[KinematicsDataColumnName.VEL_RIGHT]
        ang = curr[KinematicsDataColumnName.ANG_RIGHT]

        # Act
        motor_reference = controller.step(
            curr_signal=SensorSignal(angle_rad=ang, velocity_rad_per_sec=vel),
            timestamp=timestamp,
        )

        # Assert
        expected_motor_reference = curr[KinematicsDataColumnName.REF_MOT_RIGHT]

        assert isclose(motor_reference, expected_motor_reference, abs_tol=0.08), (
            f"Row {i}"
        )


def test_amplitude_reverse():
    """Test the amplitude function of the right lower limb controller."""
    modulation = AmplitudeModulation(reverse=True)
    df = read_csv(filepath_or_buffer=ControllerDataPath.DATA_REFERENCE_MOTION_RIGHT)

    n = len(df)
    for i in range(0, n):
        curr = df.iloc[i]

        # Arrange
        vel = curr[KinematicsDataColumnName.VEL_RIGHT]
        ang = curr[KinematicsDataColumnName.ANG_RIGHT]

        signal = SensorSignal(angle_rad=ang, velocity_rad_per_sec=vel)
        # Act

        amplitude = modulation.compute_amplitude(signal=signal)

        # Assert

        expected_amplitude = curr[KinematicsDataColumnName.AMPLITUDE_RIGHT]
        assert isclose(amplitude, expected_amplitude, rel_tol=REL_TOL), f"Row {i}"
