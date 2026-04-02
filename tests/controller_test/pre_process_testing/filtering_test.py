"""Testing velocity estimation strategies."""

from numpy import testing
from pandas import read_csv

from hip_controller.control.signal_processing.velocity_estimation import (
    DiscreteDerivativeVelocityEstimation,
    SogifllVelocityEstimation,
)
from hip_controller.definitions import PreprocessorConfig
from tests.conftest import DATA_PRE_PROCESSING, KinematicsDataColumnName


def test_filtering_sogifll() -> None:
    """Test Sogifll estimation against expected outputs."""
    # Load test data
    df = read_csv(DATA_PRE_PROCESSING)

    # Configure the notch filter for drift removal
    velocity_estimation = SogifllVelocityEstimation(
        PreprocessorConfig.filtering_sogifll_config
    )

    # Test each row
    prev_timestamp = None
    actual_angles = []
    actual_velocitys = []

    expected_angles = []
    expected_velocitys = []

    for i in range(0, len(df)):
        curr = df.iloc[i]
        timestamp = float(curr[KinematicsDataColumnName.TIMESTAMP])
        raw_angle = float(curr[KinematicsDataColumnName.RAW_ANG_LEFT])

        expected_angle = float(curr[KinematicsDataColumnName.FILTERED_ANG_SOGIFLL])
        expected_velocity = float(curr[KinematicsDataColumnName.FILTERED_VEL_SOGIFLL])

        if prev_timestamp is None:
            prev_timestamp = timestamp
            continue  # Skip first row as no dt available

        dt = timestamp - prev_timestamp
        prev_timestamp = timestamp

        # Filter
        actual_angle, actual_velocity = velocity_estimation.filter(
            angle_rad=raw_angle, time_difference=dt
        )

        actual_angles.append(actual_angle)
        actual_velocitys.append(actual_velocity)

        expected_angles.append(expected_angle)
        expected_velocitys.append(expected_velocity)

    testing.assert_array_almost_equal(actual_angles, expected_angles, decimal=5)
    testing.assert_array_almost_equal(actual_velocitys, expected_velocitys, decimal=5)


def test_filtering_discrete_derivative() -> None:
    """Test discrete estimation against expected outputs."""
    # Load test data
    df = read_csv(DATA_PRE_PROCESSING)

    # Configure the notch filter for drift removal
    velocity_estimation = DiscreteDerivativeVelocityEstimation()

    # Test each row
    prev_timestamp = None

    actual_velocitys = []
    expected_velocitys = []

    for i in range(0, len(df)):
        curr = df.iloc[i]
        timestamp = float(curr[KinematicsDataColumnName.TIMESTAMP])
        raw_angle = float(curr[KinematicsDataColumnName.RAW_ANG_LEFT])

        expected_velocity = float(curr[KinematicsDataColumnName.FILTERED_VEL_DISCRETE])

        if prev_timestamp is None:
            prev_timestamp = timestamp
            continue  # Skip first row as no dt available

        dt = timestamp - prev_timestamp
        prev_timestamp = timestamp

        # Filter
        _, actual_velocity = velocity_estimation.filter(
            angle_rad=raw_angle, time_difference=dt
        )

        actual_velocitys.append(actual_velocity)
        expected_velocitys.append(expected_velocity)

    testing.assert_array_almost_equal(actual_velocitys, expected_velocitys, decimal=5)
