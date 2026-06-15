"""Testing drift removal strategies."""

import numpy as np
import pandas as pd

from hip_controller.control.signal_processing.drift_removal import (
    LowPassDriftRemoval,
    NotchDriftRemoval,
)
from hip_controller.definitions import PreprocessorConfig
from tests.conftest import DATA_PRE_PROCESSING, KinematicsDataColumnName


def test_low_pass_drift_removal() -> None:
    """Test LowPassDriftRemoval against expected outputs."""
    # Load test data
    df = pd.read_csv(DATA_PRE_PROCESSING)

    # Configure the low-pass filter for drift removal (slow cutoff)
    drift_removal = LowPassDriftRemoval(
        PreprocessorConfig.drift_removal_second_order_lpf_config
    )

    # Test each row
    prev_timestamp = None
    actual_results = []
    expected_results = []

    for i in range(0, len(df)):
        curr = df.iloc[i]
        timestamp = float(curr[KinematicsDataColumnName.TIMESTAMP])
        raw_angle = float(curr[KinematicsDataColumnName.RAW_ANG_LEFT])
        expected_angle_no_drift = float(curr[KinematicsDataColumnName.NO_DRIFT_ANG_LPF])

        if prev_timestamp is None:
            prev_timestamp = timestamp
            continue  # Skip first row as no dt available

        dt = timestamp - prev_timestamp
        prev_timestamp = timestamp

        # Filter
        actual_angle_no_drift = drift_removal.filter(
            raw_angle=raw_angle, time_difference=dt
        )

        actual_results.append(actual_angle_no_drift)
        expected_results.append(expected_angle_no_drift)

    np.testing.assert_array_almost_equal(actual_results, expected_results, decimal=2)


def test_notch_drift_removal() -> None:
    """Test NotchDriftRemoval against expected outputs."""
    # Load test data
    df = pd.read_csv(DATA_PRE_PROCESSING)

    # Configure the notch filter for drift removal
    drift_removal = NotchDriftRemoval(PreprocessorConfig.drift_removal_notch_config)

    # Test each row
    prev_timestamp = None
    actual_results = []
    expected_results = []

    for i in range(0, len(df)):
        curr = df.iloc[i]
        timestamp = float(curr[KinematicsDataColumnName.TIMESTAMP])
        raw_angle = float(curr[KinematicsDataColumnName.RAW_ANG_LEFT])
        expected_angle_no_drift = float(
            curr[KinematicsDataColumnName.NO_DRIFT_ANG_NOTCH]
        )

        if prev_timestamp is None:
            prev_timestamp = timestamp
            continue  # Skip first row as no dt available

        dt = timestamp - prev_timestamp
        prev_timestamp = timestamp

        # Filter
        actual_angle_no_drift = drift_removal.filter(
            raw_angle=raw_angle, time_difference=dt
        )

        actual_results.append(actual_angle_no_drift)
        expected_results.append(expected_angle_no_drift)

    np.testing.assert_array_almost_equal(actual_results, expected_results, decimal=5)
