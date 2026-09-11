"""Testing drift removal strategies.

The two recorded-trace tests below pin the filters against a reference
capture. The property tests after them assert what drift removal is *for* --
rejecting a constant offset while passing the motion band -- which holds for
any cut-off and so survives retuning.
"""

import math

import numpy as np
import pandas as pd

from hip_controller.control.signal_processing.drift_removal import (
    DriftRemovalStrategy,
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
    """Test LowPassDriftRemoval against expected outputs."""
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


# --------------------------------------------------------------------------
# Property tests: invariants that survive retuning the cut-off frequency.
# --------------------------------------------------------------------------

SAMPLE_PERIOD_S = 0.01
DC_OFFSET_RAD = 0.35


def test_low_pass_drift_removal_rejects_a_constant_offset() -> None:
    """A constant sensor offset is driven out of the output in steady state.

    This is the purpose of the block: the hip angle the controller sees should
    be centred regardless of how the IMU happens to be mounted or zeroed.
    """
    drift_removal = LowPassDriftRemoval(
        PreprocessorConfig.drift_removal_second_order_lpf_config
    )
    output = 0.0
    for _ in range(6000):  # 60 s, well past the 1.25 rad/s cut-off settling
        output = drift_removal.filter(
            raw_angle=DC_OFFSET_RAD, time_difference=SAMPLE_PERIOD_S
        )
    assert abs(output) < 0.01 * DC_OFFSET_RAD


def test_notch_drift_removal_rejects_a_constant_offset() -> None:
    """The DC notch suppresses a constant offset to a small steady residual.

    Unlike the low-pass high-pass path, the notch has finite rather than
    infinite attenuation at DC, so it settles on a residual of roughly 1% of
    the offset instead of decaying to zero. The bound below sits just above
    that asymptote so the test catches a real loss of rejection without
    pinning the exact figure.
    """
    drift_removal = NotchDriftRemoval(PreprocessorConfig.drift_removal_notch_config)
    output = 0.0
    for _ in range(6000):
        output = drift_removal.filter(
            raw_angle=DC_OFFSET_RAD, time_difference=SAMPLE_PERIOD_S
        )
    assert abs(output) < 0.02 * DC_OFFSET_RAD


def test_low_pass_drift_removal_passes_the_walking_band() -> None:
    """Motion at walking cadence survives drift removal largely intact.

    A high-pass that also ate the gait signal would zero the offset and be
    useless, so the complement of the rejection test matters just as much.
    """
    drift_removal = LowPassDriftRemoval(
        PreprocessorConfig.drift_removal_second_order_lpf_config
    )
    amplitude_rad, frequency_hz = 0.4, 0.9
    outputs = []
    for i in range(6000):
        angle_rad = DC_OFFSET_RAD + amplitude_rad * math.sin(
            2.0 * math.pi * frequency_hz * i * SAMPLE_PERIOD_S
        )
        outputs.append(
            drift_removal.filter(raw_angle=angle_rad, time_difference=SAMPLE_PERIOD_S)
        )

    steady = outputs[3000:]
    assert max(steady) - min(steady) > 1.6 * amplitude_rad  # peak-to-peak retained
    assert abs(sum(steady) / len(steady)) < 0.01 * DC_OFFSET_RAD  # offset gone


def _assert_reset_matches_fresh(
    used: DriftRemovalStrategy, fresh: DriftRemovalStrategy
) -> None:
    """Drive a settled strategy and a fresh one and require identical output.

    :param DriftRemovalStrategy used: Strategy that has been driven then reset.
    :param DriftRemovalStrategy fresh: Newly constructed strategy.
    :return: None
    """
    for i in range(100):
        angle_rad = 0.2 * math.sin(0.1 * i)
        assert used.filter(
            raw_angle=angle_rad, time_difference=SAMPLE_PERIOD_S
        ) == fresh.filter(raw_angle=angle_rad, time_difference=SAMPLE_PERIOD_S)


def test_low_pass_drift_removal_reset_restores_initial_response() -> None:
    """The low-pass strategy returns to its construction state on reset."""
    config = PreprocessorConfig.drift_removal_second_order_lpf_config
    used = LowPassDriftRemoval(config)
    for _ in range(500):
        used.filter(raw_angle=DC_OFFSET_RAD, time_difference=SAMPLE_PERIOD_S)
    used.reset()
    _assert_reset_matches_fresh(used, LowPassDriftRemoval(config))


def test_notch_drift_removal_reset_restores_initial_response() -> None:
    """The notch strategy returns to its construction state on reset."""
    config = PreprocessorConfig.drift_removal_notch_config
    used = NotchDriftRemoval(config)
    for _ in range(500):
        used.filter(raw_angle=DC_OFFSET_RAD, time_difference=SAMPLE_PERIOD_S)
    used.reset()
    _assert_reset_matches_fresh(used, NotchDriftRemoval(config))
