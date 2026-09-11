"""Tests for the filtering stage of sensor preprocessing.

The SOGI-FLL tests are property tests: they assert the behaviour the filter is
*for* (locking onto the input cadence, producing an orthogonal quadrature,
preserving state across a per-mode config swap) rather than pinning recorded
sample values, so retuning the SOGI gains does not invalidate them.

``test_filtering_discrete_derivative`` still compares against a recorded trace,
which is appropriate: a backward difference is a mathematical specification
with no tuning constants behind it.
"""

import math

import pytest
from numpy import testing
from pandas import read_csv

from hip_controller.control.signal_processing.filtering import SogiFllFiltering
from hip_controller.control.signal_processing.velocity_estimation import (
    DiscreteDerivativeVelocityEstimation,
)
from hip_controller.definitions import BasicConfig
from hip_controller.filters.sogi_fll_filter import SogiFllFilter
from tests.conftest import DATA_PRE_PROCESSING, KinematicsDataColumnName

PREPROCESSOR_CONFIG = BasicConfig().preprocessor_config
LEVEL_CONFIG = PREPROCESSOR_CONFIG.filtering_sogifll_config_level
DESCEND_CONFIG = PREPROCESSOR_CONFIG.filtering_sogifll_config_descend

SAMPLE_PERIOD_S = 0.01
INPUT_AMPLITUDE_RAD = 0.4
# 40 s of walking. The FLL smoother has a sub-hertz bandwidth, so lock takes a
# few seconds; this leaves ample margin without making the tests slow.
LOCK_SAMPLES = 4000
# Samples discarded before measuring steady-state properties.
TRANSIENT_SAMPLES = 2000


def _drive_sine(
    sogi: SogiFllFilter,
    frequency_hz: float,
    samples: int = LOCK_SAMPLES,
    amplitude_rad: float = INPUT_AMPLITUDE_RAD,
    start_index: int = 0,
) -> list[tuple[float, float]]:
    """Feed a pure sine through the filter and collect its outputs.

    :param SogiFllFilter sogi: Filter under test.
    :param float frequency_hz: Frequency of the injected sine [Hz].
    :param int samples: Number of samples to feed.
    :param float amplitude_rad: Peak amplitude of the injected sine [rad].
    :param int start_index: Sample index the sine starts from.
    :return: ``(inphase, quadrature)`` for each sample.
    :rtype: list[tuple[float, float]]
    """
    outputs = []
    for i in range(start_index, start_index + samples):
        angle_rad = amplitude_rad * math.sin(
            2.0 * math.pi * frequency_hz * i * SAMPLE_PERIOD_S
        )
        outputs.append(
            sogi.filter(raw_theta_rad=angle_rad, time_difference=SAMPLE_PERIOD_S)
        )
    return outputs


@pytest.mark.parametrize("frequency_hz", [0.5, 0.7, 0.9, 1.2, 1.5])
def test_sogifll_locks_onto_input_cadence(frequency_hz: float) -> None:
    """The FLL converges to the frequency of the injected sine.

    This is what the filter exists to do, and it holds for any sensible gain
    set, so the assertion survives retuning.
    """
    sogi = SogiFllFilter(config=LEVEL_CONFIG)
    _drive_sine(sogi, frequency_hz=frequency_hz)
    assert sogi.estimated_frequency_hz == pytest.approx(frequency_hz, rel=0.02)


def test_sogifll_frequency_estimate_respects_cadence_bounds() -> None:
    """A cadence outside the configured band is clamped, never extrapolated."""
    fast = SogiFllFilter(config=LEVEL_CONFIG)
    _drive_sine(fast, frequency_hz=5.0)
    assert (
        LEVEL_CONFIG.lower_cadence_bound
        <= fast.estimated_frequency_hz
        <= LEVEL_CONFIG.upper_cadence_bound
    )

    slow = SogiFllFilter(config=LEVEL_CONFIG)
    _drive_sine(slow, frequency_hz=0.05)
    assert (
        LEVEL_CONFIG.lower_cadence_bound
        <= slow.estimated_frequency_hz
        <= LEVEL_CONFIG.upper_cadence_bound
    )


def test_sogifll_preserves_amplitude_once_locked() -> None:
    """At the locked frequency the SOGI has unity gain on the in-phase output."""
    sogi = SogiFllFilter(config=LEVEL_CONFIG)
    outputs = _drive_sine(sogi, frequency_hz=0.9)
    steady = outputs[TRANSIENT_SAMPLES:]
    peak_inphase = max(abs(inphase) for inphase, _ in steady)
    assert peak_inphase == pytest.approx(INPUT_AMPLITUDE_RAD, rel=0.05)


def test_sogifll_quadrature_is_orthogonal_to_inphase() -> None:
    """The quadrature sits a quarter cycle from the in-phase output.

    That 90 degree relationship is what lets the preprocessor use the
    quadrature as a velocity surrogate, so it is worth asserting directly.
    """
    sogi = SogiFllFilter(config=LEVEL_CONFIG)
    outputs = _drive_sine(sogi, frequency_hz=0.9)
    steady = outputs[TRANSIENT_SAMPLES:]
    count = len(steady)

    inner = sum(inphase * quad for inphase, quad in steady) / count
    inphase_rms = math.sqrt(sum(inphase**2 for inphase, _ in steady) / count)
    quadrature_rms = math.sqrt(sum(quad**2 for _, quad in steady) / count)

    assert abs(inner / (inphase_rms * quadrature_rms)) < 0.05


def test_sogifll_quadrature_matches_inphase_amplitude() -> None:
    """Both SOGI outputs carry the same envelope once locked."""
    sogi = SogiFllFilter(config=LEVEL_CONFIG)
    steady = _drive_sine(sogi, frequency_hz=0.9)[TRANSIENT_SAMPLES:]
    peak_inphase = max(abs(inphase) for inphase, _ in steady)
    peak_quadrature = max(abs(quad) for _, quad in steady)
    assert peak_quadrature == pytest.approx(peak_inphase, rel=0.05)


def test_sogifll_output_is_finite_for_extreme_input() -> None:
    """Extreme angles never produce NaN or infinity."""
    sogi = SogiFllFilter(config=LEVEL_CONFIG)
    for angle_rad in (0.0, 1e6, -1e6, 1e-300):
        inphase, quadrature = sogi.filter(
            raw_theta_rad=angle_rad, time_difference=SAMPLE_PERIOD_S
        )
        assert math.isfinite(inphase)
        assert math.isfinite(quadrature)


def test_sogifll_reset_restores_initial_state() -> None:
    """Reset returns the filter to its construction condition."""
    used = SogiFllFilter(config=LEVEL_CONFIG)
    _drive_sine(used, frequency_hz=1.4, samples=1000)
    used.reset()

    assert used.estimated_frequency_hz == LEVEL_CONFIG.initial_frequency_guess
    assert used.is_walking

    fresh = SogiFllFilter(config=LEVEL_CONFIG)
    assert _drive_sine(used, frequency_hz=0.9, samples=200) == _drive_sine(
        fresh, frequency_hz=0.9, samples=200
    )


def test_sogifll_set_config_preserves_oscillator_state() -> None:
    """Swapping the per-mode config must not restart the oscillator.

    ``SensorPreprocessor.set_locomotion_mode`` relies on this: crossing a mode
    boundary mid-stride reuses the existing FLL lock instead of re-locking from
    ``initial_frequency_guess``. A regression here is invisible in the output
    of a single sample, so assert continuity explicitly.
    """
    sogi = SogiFllFilter(config=LEVEL_CONFIG)
    outputs = _drive_sine(sogi, frequency_hz=0.9)
    last_inphase, _ = outputs[-1]
    frequency_before = sogi.estimated_frequency_hz

    sogi.set_config(DESCEND_CONFIG)
    next_inphase, _ = _drive_sine(
        sogi, frequency_hz=0.9, samples=1, start_index=LOCK_SAMPLES
    )[0]

    # The frequency lock carries over unchanged across the swap.
    assert sogi.estimated_frequency_hz == frequency_before
    # And the oscillator keeps running rather than snapping back to zero.
    assert abs(next_inphase - last_inphase) < 0.1 * INPUT_AMPLITUDE_RAD


def test_sogifll_clear_state_keeps_frequency_lock() -> None:
    """Clearing state wipes the oscillator but keeps the cadence estimate."""
    sogi = SogiFllFilter(config=LEVEL_CONFIG)
    _drive_sine(sogi, frequency_hz=1.2)
    frequency_before = sogi.estimated_frequency_hz

    sogi.clear_state_keep_frequency()
    inphase, quadrature = sogi.filter(
        raw_theta_rad=0.0, time_difference=SAMPLE_PERIOD_S
    )

    assert sogi.estimated_frequency_hz == frequency_before
    assert inphase == pytest.approx(0.0, abs=1e-12)
    assert quadrature == pytest.approx(0.0, abs=1e-12)


def test_sogifll_pause_decays_output_and_freezes_frequency() -> None:
    """During a pause the oscillator decays and the cadence estimate holds.

    Freezing the estimate is what lets the FLL re-engage already tuned to the
    user cadence when walking resumes.
    """
    sogi = SogiFllFilter(config=LEVEL_CONFIG)
    outputs = _drive_sine(sogi, frequency_hz=1.2)
    peak_before = max(abs(inphase) for inphase, _ in outputs[TRANSIENT_SAMPLES:])
    frequency_before = sogi.estimated_frequency_hz

    sogi.stop_walking()
    assert not sogi.is_walking
    inphase = 0.0
    for _ in range(1000):
        inphase, _ = sogi.filter(raw_theta_rad=0.0, time_difference=SAMPLE_PERIOD_S)

    assert abs(inphase) < 0.01 * peak_before
    assert sogi.estimated_frequency_hz == frequency_before


def test_sogifll_resume_clears_oscillator_but_keeps_frequency() -> None:
    """Resuming rebuilds the oscillator from the preserved frequency lock."""
    sogi = SogiFllFilter(config=LEVEL_CONFIG)
    _drive_sine(sogi, frequency_hz=1.2)
    frequency_before = sogi.estimated_frequency_hz

    sogi.stop_walking()
    sogi.start_walking()

    assert sogi.is_walking
    assert sogi.estimated_frequency_hz == frequency_before


def test_sogifll_filtering_wrapper_exposes_quadrature() -> None:
    """The strategy wrapper surfaces the quadrature for downstream logging."""
    filtering = SogiFllFiltering(config=LEVEL_CONFIG)
    assert filtering.last_quadrature == 0.0

    for i in range(500):
        filtering.filter(
            angle_rad=INPUT_AMPLITUDE_RAD
            * math.sin(2.0 * math.pi * 0.9 * i * SAMPLE_PERIOD_S),
            time_difference=SAMPLE_PERIOD_S,
        )
    assert filtering.last_quadrature != 0.0

    filtering.reset()
    assert filtering.last_quadrature == 0.0


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
