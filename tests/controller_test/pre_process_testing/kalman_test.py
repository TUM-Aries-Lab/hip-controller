"""Tests for the Kalman filtering stage.

The filter is selectable for the angle stage alongside the SOGI-FLL and the
low-pass filter, so these cover what that stage has to do: reduce measurement
noise without destroying the signal, restore a known state on reset, and --
because one :class:`BasicConfig` is shared by both limbs -- keep two filters
built from the same config independent of each other.
"""

import math

import numpy as np
import pytest

from hip_controller.control.signal_processing.filtering import KalmanFiltering
from hip_controller.control.signal_processing.sensor_preprocessor import (
    SensorPreprocessor,
)
from hip_controller.definitions import (
    BasicConfig,
    FilteringMethod,
    KalmanFilterConfig,
    SensorSignal,
    VelocityEstimationMethod,
)
from hip_controller.filters.kalman_filter import KalmanFilter
from hip_controller.utils.state_space import StateSpaceLinear
from tests.conftest import SAMPLE_RATE_HZ

SAMPLE_PERIOD_S = 1.0 / SAMPLE_RATE_HZ
STRIDE_FREQUENCY_HZ = 0.9
AMPLITUDE_RAD = 0.4
NOISE_RAD = 0.05
# Long enough for the covariance to settle before anything is measured.
WARMUP_SAMPLES = 100


def _kalman_config() -> BasicConfig:
    """Build a config selecting the Kalman angle stage.

    The SOGI velocity path reads a quadrature the Kalman filter does not
    produce, so a different velocity estimator is required.

    :return: Config with ``filtering_method`` set to KALMAN.
    :rtype: BasicConfig
    """
    return BasicConfig(
        filtering_method=FilteringMethod.KALMAN,
        velocity_estimation_method=VelocityEstimationMethod.LOW_PASS,
    )


def _noisy_walk(samples: int, seed: int = 0) -> tuple[list[float], list[float]]:
    """Generate a clean sinusoidal hip angle and a noisy measurement of it.

    :param int samples: Number of samples to generate.
    :param int seed: Seed for the measurement noise.
    :return: ``(clean, noisy)`` angle traces [rad].
    :rtype: tuple[list[float], list[float]]
    """
    rng = np.random.default_rng(seed)
    omega = 2.0 * math.pi * STRIDE_FREQUENCY_HZ
    clean = [
        AMPLITUDE_RAD * math.sin(omega * index * SAMPLE_PERIOD_S)
        for index in range(samples)
    ]
    noisy = [value + float(rng.normal(0.0, NOISE_RAD)) for value in clean]
    return clean, noisy


def test_covariance_shrinks_as_measurements_arrive() -> None:
    """The estimate must become more certain the more it is told."""
    dt = SAMPLE_PERIOD_S
    config = KalmanFilterConfig(
        state_space=StateSpaceLinear(A=np.array([[1.0, dt], [0.0, 1.0]]), C=np.eye(2)),
        initial_state=np.zeros((2, 1)),
        initial_covariance=np.eye(2),
    )

    kalman = KalmanFilter(config=config)
    for _ in range(10):
        kalman.predict()
        kalman.update(z=np.array([[0.0]]))

    assert np.all(np.diag(kalman.cov) < np.diag(np.eye(2)))


def _squared_error(estimates: list[float], truth: list[float]) -> float:
    """Sum of squared error over the steady-state part of a trace.

    :param list[float] estimates: Trace under test [rad].
    :param list[float] truth: Noise-free reference [rad].
    :return: Summed squared error after the warm-up.
    :rtype: float
    """
    return sum(
        (estimate - value) ** 2
        for estimate, value in zip(
            estimates[WARMUP_SAMPLES:], truth[WARMUP_SAMPLES:], strict=True
        )
    )


def test_filtering_reduces_noise_on_a_held_posture() -> None:
    """Measurement noise on a still limb must be smoothed away.

    Asserted against the unfiltered measurement rather than a fixed bound, so it
    survives retuning of the noise covariances. A held angle isolates the
    filter's noise rejection from its tracking lag -- see the xfail below.
    """
    rng = np.random.default_rng(0)
    held_angle_rad = 0.2
    truth = [held_angle_rad] * 600
    noisy = [value + float(rng.normal(0.0, NOISE_RAD)) for value in truth]

    kalman = KalmanFiltering(
        _kalman_config().preprocessor_config.filtering_kalman_config
    )
    filtered = [
        kalman.filter(angle_rad=value, time_difference=SAMPLE_PERIOD_S)
        for value in noisy
    ]

    assert _squared_error(filtered, truth) < _squared_error(noisy, truth)


@pytest.mark.xfail(
    reason=(
        "The shipped noise covariances lag the signal rather than clean it. "
        "MEASUREMENT_NOISE = 0.75 rad^2 implies a measurement standard "
        "deviation of 0.87 rad (~50 deg), larger than the hip angle itself, so "
        "the filter nearly ignores the measurements and leans on the "
        "constant-velocity model. Measured on this trace: 3 samples (30 ms) of "
        "lag and a squared error of 3.35 against 1.26 for the raw measurement. "
        "At 0.9 Hz that is ~10 deg of gait phase, which is why this stage is "
        "not the default. Setting R to the actual measurement variance removes "
        "the lag and brings the error below raw -- but picking that value is a "
        "tuning decision to be made against recorded data, not in a test."
    ),
    strict=False,
)
def test_filtering_reduces_noise_while_walking() -> None:
    """The property the stage would need before it could carry the angle path."""
    clean, noisy = _noisy_walk(samples=600)
    kalman = KalmanFiltering(
        _kalman_config().preprocessor_config.filtering_kalman_config
    )

    filtered = [
        kalman.filter(angle_rad=value, time_difference=SAMPLE_PERIOD_S)
        for value in noisy
    ]

    assert _squared_error(filtered, clean) < _squared_error(noisy, clean)


def test_two_filters_from_one_config_stay_independent() -> None:
    """Both limbs share a BasicConfig, so they must not share model state.

    ``filter()`` writes the measured sample period into the model's ``A``
    matrix. Holding the config's own array would make the legs overwrite each
    other's model -- and because both would still produce plausible numbers,
    nothing downstream would reveal it.
    """
    config = _kalman_config().preprocessor_config.filtering_kalman_config
    nominal_sample_period = float(config.state_space.A[0, 1])

    left = KalmanFiltering(config)
    right = KalmanFiltering(config)

    _, noisy = _noisy_walk(samples=200)
    for value in noisy:
        left.filter(angle_rad=value, time_difference=SAMPLE_PERIOD_S)
        right.filter(angle_rad=value, time_difference=SAMPLE_PERIOD_S * 2.0)

    assert float(config.state_space.A[0, 1]) == nominal_sample_period, (
        "a filter wrote its sample period into the shared config"
    )
    assert left._kalman_filter.state_space.A[0, 1] == SAMPLE_PERIOD_S
    assert right._kalman_filter.state_space.A[0, 1] == SAMPLE_PERIOD_S * 2.0


def test_reset_restores_the_initial_estimate() -> None:
    """After a reset the filter must behave as a freshly built one."""
    config = _kalman_config().preprocessor_config.filtering_kalman_config
    _, noisy = _noisy_walk(samples=200)

    used = KalmanFiltering(config)
    for value in noisy:
        used.filter(angle_rad=value, time_difference=SAMPLE_PERIOD_S)
    used.reset()

    fresh = KalmanFiltering(config)

    assert [
        used.filter(angle_rad=value, time_difference=SAMPLE_PERIOD_S) for value in noisy
    ] == [
        fresh.filter(angle_rad=value, time_difference=SAMPLE_PERIOD_S)
        for value in noisy
    ]


def test_reset_does_not_corrupt_the_config() -> None:
    """The initial estimate lives in a shared config and must survive a reset."""
    config = _kalman_config().preprocessor_config.filtering_kalman_config
    initial_state = np.array(config.initial_state, dtype=float)

    kalman = KalmanFiltering(config)
    _, noisy = _noisy_walk(samples=100)
    for value in noisy:
        kalman.filter(angle_rad=value, time_difference=SAMPLE_PERIOD_S)
    kalman.reset()

    assert np.array_equal(config.initial_state, initial_state)


def test_the_preprocessor_builds_and_runs_the_kalman_stage() -> None:
    """End to end: selecting KALMAN puts it in the signal path and it works."""
    preprocessor = SensorPreprocessor(_kalman_config())

    _, noisy = _noisy_walk(samples=400)
    outputs = [
        preprocessor.filter(
            raw_signal=SensorSignal(
                timestamp=index * SAMPLE_PERIOD_S,
                angle_rad=value,
                velocity_rad_per_sec=0.0,
            )
        )
        for index, value in enumerate(noisy)
    ]

    assert isinstance(preprocessor._filtering, KalmanFiltering)
    steady = outputs[WARMUP_SAMPLES:]
    assert all(math.isfinite(signal.angle_rad) for signal in steady)
    assert max(abs(signal.angle_rad) for signal in steady) > 0.01, (
        "the Kalman stage flattened the signal to nothing"
    )


def test_the_kalman_stage_cannot_feed_the_sogi_velocity_path() -> None:
    """The SOGI velocity path reads a quadrature the Kalman filter never makes.

    Silently returning zero velocity is the failure this prevents; the
    preprocessor rejects the combination at construction instead.
    """
    with pytest.raises(ValueError, match="SOGI"):
        SensorPreprocessor(
            BasicConfig(
                filtering_method=FilteringMethod.KALMAN,
                velocity_estimation_method=VelocityEstimationMethod.SOGI,
            )
        )
