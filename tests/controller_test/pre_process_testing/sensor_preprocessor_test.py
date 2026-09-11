"""Property tests for the sensor preprocessing pipeline as a whole.

The individual stages (drift removal, SOGI-FLL filtering, velocity estimation)
have their own tests. This module covers the thing those cannot see: that
:class:`SensorPreprocessor` actually wires them together, and that the runtime
mode switches reach the filter instance that is really in the signal path.

That last point is not hypothetical. A previous attempt to refactor this class
ended up constructing two ``SogiFllFiltering`` objects -- one used to filter the
angle, a second one that ``set_locomotion_mode`` / ``set_demo_mode`` configured
and which was never stepped. Per-mode retuning became a silent no-op and, on the
SOGI velocity path, the output velocity was identically zero. Every test in the
suite still passed, because the end-to-end tests only assert that the command is
finite and inside the motor limits, which zero velocity does not violate. The
tests here are written to fail loudly in that situation.
"""

import math

import pytest

from hip_controller.control.signal_processing.sensor_preprocessor import (
    SensorPreprocessor,
)
from hip_controller.definitions import BasicConfig, SensorSignal
from tests.conftest import SAMPLE_RATE_HZ, synthetic_gait

# Long enough for the SOGI-FLL to lock and the drift-removal LPF to settle.
WARMUP_S = 20.0
# Fraction of the run discarded before measuring steady-state behaviour.
TRANSIENT_FRACTION = 0.5


def _run(
    preprocessor: SensorPreprocessor, signals: list[SensorSignal]
) -> list[SensorSignal]:
    """Feed a trace through the preprocessor and collect its outputs.

    :param SensorPreprocessor preprocessor: Instance under test.
    :param list[SensorSignal] signals: Trace to feed, sample by sample.
    :return: Preprocessed signal for each input sample.
    :rtype: list[SensorSignal]
    """
    return [preprocessor.filter(raw_signal=signal) for signal in signals]


def _steady(outputs: list[SensorSignal]) -> list[SensorSignal]:
    """Drop the start-up transient from a collected trace.

    :param list[SensorSignal] outputs: Full collected trace.
    :return: The tail of the trace, after the transient.
    :rtype: list[SensorSignal]
    """
    return outputs[int(len(outputs) * TRANSIENT_FRACTION) :]


def _correlation(first: list[float], second: list[float]) -> float:
    """Compute the normalized zero-lag cross-correlation of two sequences.

    :param list[float] first: First sequence.
    :param list[float] second: Second sequence.
    :return: Correlation in [-1, 1]; 0 when either sequence is flat.
    :rtype: float
    """
    count = len(first)
    mean_first = sum(first) / count
    mean_second = sum(second) / count
    centered_first = [value - mean_first for value in first]
    centered_second = [value - mean_second for value in second]

    covariance = sum(
        a * b for a, b in zip(centered_first, centered_second, strict=True)
    )
    norm = math.sqrt(sum(a * a for a in centered_first)) * math.sqrt(
        sum(b * b for b in centered_second)
    )
    return covariance / norm if norm > 0.0 else 0.0


def test_preprocessor_outputs_a_moving_velocity_for_a_moving_input() -> None:
    """A walking input must produce a velocity signal that actually moves.

    The single most valuable assertion in this module: it is what a dead or
    unstepped filter in the velocity path fails.
    """
    preprocessor = SensorPreprocessor(BasicConfig())
    steady = _steady(_run(preprocessor, synthetic_gait(duration_s=WARMUP_S)))

    peak_velocity = max(abs(signal.velocity_rad_per_sec) for signal in steady)
    assert peak_velocity > 0.01, (
        "preprocessor produced an essentially static velocity for a walking "
        "input -- the velocity stage is not in the signal path"
    )


def test_preprocessor_velocity_follows_the_true_derivative() -> None:
    """The velocity output is in phase with the analytic derivative of the input.

    The SOGI quadrature is proportional to, not equal to, the true derivative
    (it carries a 1/omega scaling), so this checks shape and phase rather than
    magnitude.
    """
    signals = synthetic_gait(duration_s=WARMUP_S)
    preprocessor = SensorPreprocessor(BasicConfig())
    outputs = _run(preprocessor, signals)

    start = int(len(outputs) * TRANSIENT_FRACTION)
    measured = [signal.velocity_rad_per_sec for signal in outputs[start:]]
    expected = [signal.velocity_rad_per_sec for signal in signals[start:]]

    assert _correlation(measured, expected) > 0.9


def test_preprocessor_angle_follows_the_input_angle() -> None:
    """The filtered angle stays in phase with the raw angle."""
    signals = synthetic_gait(duration_s=WARMUP_S)
    preprocessor = SensorPreprocessor(BasicConfig())
    outputs = _run(preprocessor, signals)

    start = int(len(outputs) * TRANSIENT_FRACTION)
    measured = [signal.angle_rad for signal in outputs[start:]]
    expected = [signal.angle_rad for signal in signals[start:]]

    assert _correlation(measured, expected) > 0.9


def test_preprocessor_removes_a_constant_offset() -> None:
    """A mounting offset on the raw angle does not survive to the output."""
    offset_rad = 0.5
    biased = [
        SensorSignal(
            timestamp=signal.timestamp,
            angle_rad=signal.angle_rad + offset_rad,
            velocity_rad_per_sec=signal.velocity_rad_per_sec,
        )
        for signal in synthetic_gait(duration_s=WARMUP_S)
    ]
    preprocessor = SensorPreprocessor(BasicConfig())
    steady = _steady(_run(preprocessor, biased))

    mean_angle = sum(signal.angle_rad for signal in steady) / len(steady)
    assert abs(mean_angle) < 0.1 * offset_rad


def test_set_locomotion_mode_changes_the_filtered_output() -> None:
    """Per-mode retuning must reach the filter that is actually in the path.

    Level and descend carry different SOGI parameters, so identical input fed
    to two preprocessors in different modes must not produce identical output.
    An unstepped or mis-wired second filter instance makes these identical.
    """
    signals = synthetic_gait(duration_s=WARMUP_S)

    level = SensorPreprocessor(BasicConfig())
    level.set_locomotion_mode(0)
    level_out = _steady(_run(level, signals))

    descend = SensorPreprocessor(BasicConfig())
    descend.set_locomotion_mode(2)
    descend_out = _steady(_run(descend, signals))

    differences = [
        abs(a.angle_rad - b.angle_rad)
        for a, b in zip(level_out, descend_out, strict=True)
    ]
    assert max(differences) > 1e-6, (
        "level and descend produced identical filtered angles -- "
        "set_locomotion_mode is not reaching the active filter"
    )


def test_set_demo_mode_changes_the_filtered_output() -> None:
    """Demo mode swaps in a wider-bandwidth SOGI config, so output must change."""
    signals = synthetic_gait(duration_s=WARMUP_S)

    default = SensorPreprocessor(BasicConfig())
    default_out = _steady(_run(default, signals))

    demo = SensorPreprocessor(BasicConfig())
    demo.set_demo_mode()
    demo_out = _steady(_run(demo, signals))

    differences = [
        abs(a.angle_rad - b.angle_rad)
        for a, b in zip(default_out, demo_out, strict=True)
    ]
    assert max(differences) > 1e-6, (
        "demo mode produced identical filtered angles -- "
        "set_demo_mode is not reaching the active filter"
    )


def test_pausing_walking_mode_decays_the_filtered_angle() -> None:
    """The stand-still gate must reach the SOGI and quiet its oscillator."""
    preprocessor = SensorPreprocessor(BasicConfig())
    walking = _steady(_run(preprocessor, synthetic_gait(duration_s=WARMUP_S)))
    peak_walking = max(abs(signal.angle_rad) for signal in walking)

    preprocessor.set_walking_mode(False)
    still = [
        SensorSignal(
            timestamp=WARMUP_S + i / SAMPLE_RATE_HZ,
            angle_rad=0.0,
            velocity_rad_per_sec=0.0,
        )
        for i in range(1000)
    ]
    paused = _run(preprocessor, still)
    assert abs(paused[-1].angle_rad) < 0.05 * peak_walking


def test_preprocessor_reset_restores_initial_response() -> None:
    """After reset the preprocessor behaves as a freshly constructed one."""
    signals = synthetic_gait(duration_s=5.0)

    used = SensorPreprocessor(BasicConfig())
    _run(used, signals)
    used.reset()
    assert used.last_velocity_surrogate_rad_per_sec is None
    assert used.last_drift_removed_angle_rad is None

    fresh = SensorPreprocessor(BasicConfig())
    for from_used, from_fresh in zip(
        _run(used, signals), _run(fresh, signals), strict=True
    ):
        assert from_used.angle_rad == from_fresh.angle_rad
        assert from_used.velocity_rad_per_sec == from_fresh.velocity_rad_per_sec


def test_preprocessor_output_is_finite_for_extreme_input() -> None:
    """Implausible sensor values never produce NaN or infinity."""
    preprocessor = SensorPreprocessor(BasicConfig())
    extremes = [
        SensorSignal(
            timestamp=i / SAMPLE_RATE_HZ,
            angle_rad=angle_rad,
            velocity_rad_per_sec=velocity_rad_per_sec,
        )
        for i, (angle_rad, velocity_rad_per_sec) in enumerate(
            [(0.0, 0.0), (1e6, 1e6), (-1e6, -1e6), (1e-300, 1e-300), (0.0, 1e12)]
        )
    ]
    for signal in _run(preprocessor, extremes):
        assert math.isfinite(signal.angle_rad)
        assert math.isfinite(signal.velocity_rad_per_sec)


def test_preprocessor_rejects_non_monotonic_timestamps() -> None:
    """A backwards or repeated timestamp is an error, not silently absorbed."""
    preprocessor = SensorPreprocessor(BasicConfig())
    preprocessor.filter(
        raw_signal=SensorSignal(timestamp=0.0, angle_rad=0.0, velocity_rad_per_sec=0.0)
    )
    preprocessor.filter(
        raw_signal=SensorSignal(timestamp=0.01, angle_rad=0.1, velocity_rad_per_sec=0.1)
    )
    with pytest.raises(ValueError, match="Non-positive time_difference"):
        preprocessor.filter(
            raw_signal=SensorSignal(
                timestamp=0.005, angle_rad=0.1, velocity_rad_per_sec=0.1
            )
        )


def test_preprocessor_first_sample_passes_through() -> None:
    """The first sample has no dt available, so it is returned unchanged."""
    preprocessor = SensorPreprocessor(BasicConfig())
    first = SensorSignal(timestamp=0.0, angle_rad=0.3, velocity_rad_per_sec=0.7)
    out = preprocessor.filter(raw_signal=first)
    assert out.angle_rad == first.angle_rad
    assert out.velocity_rad_per_sec == first.velocity_rad_per_sec
