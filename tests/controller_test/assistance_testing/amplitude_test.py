"""Property tests for amplitude modulation.

These assert invariants that hold for *any* tuning of ``AMPLITUDE_GAIN``,
``SIGMOID_POWER`` or ``SCALE_LEVEL_MODE``, so retuning the controller does not
invalidate them. Two deliberate exceptions are marked in their docstrings: the
closed-form sigmoid tests (pinned to the mathematical definition of the
function, which tuning cannot change) and
:func:`test_mode_gains_match_specification` (pinned to a cross-repository
contract).
"""

import math
from itertools import pairwise

import pytest

from hip_controller.control.motor_reference_control.amplitude_modulation import (
    AmplitudeModulation,
    AscendStairsMode,
    DescendStairsMode,
    LevelGroundMode,
)
from hip_controller.definitions import SensorSignal
from tests.conftest import SAMPLE_RATE_HZ, synthetic_gait

# Samples needed for the sigmoid LPF to reach steady state. The filter is
# critically damped at wn = 30 rad/s, so ~0.2 s suffices; 400 samples (4 s) is
# a wide margin that stays valid if the cut-off is lowered substantially.
SETTLING_SAMPLES = 400


def _settle(
    modulation: AmplitudeModulation, angle_rad: float, velocity_rad_per_sec: float
) -> float:
    """Hold a constant signal until the sigmoid LPF settles; return the amplitude.

    :param AmplitudeModulation modulation: Instance under test.
    :param float angle_rad: Constant angle to hold [rad].
    :param float velocity_rad_per_sec: Constant velocity to hold [rad/s].
    :return: Steady-state amplitude.
    :rtype: float
    """
    amplitude = 0.0
    for i in range(SETTLING_SAMPLES):
        amplitude = modulation.compute_amplitude(
            signal=SensorSignal(
                timestamp=i / SAMPLE_RATE_HZ,
                angle_rad=angle_rad,
                velocity_rad_per_sec=velocity_rad_per_sec,
            )
        )
    return amplitude


# --------------------------------------------------------------------------
# apply_sigmoid_scaling -- a pure function, pinned to its mathematical spec.
# The expected values below follow from the definition a**n / (a**n + 1) and
# are therefore independent of any tuning constant.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("power", [2, 4, 8, 30])
@pytest.mark.parametrize("value", [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
def test_sigmoid_scaling_matches_closed_form(value: float, power: int) -> None:
    """Sigmoid scaling implements a**n / (a**n + 1)."""
    expected = value**power / (value**power + 1.0)
    actual = AmplitudeModulation.apply_sigmoid_scaling(value=value, power=power)
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15)


def test_sigmoid_scaling_is_zero_at_origin() -> None:
    """A stationary portrait radius produces no assist."""
    assert AmplitudeModulation.apply_sigmoid_scaling(value=0.0, power=30) == 0.0


def test_sigmoid_scaling_crosses_one_half_at_unit_radius() -> None:
    """The sigmoid threshold sits at a scaled radius of 1 for every power."""
    for power in (2, 10, 30, 80):
        assert math.isclose(
            AmplitudeModulation.apply_sigmoid_scaling(value=1.0, power=power), 0.5
        )


def test_sigmoid_scaling_is_monotonic_in_radius() -> None:
    """Larger motion never yields less assist."""
    values = [i * 0.05 for i in range(80)]
    outputs = [
        AmplitudeModulation.apply_sigmoid_scaling(value=v, power=30) for v in values
    ]
    assert all(b >= a for a, b in pairwise(outputs))


def test_sigmoid_scaling_stays_in_unit_interval() -> None:
    """Output is a gate in [0, 1] for every input, including overflowing ones."""
    for value in (0.0, 0.5, 1.0, 2.0, 1e3, 1e20):
        result = AmplitudeModulation.apply_sigmoid_scaling(value=value, power=30)
        assert 0.0 <= result <= 1.0


def test_sigmoid_scaling_saturates_instead_of_overflowing() -> None:
    """A radius that overflows float64 saturates the gate rather than raising."""
    assert AmplitudeModulation.apply_sigmoid_scaling(value=1e20, power=30) == 1.0


# --------------------------------------------------------------------------
# compute_amplitude -- invariants that survive retuning.
# --------------------------------------------------------------------------


def test_amplitude_is_zero_when_standing_still() -> None:
    """A perfectly still limb produces exactly zero assist."""
    modulation = AmplitudeModulation(reverse=False)
    assert _settle(modulation, angle_rad=0.0, velocity_rad_per_sec=0.0) == 0.0


def test_amplitude_never_exceeds_mode_gain() -> None:
    """|amplitude| is bounded by the active mode gain for any input."""
    for reverse in (False, True):
        modulation = AmplitudeModulation(reverse=reverse)
        limit = abs(LevelGroundMode().get_parameters().gain)
        for signal in synthetic_gait(amplitude_rad=2.0):
            assert abs(modulation.compute_amplitude(signal=signal)) <= limit + 1e-12


def test_amplitude_is_monotonic_in_portrait_radius() -> None:
    """Sustained larger motion yields at least as much assist."""
    magnitudes = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
    settled = [
        abs(
            _settle(
                AmplitudeModulation(reverse=False),
                angle_rad=magnitude,
                velocity_rad_per_sec=0.0,
            )
        )
        for magnitude in magnitudes
    ]
    assert all(b >= a - 1e-12 for a, b in pairwise(settled))


def test_amplitude_saturates_to_gain_for_large_motion() -> None:
    """Well past the sigmoid threshold the amplitude reaches the mode gain."""
    modulation = AmplitudeModulation(reverse=False)
    gain = LevelGroundMode().get_parameters().gain
    settled = _settle(modulation, angle_rad=10.0, velocity_rad_per_sec=0.0)
    assert math.isclose(settled, gain, rel_tol=1e-6)


def test_reverse_mirrors_amplitude_exactly() -> None:
    """The reverse flag negates the amplitude and changes nothing else."""
    forward = AmplitudeModulation(reverse=False)
    mirrored = AmplitudeModulation(reverse=True)
    for signal in synthetic_gait():
        assert forward.compute_amplitude(signal=signal) == -mirrored.compute_amplitude(
            signal=signal
        )


def test_amplitude_is_finite_for_extreme_inputs() -> None:
    """Extreme sensor values never produce NaN or infinity."""
    modulation = AmplitudeModulation(reverse=False)
    for angle_rad, velocity_rad_per_sec in (
        (1e6, 1e6),
        (-1e6, 1e6),
        (0.0, 1e12),
        (1e-300, 1e-300),
    ):
        amplitude = modulation.compute_amplitude(
            signal=SensorSignal(
                timestamp=None,
                angle_rad=angle_rad,
                velocity_rad_per_sec=velocity_rad_per_sec,
            )
        )
        assert math.isfinite(amplitude)


def test_reset_restores_initial_response() -> None:
    """After reset the modulation responds as a freshly constructed instance."""
    used = AmplitudeModulation(reverse=False)
    for signal in synthetic_gait(duration_s=3.0):
        used.compute_amplitude(signal=signal)
    used.reset()

    fresh = AmplitudeModulation(reverse=False)
    for signal in synthetic_gait(duration_s=1.0):
        assert used.compute_amplitude(signal=signal) == fresh.compute_amplitude(
            signal=signal
        )


def test_reset_clears_intermediates() -> None:
    """Reset drops the cached per-sample intermediates."""
    modulation = AmplitudeModulation(reverse=False)
    modulation.compute_amplitude(
        signal=SensorSignal(timestamp=0.0, angle_rad=0.3, velocity_rad_per_sec=0.1)
    )
    assert modulation.last_intermediates is not None
    modulation.reset()
    assert modulation.last_intermediates is None


def test_mode_switch_changes_gain() -> None:
    """Switching mode swaps in the new mode parameters."""
    modulation = AmplitudeModulation(reverse=False)
    modulation.set_mode(AscendStairsMode())
    settled = _settle(modulation, angle_rad=10.0, velocity_rad_per_sec=0.0)
    assert math.isclose(settled, AscendStairsMode().get_parameters().gain, rel_tol=1e-6)


def test_mode_gains_match_specification() -> None:
    """Pin the three per-mode amplitude gains.

    This is intentionally a value test. These gains are a contract with the
    downstream ``LocomotionMode_IMUbased`` repository, which carried its own
    local overrides; the two drifted apart silently once ``AMPLITUDE_GAIN`` was
    retuned here. Pinning them means a future retune has to be a deliberate
    edit to this list rather than an accident.
    """
    assert LevelGroundMode().get_parameters().gain == -6.5
    assert AscendStairsMode().get_parameters().gain == -9.5
    assert DescendStairsMode().get_parameters().gain == -5.5
