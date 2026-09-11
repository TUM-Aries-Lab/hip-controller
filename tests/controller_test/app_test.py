"""End-to-end property tests for the single-limb controller.

These drive :class:`WalkOnController` with generated gait traces and assert
invariants that hold for any tuning: the command stays inside the motor
position limits, stays finite, is deterministic, mirrors under the reverse
flag, and tracks the cadence it is given. None of them pin a recorded sample
value, so retuning the controller does not invalidate them.
"""

import math

import numpy as np
import pytest

from hip_controller.control.app import SensorSignal, WalkOnController
from hip_controller.definitions import BasicConfig, PositionLimitation
from tests.conftest import SAMPLE_RATE_HZ, synthetic_gait


def _run(controller: WalkOnController, signals: list[SensorSignal]) -> list[float]:
    """Step the controller over a trace and collect the motor commands.

    :param WalkOnController controller: Instance under test.
    :param list[SensorSignal] signals: Trace to feed, sample by sample.
    :return: Motor position command for each sample [rad].
    :rtype: list[float]
    """
    return [controller.step(curr_signal=signal) for signal in signals]


def test_command_never_leaves_position_limits() -> None:
    """The commanded motor position stays inside the configured limits.

    This is the controller's central safety invariant, so it is checked across
    a wide range of cadences and hip excursions rather than one nominal trace.
    """
    for frequency_hz in (0.4, 0.9, 1.6):
        for amplitude_rad in (0.05, 0.4, 1.5):
            controller = WalkOnController(
                left_limb=True, config=BasicConfig(filtered=True)
            )
            for command in _run(
                controller,
                synthetic_gait(frequency_hz=frequency_hz, amplitude_rad=amplitude_rad),
            ):
                assert PositionLimitation.lower <= command <= PositionLimitation.upper


def test_command_within_limits_through_the_preprocessing_path() -> None:
    """Limits also hold when the raw preprocessing pipeline is engaged."""
    controller = WalkOnController(left_limb=True, config=BasicConfig(filtered=False))
    for command in _run(controller, synthetic_gait(duration_s=10.0)):
        assert PositionLimitation.lower <= command <= PositionLimitation.upper


def test_command_is_finite_for_extreme_inputs() -> None:
    """Implausible sensor values never produce NaN or infinity."""
    controller = WalkOnController(left_limb=True, config=BasicConfig(filtered=True))
    extremes = [
        SensorSignal(timestamp=i * 0.01, angle_rad=angle, velocity_rad_per_sec=velocity)
        for i, (angle, velocity) in enumerate(
            [(0.0, 0.0), (1e6, 1e6), (-1e6, -1e6), (1e-300, 1e-300), (0.0, 1e12)]
        )
    ]
    for command in _run(controller, extremes):
        assert math.isfinite(command)


def test_standing_still_commands_no_assist() -> None:
    """A motionless limb produces no motor command."""
    controller = WalkOnController(left_limb=True, config=BasicConfig(filtered=True))
    still = [
        SensorSignal(
            timestamp=i / SAMPLE_RATE_HZ, angle_rad=0.0, velocity_rad_per_sec=0.0
        )
        for i in range(500)
    ]
    assert all(command == 0.0 for command in _run(controller, still))


def test_controller_is_deterministic() -> None:
    """Two fresh controllers given the same trace agree sample for sample."""
    signals = synthetic_gait()
    first = _run(
        WalkOnController(left_limb=True, config=BasicConfig(filtered=True)), signals
    )
    second = _run(
        WalkOnController(left_limb=True, config=BasicConfig(filtered=True)), signals
    )
    assert first == second


def test_reverse_mirrors_the_command() -> None:
    """The reverse flag negates the command and changes nothing else."""
    signals = synthetic_gait()
    forward = _run(
        WalkOnController(left_limb=True, config=BasicConfig(filtered=True)), signals
    )
    mirrored = _run(
        WalkOnController(left_limb=False, config=BasicConfig(filtered=True)), signals
    )
    for a, b in zip(forward, mirrored, strict=True):
        assert a == -b


def test_reset_clears_the_state_it_owns() -> None:
    """Reset clears the preprocessing, amplitude and motor-command state."""
    used = WalkOnController(left_limb=True, config=BasicConfig(filtered=True))
    _run(used, synthetic_gait(duration_s=3.0))
    used.reset()

    assert used.last_filtered_signal is None
    assert used.last_gait_phase_rad is None
    assert used.amplitude_modulation.last_intermediates is None
    assert used.motion_reference_controller.last_mapping_value is None


@pytest.mark.xfail(
    reason=(
        "WalkOnController.reset() does not reset the gait controller -- see the "
        "TODO in app.py. GaitController exposes no reset(), so extrema, "
        "centering and stride-detector state survive a reset and the next "
        "session starts from the previous session's gait estimate."
    ),
    strict=False,
)
def test_reset_restores_initial_response() -> None:
    """After reset the controller should respond as a freshly constructed one."""
    signals = synthetic_gait(duration_s=3.0)
    used = WalkOnController(left_limb=True, config=BasicConfig(filtered=True))
    _run(used, signals)
    used.reset()

    fresh = WalkOnController(left_limb=True, config=BasicConfig(filtered=True))
    assert _run(used, signals) == _run(fresh, signals)


def test_command_follows_the_imposed_cadence() -> None:
    """The command oscillates at the stride frequency it is driven with.

    Verified in the frequency domain: the dominant spectral component of the
    motor command should sit at the cadence of the input gait, which is the
    whole point of gait-phase-based assistance.
    """
    for frequency_hz in (0.6, 0.9, 1.3):
        controller = WalkOnController(left_limb=True, config=BasicConfig(filtered=True))
        commands = _run(
            controller, synthetic_gait(frequency_hz=frequency_hz, duration_s=40.0)
        )
        # Discard the start-up transient before measuring the spectrum.
        steady = np.asarray(commands[len(commands) // 4 :], dtype=float)
        steady = steady - steady.mean()

        spectrum = np.abs(np.fft.rfft(steady))
        freqs = np.fft.rfftfreq(steady.size, d=1.0 / SAMPLE_RATE_HZ)
        dominant_hz = float(freqs[int(np.argmax(spectrum))])

        assert abs(dominant_hz - frequency_hz) < 0.1 * frequency_hz


def test_locomotion_mode_switch_is_accepted_mid_trace() -> None:
    """Switching mode mid-walk keeps the command finite and inside the limits."""
    controller = WalkOnController(left_limb=True, config=BasicConfig(filtered=True))
    signals = synthetic_gait(duration_s=12.0)
    for index, signal in enumerate(signals):
        if index == len(signals) // 3:
            controller.set_locomotion_mode(1)
        elif index == 2 * len(signals) // 3:
            controller.set_locomotion_mode(2)
        command = controller.step(curr_signal=signal)
        assert math.isfinite(command)
        assert PositionLimitation.lower <= command <= PositionLimitation.upper


def test_demo_mode_is_accepted_mid_trace() -> None:
    """Enabling demo mode keeps the command finite and inside the limits."""
    controller = WalkOnController(left_limb=True, config=BasicConfig(filtered=True))
    controller.set_demo_mode()
    for command in _run(controller, synthetic_gait(duration_s=8.0)):
        assert math.isfinite(command)
        assert PositionLimitation.lower <= command <= PositionLimitation.upper
