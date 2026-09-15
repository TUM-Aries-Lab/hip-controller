"""Property tests for trigger-driven baseline removal (hip angle offset).

These assert what baseline removal is *for* rather than pinning numbers, so they
survive retuning of the window length or the stand-still threshold.

The behaviour that matters, and that the earlier first-N-samples-at-start-up
design got wrong, is *when* the offset is taken: on an operator trigger, only
while the limb is still, repeatable at any time, and never by fabricating an
output value.
"""

import math

from hip_controller.control.signal_processing.baseline_removal import (
    BaselineRemoval,
)
from hip_controller.control.signal_processing.sensor_preprocessor import (
    SensorPreprocessor,
)
from hip_controller.definitions import (
    BaselineRemovalConfig,
    BasicConfig,
    SensorSignal,
)
from tests.conftest import SAMPLE_RATE_HZ, synthetic_gait

WINDOW_SAMPLE_COUNT = 20
STILL_VELOCITY_RAD_PER_SEC = 0.01
MOVING_VELOCITY_RAD_PER_SEC = 5.0
TOLERANCE_RAD = 1e-9


def _baseline_removal() -> BaselineRemoval:
    """Build a baseline removal with a known window length.

    :return: A freshly constructed baseline removal.
    :rtype: BaselineRemoval
    """
    return BaselineRemoval(
        BaselineRemovalConfig(window_sample_count=WINDOW_SAMPLE_COUNT)
    )


def _hold_still(
    baseline: BaselineRemoval, angle_rad: float, samples: int
) -> list[float]:
    """Feed still samples at a constant angle.

    :param BaselineRemoval baseline: Instance under test.
    :param float angle_rad: Constant angle to feed [rad].
    :param int samples: Number of samples to feed.
    :return: The corrected angle for each sample [rad].
    :rtype: list[float]
    """
    return [
        baseline.apply(
            angle_rad=angle_rad, velocity_rad_per_sec=STILL_VELOCITY_RAD_PER_SEC
        )
        for _ in range(samples)
    ]


def test_angle_passes_through_untouched_before_any_baseline() -> None:
    """Without a trigger the angle must be unchanged -- no offset, no forced zero.

    The start-up grab this replaces forced the output to exactly 0.0 for its
    first samples, injecting a value the sensor never produced into stateful
    downstream filters.
    """
    baseline = _baseline_removal()

    outputs = _hold_still(baseline, angle_rad=0.3, samples=WINDOW_SAMPLE_COUNT * 3)

    assert baseline.offset_rad == 0.0
    assert all(abs(output - 0.3) < TOLERANCE_RAD for output in outputs)


def test_trigger_takes_the_offset_and_zeroes_the_reference_posture() -> None:
    """A held trigger over a still limb must centre the angle on that posture."""
    baseline = _baseline_removal()
    offset_rad = 0.42

    baseline.set_trigger(active=True)
    _hold_still(baseline, angle_rad=offset_rad, samples=WINDOW_SAMPLE_COUNT)

    assert abs(baseline.offset_rad - offset_rad) < TOLERANCE_RAD
    corrected = baseline.apply(
        angle_rad=offset_rad, velocity_rad_per_sec=STILL_VELOCITY_RAD_PER_SEC
    )
    assert abs(corrected) < TOLERANCE_RAD


def test_motion_during_the_window_restarts_it_instead_of_averaging() -> None:
    """Triggering mid-stride must not produce an offset from a moving leg.

    This is the failure mode the trigger exists to prevent: the offset has to
    come from the reference posture, not from wherever the leg happened to be.
    """
    baseline = _baseline_removal()

    baseline.set_trigger(active=True)
    for _ in range(WINDOW_SAMPLE_COUNT * 3):
        baseline.apply(angle_rad=0.8, velocity_rad_per_sec=MOVING_VELOCITY_RAD_PER_SEC)

    assert baseline.offset_rad == 0.0
    assert baseline.is_collecting, "the window must stay open until the limb stills"

    _hold_still(baseline, angle_rad=0.25, samples=WINDOW_SAMPLE_COUNT)
    assert abs(baseline.offset_rad - 0.25) < TOLERANCE_RAD


def test_release_before_the_window_fills_uses_what_was_collected() -> None:
    """Releasing the trigger early averages the samples gathered so far.

    Matches the Simulink falling-edge behaviour in ``getHipKinematics_IMU.m``.
    """
    baseline = _baseline_removal()

    baseline.set_trigger(active=True)
    _hold_still(baseline, angle_rad=0.5, samples=WINDOW_SAMPLE_COUNT // 2)
    baseline.set_trigger(active=False)

    assert abs(baseline.offset_rad - 0.5) < TOLERANCE_RAD
    assert not baseline.is_collecting


def test_release_without_a_still_sample_keeps_the_previous_offset() -> None:
    """A useless baseline attempt must not destroy a good offset."""
    baseline = _baseline_removal()

    baseline.set_trigger(active=True)
    _hold_still(baseline, angle_rad=0.3, samples=WINDOW_SAMPLE_COUNT)
    baseline.set_trigger(active=False)

    baseline.set_trigger(active=True)
    baseline.apply(angle_rad=1.0, velocity_rad_per_sec=MOVING_VELOCITY_RAD_PER_SEC)
    baseline.set_trigger(active=False)

    assert abs(baseline.offset_rad - 0.3) < TOLERANCE_RAD


def test_a_second_trigger_replaces_the_offset() -> None:
    """The offset must be re-takeable mid-session, e.g. after a strap slips."""
    baseline = _baseline_removal()

    baseline.set_trigger(active=True)
    _hold_still(baseline, angle_rad=0.2, samples=WINDOW_SAMPLE_COUNT)
    baseline.set_trigger(active=False)

    baseline.set_trigger(active=True)
    _hold_still(baseline, angle_rad=-0.35, samples=WINDOW_SAMPLE_COUNT)
    baseline.set_trigger(active=False)

    assert abs(baseline.offset_rad + 0.35) < TOLERANCE_RAD


def test_repeated_identical_trigger_states_are_not_edges() -> None:
    """Calling the trigger every loop iteration must not reopen the window.

    The runner passes the main switch through on every sample, so a held switch
    must not keep re-taking the baseline while the subject walks.
    """
    baseline = _baseline_removal()

    baseline.set_trigger(active=True)
    _hold_still(baseline, angle_rad=0.2, samples=WINDOW_SAMPLE_COUNT)
    for _ in range(5):
        baseline.set_trigger(active=True)
    _hold_still(baseline, angle_rad=0.9, samples=WINDOW_SAMPLE_COUNT)

    assert abs(baseline.offset_rad - 0.2) < TOLERANCE_RAD


def test_a_replayed_trigger_reproduces_the_offset() -> None:
    """Playback drives the recorded main switch and gets the session's offset.

    This is why there is no baseline-at-start-up fallback: the trigger travels
    with the data, so the offline path reproduces what the rig did instead of
    inventing a moment to take the offset at.
    """
    live = _baseline_removal()
    replay = _baseline_removal()

    for baseline in (live, replay):
        baseline.set_trigger(active=True)
        _hold_still(baseline, angle_rad=0.15, samples=WINDOW_SAMPLE_COUNT)
        baseline.set_trigger(active=False)

    assert abs(replay.offset_rad - live.offset_rad) < TOLERANCE_RAD


def test_reset_drops_the_offset() -> None:
    """After a dropout the stale offset must not survive."""
    baseline = _baseline_removal()

    baseline.set_trigger(active=True)
    _hold_still(baseline, angle_rad=0.6, samples=WINDOW_SAMPLE_COUNT)
    baseline.reset()

    assert baseline.offset_rad == 0.0


def test_preprocessor_removes_a_constant_offset_after_a_trigger() -> None:
    """End to end: the pipeline must see the corrected angle, not the raw one."""
    preprocessor = SensorPreprocessor(BasicConfig())
    offset_rad = 0.33

    preprocessor.set_baseline_removal_trigger(active=True)
    for index in range(WINDOW_SAMPLE_COUNT * 2):
        preprocessor.filter(
            raw_signal=SensorSignal(
                timestamp=index / SAMPLE_RATE_HZ,
                angle_rad=offset_rad,
                velocity_rad_per_sec=0.0,
            )
        )
    preprocessor.set_baseline_removal_trigger(active=False)

    assert abs(preprocessor.baseline_offset_rad - offset_rad) < 1e-6


def test_preprocessor_does_not_mutate_the_caller_signal() -> None:
    """The caller's raw sample must survive preprocessing unchanged.

    The signal object is shared with the recording path, so mutating it in
    place would silently corrupt the logged raw angle -- and a recording whose
    angle is already offset-corrected would be double-corrected on replay.
    """
    preprocessor = SensorPreprocessor(BasicConfig())

    preprocessor.set_baseline_removal_trigger(active=True)
    for index in range(WINDOW_SAMPLE_COUNT * 2):
        preprocessor.filter(
            raw_signal=SensorSignal(
                timestamp=index / SAMPLE_RATE_HZ,
                angle_rad=0.7,
                velocity_rad_per_sec=0.0,
            )
        )

    raw_signal = SensorSignal(
        timestamp=(WINDOW_SAMPLE_COUNT * 2) / SAMPLE_RATE_HZ,
        angle_rad=0.7,
        velocity_rad_per_sec=0.0,
    )
    preprocessor.filter(raw_signal=raw_signal)

    assert raw_signal.angle_rad == 0.7


def test_an_untriggered_recording_replays_with_no_offset() -> None:
    """Without a trigger nothing happens, however long the trace runs.

    A run whose main switch is never driven -- because the recording has no such
    column, or because baseline removal was switched off for the replay -- must
    say so with a zero offset rather than manufacturing one from whatever the
    limb was doing at the start of the file.
    """
    preprocessor = SensorPreprocessor(BasicConfig())

    for signal in synthetic_gait(duration_s=10.0):
        preprocessor.filter(raw_signal=signal)

    assert math.isclose(preprocessor.baseline_offset_rad, 0.0, abs_tol=TOLERANCE_RAD)
