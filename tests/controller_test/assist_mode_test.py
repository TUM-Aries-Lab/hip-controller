"""Property tests for the assist-mode axis.

:class:`AssistMode` is one axis carrying two families of context: locomotion
modes (level, stair ascent, stair descent) produced by the TCN classifier, and
treadmill ramp inclinations selected by the operator. They share an axis because
they are mutually exclusive -- nobody descends stairs while on a 5% ramp.

What these tests protect is the contract that makes the axis extensible: every
consumer of a ``class_id`` falls back to level for a value it does not
recognise, and every declared mode reaches all three consumers. A mode that is
declared but wired into only some of them is the failure this catches -- it
would assist with the wrong amplitude while filtering as though it were level,
and no assertion about the command being finite would notice.
"""

import math

import pytest

from hip_controller.control.app import AMPLITUDE_MODES, WalkOnController
from hip_controller.control.motor_reference_control.motor_reference_controller import (
    MotionMapping,
)
from hip_controller.control.signal_processing.sensor_preprocessor import (
    SensorPreprocessor,
)
from hip_controller.definitions import AssistMode, BasicConfig
from tests.conftest import synthetic_gait

# A value no mode uses, to exercise the fallback every consumer promises.
UNKNOWN_CLASS_ID = 99


def test_every_assist_mode_has_an_amplitude_strategy() -> None:
    """A declared mode with no amplitude entry would silently assist as level."""
    missing = [mode.name for mode in AssistMode if mode not in AMPLITUDE_MODES]

    assert not missing, f"AssistMode values without an amplitude strategy: {missing}"


def test_every_assist_mode_has_a_motion_mapping_spline() -> None:
    """A declared mode with no spline would silently use the level table."""
    mapping = MotionMapping()

    missing = [mode.name for mode in AssistMode if mode not in mapping._splines]

    assert not missing, f"AssistMode values without a lookup table: {missing}"


def test_every_assist_mode_has_a_sogi_config() -> None:
    """A declared mode with no SOGI entry would silently filter as level."""
    preprocessor = SensorPreprocessor(BasicConfig())

    missing = [
        mode.name
        for mode in AssistMode
        if mode not in preprocessor._sogi_config_by_mode
    ]

    assert not missing, f"AssistMode values without a SOGI config: {missing}"


@pytest.mark.parametrize("mode", list(AssistMode))
def test_every_assist_mode_drives_the_controller(mode: AssistMode) -> None:
    """Selecting any mode must leave the controller producing a usable command.

    Runs the whole fan-out -- amplitude, SOGI, lookup table -- rather than
    asserting on the tables, so a mode that is declared but crashes one consumer
    fails here.
    """
    controller = WalkOnController(left_limb=True, config=BasicConfig())
    controller.set_locomotion_mode(int(mode))

    commands = [controller.step(signal) for signal in synthetic_gait(duration_s=3.0)]

    assert all(math.isfinite(command) for command in commands)
    assert any(command != 0.0 for command in commands), (
        f"mode {mode.name} produced an identically zero command"
    )


def test_an_unknown_class_id_falls_back_to_level_everywhere() -> None:
    """The fallback is what lets older code stay safe against newer mode ids.

    A project may use only the subset of the axis it cares about, and a
    class_id from a newer version must degrade to level assist rather than
    raising.
    """
    mapping = MotionMapping()
    mapping.set_locomotion_mode(UNKNOWN_CLASS_ID)
    assert mapping._cubic_spline is mapping._splines[AssistMode.LEVEL]

    controller = WalkOnController(left_limb=True, config=BasicConfig())
    controller.set_locomotion_mode(UNKNOWN_CLASS_ID)

    assert controller.amplitude_modulation._mode is AMPLITUDE_MODES[AssistMode.LEVEL], (
        "an unknown class_id must select the level amplitude strategy"
    )


def test_the_ramps_are_uphill_inclinations_not_stair_modes() -> None:
    """The ramps carry the level tuning, deliberately, until measured.

    Ramp walking is continuous gait with a swing phase; the stair parameters
    were tuned on stairs. If someone later gives the ramps real values this test
    is expected to be updated -- it exists so the copy is a recorded decision
    rather than something nobody noticed.
    """
    level = AMPLITUDE_MODES[AssistMode.LEVEL].get_parameters()

    for ramp in (AssistMode.RAMP_2_5, AssistMode.RAMP_5):
        assert AMPLITUDE_MODES[ramp].get_parameters() == level, (
            f"{ramp.name} no longer matches level -- update this test if that "
            "is the intended tuning"
        )
