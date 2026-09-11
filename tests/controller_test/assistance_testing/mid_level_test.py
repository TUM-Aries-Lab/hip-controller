"""Tests for mid-level control functions.

``test_transform_gait_phase`` and ``test_cubic_spline_interpolation`` compare
against recorded traces, which is appropriate: both are pinned to a
mathematical specification (a negative sine, and a cubic spline through a fixed
breakpoint table) rather than to a tuning constant, so retuning the controller
cannot invalidate them. Everything below them is a property test.
"""

import math

import numpy as np
from pandas import read_csv

from hip_controller.control.motor_reference_control.motor_reference_controller import (
    MotionMapping,
    MotionReferenceController,
    transform_to_cyclic,
)
from hip_controller.definitions import PositionLimitation
from tests.conftest import (
    DATA_REFERENCE_MOTION_LEFT,
    REL_TOL,
    SAMPLE_RATE_HZ,
    KinematicsDataColumnName,
)

# Samples for the motor-command LPF (critically damped, wn = 60 rad/s) to
# reach steady state. ~0.1 s is enough; 300 samples (3 s) leaves wide margin.
SETTLING_SAMPLES = 300


def _drive(
    controller: MotionReferenceController,
    gait_phase: float,
    amplitude: float,
    flexion_active: bool = True,
    start_index: int = 0,
) -> float:
    """Hold a constant phase and amplitude until the command LPF settles.

    :param MotionReferenceController controller: Instance under test.
    :param float gait_phase: Constant gait phase [rad].
    :param float amplitude: Constant amplitude modulation factor.
    :param bool flexion_active: Value forwarded as the safety gate.
    :param int start_index: Sample index the timestamps start from.
    :return: Motor command after the final sample [rad].
    :rtype: float
    """
    command = 0.0
    for i in range(start_index, start_index + SETTLING_SAMPLES):
        command = controller.compute_motor_command(
            gait_phase=gait_phase,
            amplitude=amplitude,
            timestamp=i / SAMPLE_RATE_HZ,
            flexion_active=flexion_active,
        )
    return command


def test_transform_gait_phase() -> None:
    """Test the calculation of sinusoidal behavior.

    :return: None
    """
    df = read_csv(filepath_or_buffer=DATA_REFERENCE_MOTION_LEFT)

    for i in range(0, len(df)):
        curr = df.iloc[i]

        # Arrange
        gait_phase = curr[KinematicsDataColumnName.GAIT_PHASE_LEFT]
        # Act
        sinusoidal_behavior = transform_to_cyclic(val=gait_phase)

        # Assert
        expected_sinusoidal_behavior = curr[KinematicsDataColumnName.SIN_WAVE_LEFT]

        assert math.isclose(
            sinusoidal_behavior, expected_sinusoidal_behavior, rel_tol=REL_TOL
        ), f"Row {i}"


def test_cubic_spline_interpolation() -> None:
    """Test the calculation of cubic spline interpolation.

    :return: None
    """
    df = read_csv(filepath_or_buffer=DATA_REFERENCE_MOTION_LEFT)
    lookup = MotionMapping()

    values = []
    expected_values = []

    n = len(df)
    for i in range(0, n):
        curr = df.iloc[i]

        # Arrange
        key = curr[KinematicsDataColumnName.MAPPING_KEY]

        # Act
        value = lookup.spline(key)
        values.append(value)

        # Assert
        expected_value = curr[KinematicsDataColumnName.MAPPING_VALUE]
        expected_values.append(expected_value)

    # The interpolation works slightly different with matlab simulink but the shape of the curve is the same
    np.testing.assert_array_almost_equal(values, expected_values, decimal=3)


# --------------------------------------------------------------------------
# compute_motor_command -- invariants that survive retuning.
# --------------------------------------------------------------------------


def test_motor_command_never_leaves_position_limits() -> None:
    """Saturation holds even for an amplitude far beyond anything realistic.

    This is a safety invariant: whatever the amplitude modulation produces, the
    commanded motor position must stay inside ``PositionLimitation``.
    """
    for amplitude in (1e3, -1e3, 1e9, -1e9):
        controller = MotionReferenceController()
        command = _drive(controller, gait_phase=math.pi / 2, amplitude=amplitude)
        assert PositionLimitation.lower <= command <= PositionLimitation.upper


def test_motor_command_within_limits_across_a_full_phase_sweep() -> None:
    """Every phase in a full cycle keeps the command inside the limits."""
    controller = MotionReferenceController()
    for i in range(1000):
        command = controller.compute_motor_command(
            gait_phase=(i / 1000.0) * 4.0 * math.pi - 2.0 * math.pi,
            amplitude=-6.5,
            timestamp=i / SAMPLE_RATE_HZ,
        )
        assert PositionLimitation.lower <= command <= PositionLimitation.upper


def test_motor_command_is_zero_without_amplitude() -> None:
    """Zero amplitude commands no motion regardless of gait phase."""
    controller = MotionReferenceController()
    assert _drive(controller, gait_phase=math.pi / 2, amplitude=0.0) == 0.0


def test_motor_command_is_periodic_in_gait_phase() -> None:
    """Shifting the gait phase by a full turn leaves the command unchanged."""
    base = MotionReferenceController()
    shifted = MotionReferenceController()
    for i in range(200):
        phase = i * 0.05
        timestamp = i / SAMPLE_RATE_HZ
        assert math.isclose(
            base.compute_motor_command(
                gait_phase=phase, amplitude=-6.5, timestamp=timestamp
            ),
            shifted.compute_motor_command(
                gait_phase=phase + 2.0 * math.pi, amplitude=-6.5, timestamp=timestamp
            ),
            rel_tol=1e-9,
            abs_tol=1e-12,
        )


def test_motor_command_decays_to_zero_when_flexion_gate_closes() -> None:
    """Closing the safety gate drives the command to zero through the LPF."""
    controller = MotionReferenceController()
    engaged = _drive(controller, gait_phase=math.pi / 2, amplitude=-6.5)
    assert abs(engaged) > 1e-3

    gated = _drive(
        controller,
        gait_phase=math.pi / 2,
        amplitude=-6.5,
        flexion_active=False,
        start_index=SETTLING_SAMPLES,
    )
    assert math.isclose(gated, 0.0, abs_tol=1e-9)


def test_motor_command_is_finite_for_extreme_inputs() -> None:
    """Extreme phases and amplitudes never produce NaN or infinity."""
    controller = MotionReferenceController()
    for gait_phase, amplitude in (
        (1e6, -6.5),
        (-1e6, -6.5),
        (0.0, 1e12),
        (math.pi, -1e12),
    ):
        command = controller.compute_motor_command(
            gait_phase=gait_phase, amplitude=amplitude
        )
        assert math.isfinite(command)


def test_motor_command_is_deterministic() -> None:
    """Two fresh controllers given identical inputs agree sample for sample."""
    first = MotionReferenceController()
    second = MotionReferenceController()
    for i in range(200):
        phase = i * 0.05
        timestamp = i / SAMPLE_RATE_HZ
        assert first.compute_motor_command(
            gait_phase=phase, amplitude=-6.5, timestamp=timestamp
        ) == second.compute_motor_command(
            gait_phase=phase, amplitude=-6.5, timestamp=timestamp
        )


def test_reset_restores_initial_response() -> None:
    """After reset the controller responds as a freshly constructed one."""
    used = MotionReferenceController()
    _drive(used, gait_phase=math.pi / 2, amplitude=-6.5)
    used.reset()
    assert used.last_mapping_value is None

    fresh = MotionReferenceController()
    for i in range(100):
        phase = i * 0.05
        timestamp = i / SAMPLE_RATE_HZ
        assert used.compute_motor_command(
            gait_phase=phase, amplitude=-6.5, timestamp=timestamp
        ) == fresh.compute_motor_command(
            gait_phase=phase, amplitude=-6.5, timestamp=timestamp
        )


def test_locomotion_mode_switch_changes_extension_half_of_lookup() -> None:
    """Level keeps a small extension counter-pull; the stair modes zero it.

    The flexion half is shared across modes, so the distinguishing property is
    the extension half. Values follow from ``LOOKUP_TABLEDATA_*``.
    """
    mapping = MotionMapping()

    mapping.set_locomotion_mode(0)
    level_extension = float(mapping.spline(1.0))
    mapping.set_locomotion_mode(1)
    ascend_extension = float(mapping.spline(1.0))
    mapping.set_locomotion_mode(2)
    descend_extension = float(mapping.spline(1.0))

    assert level_extension > 0.0
    assert math.isclose(ascend_extension, 0.0, abs_tol=1e-12)
    assert math.isclose(descend_extension, 0.0, abs_tol=1e-12)


def test_unknown_locomotion_mode_falls_back_to_level() -> None:
    """An unrecognised class id selects the level-ground table."""
    level = MotionMapping()
    level.set_locomotion_mode(0)
    fallback = MotionMapping()
    fallback.set_locomotion_mode(99)
    assert float(level.spline(1.0)) == float(fallback.spline(1.0))
