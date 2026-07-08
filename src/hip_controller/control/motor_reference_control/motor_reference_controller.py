"""Mid-level control functions."""

from loguru import logger
from scipy.interpolate import CubicSpline

from hip_controller.definitions import (
    LAG_COMPENSATION,
    LOOKUP_TABLEDATA_ASCEND,
    LOOKUP_TABLEDATA_DESCEND,
    LOOKUP_TABLEDATA_LEVEL,
    LookUpTable,
    LowPassFilterConfig,
    PositionLimitation,
)
from hip_controller.filters.second_order_low_pass_filter import SecondOrderLowPassFilter
from hip_controller.utils.math_utils import transform_to_cyclic

# Cutoff for the motor-command low-pass filter. Originally added to smooth a
# step input created by the angle-sign safety gate in app.py; that gate has
# since been removed and the raw lookup*amplitude signal is naturally smooth.
# The LPF now serves only as a light noise filter on top of the SOGI/gait
# phase outputs, so we use a higher cutoff (less lag) than the colleague's
# wn=40 amplitude-path LPF (FLEX_EXT.slx -> HL_Contr/Subsystem/Subsystem2).
# Drop to 40 if you re-enable the gate; raise toward 100 (or remove the LPF
# entirely in compute_motor_command) for an even snappier response.
MOTOR_CMD_LPF_WN_RAD_PER_SEC = 60.0
MOTOR_CMD_LPF_ZT = 1.0
MOTOR_CMD_LPF_DT_FALLBACK = 0.01


class MotionReferenceController:
    """Mid-level controller for motion reference."""

    def __init__(self) -> None:
        """Initialize the mid-level controller."""
        # Initialize the mid-level controller with a 1-D Lookup Table for motion mapping.
        self.motion_mapping = MotionMapping()

        # Most recent motion-mapping (cubic-spline) output, before amplitude
        # scaling and saturation. None until the first compute_motor_command
        # call or after a reset. Exposed for logging by external code.
        self.last_mapping_value: float | None = None

        # 2nd-order LPF on the motor command. See MOTOR_CMD_LPF_WN_RAD_PER_SEC.
        self._motor_cmd_lpf = SecondOrderLowPassFilter(
            LowPassFilterConfig(
                cut_off_frequency_rad_per_sec=MOTOR_CMD_LPF_WN_RAD_PER_SEC,
                damping_ratio=MOTOR_CMD_LPF_ZT,
                initial_condition=0.0,
            )
        )
        self._prev_timestamp: float | None = None

    def compute_motor_command(
        self,
        gait_phase: float,
        amplitude: float,
        timestamp: float | None = None,
        flexion_active: bool = True,
    ) -> float:
        """Compute the motor command based on the gait phase and amplitude.

        :param gait_phase: current gait phase in radians.
        :param amplitude: current amplitude modulation factor.
        :param timestamp: optional current sample time [s]. When provided the
            LPF uses the actual dt from the previous sample; otherwise it
            falls back to ``MOTOR_CMD_LPF_DT_FALLBACK`` (100 Hz default loop).
        :param flexion_active: when False, the raw lookup*amplitude product is
            forced to 0 BEFORE the LPF so the safety-gate transition is itself
            smoothed by the LPF instead of stepping the output. Caller should
            pass ``filtered_signal.angle_rad >= 0`` (the existing safety gate).
        :return: Reference motion command for the motor.
        :rtype: float
        """
        # Applies a phase offset and sinusoidal transformation to the computed gait phase
        sinusoidal_behavior_gait_phase = transform_to_cyclic(
            gait_phase + LAG_COMPENSATION
        )

        mapping_value = self.motion_mapping.spline(value=sinusoidal_behavior_gait_phase)
        self.last_mapping_value = float(mapping_value)

        raw_command = float(mapping_value * amplitude)
        # Safety gate folded INTO the LPF input: when the limb is in extension
        # the input drops to 0 and the LPF smoothly decays the command toward
        # zero, instead of the outer caller snapping it to 0 (which would
        # bypass the LPF and re-introduce the step we are trying to avoid).
        if not flexion_active:
            raw_command = 0.0

        # Smooth the command so motor velocity stays within physical V_MAX.
        if timestamp is not None and self._prev_timestamp is not None:
            dt = timestamp - self._prev_timestamp
            if dt <= 0.0 or dt > 1.0:
                dt = MOTOR_CMD_LPF_DT_FALLBACK
        else:
            dt = MOTOR_CMD_LPF_DT_FALLBACK
        if timestamp is not None:
            self._prev_timestamp = timestamp
        motor_command, _ = self._motor_cmd_lpf.step(x=raw_command, time_difference=dt)

        # Saturation
        if motor_command < PositionLimitation.lower:
            logger.warning("Motor velocity command reached the lower limitation.")
            return PositionLimitation.lower
        elif motor_command > PositionLimitation.upper:
            logger.warning("Motor velocity command reached the upper limitation.")
            return PositionLimitation.upper
        else:
            return motor_command

    def reset(self) -> None:
        """Reset the motor-command LPF state (call on session start)."""
        self._motor_cmd_lpf.reset()
        self._prev_timestamp = None
        self.last_mapping_value = None

    def set_locomotion_mode(self, class_id: int) -> None:
        """Forward per-mode lookup-table swap to ``MotionMapping``."""
        self.motion_mapping.set_locomotion_mode(class_id)


class MotionMapping:
    """1-D Lookup Table for motion mapping with cubic spline interpolation.

    Per-locomotion-mode tables: the flexion half is identical across
    modes (only ``ModeParameters.gain`` scales flexion-assist strength).
    The extension half differs:
      0 (Level)   -> small positive counter-pull (table values up to 0.025).
      1 (Ascend)  -> zero (motor holds rest position between strides).
      2 (Descend) -> zero (motor holds rest position between strides).

    This prevents the motor from paying out cable between strides on
    stair modes; see ``definitions.py:LOOKUP_TABLEDATA_*`` for the full
    rationale. At construction all three splines are built and Level is
    selected by default; ``set_locomotion_mode(class_id)`` switches
    which spline ``spline()`` evaluates.
    """

    def __init__(self):
        """Build all three per-mode splines; start in Level Ground."""
        bp = LookUpTable.breakpoints
        self._splines = {
            0: CubicSpline(x=bp, y=LOOKUP_TABLEDATA_LEVEL, extrapolate=True),
            1: CubicSpline(x=bp, y=LOOKUP_TABLEDATA_ASCEND, extrapolate=True),
            2: CubicSpline(x=bp, y=LOOKUP_TABLEDATA_DESCEND, extrapolate=True),
        }
        self._cubic_spline = self._splines[0]

    def set_locomotion_mode(self, class_id: int) -> None:
        """Activate the per-mode spline. Unknown class_ids fall back to Level."""
        self._cubic_spline = self._splines.get(class_id, self._splines[0])

    def spline(self, value: float):
        """Evaluate the currently-active per-mode lookup table at value."""
        return self._cubic_spline(value)
