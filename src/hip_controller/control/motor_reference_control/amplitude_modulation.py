"""Amplitude modulation to eliminate too much motor activity due to small sensor movements."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from loguru import logger

from hip_controller.definitions import (
    AMPLITUDE_GAIN,
    SCALE_LEVEL_MODE,
    SIGMOID_POWER,
    VELOCITY_WEIGHT_LEVEL_MODE,
    LowPassFilterConfig,
    SensorSignal,
)
from hip_controller.filters.second_order_low_pass_filter import SecondOrderLowPassFilter

# 2nd-order LPF on the sigmoid gate output. Requires SUSTAINED motion to
# engage the assist -- brief wiggles only get partial engagement. Matches the
# colleague's FLEX_EXT.slx design (HL_Contr/Subsystem/SIGMOID SCALING1/
# Subsystem2 at wn=2 rad/s). At wn=2 the LPF reaches ~95% in about 1.5 s of
# sustained walking; sub-200ms wiggles only push it to ~20%.
#
# Tuning guide:
#   wn = 2  : 1.5 s full engagement (colleague default, strong stand-still rejection)
#   wn = 5  : 0.6 s engagement (still rejects sub-200ms wiggles)
#   wn = 10 : 0.3 s engagement (filters only very brief blips)
SIGMOID_LPF_WN_RAD_PER_SEC = 10.0
SIGMOID_LPF_ZT = 1.0
SIGMOID_LPF_DT_FALLBACK = 0.01


@dataclass
class ModeParameters:
    """Dataclass for mode parameters.

    :scale: multiplier applied to the whole portrait radius.
    :sigmoid_power: exponent sharpening the sigmoid threshold.
    :gain: multiplier applied to the final amplitude.
    :velocity_weight: per-component weight on velocity inside the portrait
        radius, ``r = sqrt(angle^2 + (velocity_weight * velocity)^2)``. 1.0
        reproduces the symmetric historical behavior.
    """

    scale: float
    sigmoid_power: int
    gain: float
    velocity_weight: float = 1.0


@dataclass
class AmplitudeIntermediates:
    """Per-sample intermediate values produced inside ``compute_amplitude``.

    Exposed so external code (e.g. the simulator) can log the full pipeline:
    portrait radius -> scaled portrait radius -> sigmoid -> scaled sigmoid ->
    final amplitude.

    :portrait_radius: ``sqrt(angle**2 + velocity**2)`` of the input signal.
    :scaled_portrait_radius: ``portrait_radius * mode.scale``.
    :sigmoid_scaling: Sigmoid output in [0, 1] (before gain/reverse).
    :scaled_sigmoid_scaling: ``sigmoid_scaling * mode.gain`` (before reverse).
    :amplitude: Final amplitude (``scaled_sigmoid_scaling * reverse``).
    """

    portrait_radius: float
    scaled_portrait_radius: float
    sigmoid_scaling: float
    scaled_sigmoid_scaling: float
    amplitude: float


class ModeStrategy(ABC):
    """Abstract mode class."""

    @abstractmethod
    def get_parameters(self) -> ModeParameters:
        """Abstract Method."""
        pass


class LevelGroundMode(ModeStrategy):
    """Mode for level grount."""

    def get_parameters(self) -> ModeParameters:
        """Get parameters for no stairs."""
        return ModeParameters(
            scale=SCALE_LEVEL_MODE,
            sigmoid_power=SIGMOID_POWER,
            gain=AMPLITUDE_GAIN,
            velocity_weight=VELOCITY_WEIGHT_LEVEL_MODE,
        )


# Parameters of ascending and descending stairs mode are not certain yet
class AscendStairsMode(ModeStrategy):
    """Mode for ascending stairs."""

    def get_parameters(self) -> ModeParameters:
        """Get parameters for ascending stairs."""
        return ModeParameters(
            scale=SCALE_LEVEL_MODE - 0,  # -0.6
            sigmoid_power=SIGMOID_POWER + 50,  # +100
            gain=AMPLITUDE_GAIN - 2,
            velocity_weight=VELOCITY_WEIGHT_LEVEL_MODE,
        )


class DescendStairsMode(ModeStrategy):
    """Mode for descending stairs."""

    def get_parameters(self) -> ModeParameters:
        """Get parameters for descending stairs."""
        return ModeParameters(
            scale=SCALE_LEVEL_MODE + 2.0,  # -0.5
            sigmoid_power=SIGMOID_POWER + 50,  # +100
            gain=AMPLITUDE_GAIN + 0.5,
            velocity_weight=VELOCITY_WEIGHT_LEVEL_MODE,
        )


class AmplitudeModulation:
    """Amplitude Modulation Class."""

    def __init__(self, reverse: bool):
        """Initialize for the amplitude modulation block with level ground mode set as default."""
        self._mode = LevelGroundMode()

        # The amplitude could be reversed due to different wiring settings
        if reverse:
            self.reverse_amplitude: int = -1
        else:
            self.reverse_amplitude: int = 1

        # Most recent per-stage values from compute_amplitude(); None until the
        # first call. Exposed for logging by external code.
        self.last_intermediates: AmplitudeIntermediates | None = None

        # LPF on the sigmoid gate output (see SIGMOID_LPF_WN_RAD_PER_SEC
        # comment at module top). Without this, a brief small motion (e.g.
        # weight shift while standing) trips the sigmoid for a few samples
        # and produces a full-amplitude command -- exactly the "standing
        # still still triggers the motor" complaint.
        self._sigmoid_lpf = SecondOrderLowPassFilter(
            LowPassFilterConfig(
                cut_off_frequency_rad_per_sec=SIGMOID_LPF_WN_RAD_PER_SEC,
                damping_ratio=SIGMOID_LPF_ZT,
                initial_condition=0.0,
            )
        )
        self._prev_timestamp: float | None = None

    def set_mode(self, mode: ModeStrategy):
        """Switch mode at runtime."""
        self._mode = mode

    @staticmethod
    def _compute_portrait_radius(
        signal: SensorSignal, velocity_weight: float = 1.0
    ) -> float:
        """Calculate the portrait radius with an optional velocity weighting.

        ``r = sqrt(angle^2 + (velocity_weight * velocity)^2)``. ``velocity_weight``
        defaults to 1.0 (symmetric historical behavior).

        :param SensorSignal signal: angle estimation in radians and quarature component velocity surrogate in radians per sec.
        :param float velocity_weight: Weight applied to the velocity term before squaring.

        :return: The portrait radius.
        :rtype: float
        """
        return np.sqrt(
            signal.angle_rad**2 + (velocity_weight * signal.velocity_rad_per_sec) ** 2
        )

    def compute_amplitude(self, signal: SensorSignal) -> float:
        """Calculate the amplitude modulation factor based on the current signal and mode parameters.

        :param SensorSignal signal: angle estimation in radians and quarature component velocity surrogate in radians per sec.
        :return: Amplitude modulation factor.
        :rtype: float
        """
        params = self._mode.get_parameters()

        portrait_radius = self._compute_portrait_radius(
            signal=signal, velocity_weight=params.velocity_weight
        )
        scaled_portrait_radius = portrait_radius * params.scale

        sigmoid_raw = self.apply_sigmoid_scaling(
            value=scaled_portrait_radius, power=params.sigmoid_power
        )

        # Smooth the sigmoid gate so engagement requires sustained motion.
        # Uses the SensorSignal timestamp to compute dt; falls back to a
        # 100 Hz tick if timestamps are missing or unreasonable.
        ts = signal.timestamp
        if ts is not None and self._prev_timestamp is not None:
            dt = ts - self._prev_timestamp
            if dt <= 0.0 or dt > 1.0:
                dt = SIGMOID_LPF_DT_FALLBACK
        else:
            dt = SIGMOID_LPF_DT_FALLBACK
        if ts is not None:
            self._prev_timestamp = ts
        sigmoid_smoothed, _ = self._sigmoid_lpf.step(
            x=float(sigmoid_raw), time_difference=dt
        )
        # The LPF can overshoot slightly outside [0, 1]; clip back so the
        # rest of the pipeline behaves like the un-smoothed sigmoid.
        sigmoid_scaling = max(0.0, min(1.0, sigmoid_smoothed))

        scaled_sigmoid_scaling = sigmoid_scaling * params.gain
        amplitude = scaled_sigmoid_scaling * self.reverse_amplitude

        self.last_intermediates = AmplitudeIntermediates(
            portrait_radius=portrait_radius,
            scaled_portrait_radius=scaled_portrait_radius,
            sigmoid_scaling=sigmoid_scaling,
            scaled_sigmoid_scaling=scaled_sigmoid_scaling,
            amplitude=amplitude,
        )
        return amplitude

    def reset(self) -> None:
        """Reset LPF state and last-sample cache. Call on session start."""
        self._sigmoid_lpf.reset()
        self._prev_timestamp = None
        self.last_intermediates = None

    @staticmethod
    def apply_sigmoid_scaling(value: float, power: int) -> float:
        """Apply sigmoid scaling a^n / (a^n + 1) so that the higher n is, the lower amplitudes are scaled down.

        Numerically stable implementation that handles overflow by clamping large exponents.

        :param float value: Variable a.
        :param float power: Variable n.

        :return: Amplitude in range [0, 1].
        :rtype: float
        """
        with np.errstate(over="raise"):
            try:
                exponent = np.float64(value) ** power
                # If exponent is too large, return value close to 1 (saturated sigmoid)
                if np.isinf(exponent):
                    return 1.0
                elif np.isnan(exponent):
                    return 1.0
                else:
                    result = exponent / (exponent + 1.0)
                    return result
            except FloatingPointError:
                # Fallback for edge cases
                logger.warning(f"Exponent = {value} ** {power}")
                return 1.0

    def fake_sigmoid_scaling(self, value: float) -> float:
        """Substitute for apply_sigmoid_scaling, value=scaled_portrait_radius, power=params.sigmoid_power."""
        if value > 1.2:
            return 1.0
        elif value < 0.8:
            return 0.0
        else:
            return (value**50.0) / (value**50.0 + 1.0)
