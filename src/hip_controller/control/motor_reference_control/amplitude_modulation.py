"""Amplitude modulation to eliminate too much motor activity due to small sensor movements."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from loguru import logger

from hip_controller.definitions import (
    AMPLITUDE_GAIN,
    SCALE_LEVEL_MODE,
    SIGMOID_POWER,
    SensorSignal,
)


@dataclass
class ModeParameters:
    """Dataclass for mode parameters."""

    scale: float
    sigmoid_power: int
    gain: float


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
            scale=SCALE_LEVEL_MODE, sigmoid_power=SIGMOID_POWER, gain=AMPLITUDE_GAIN
        )


# Parameters of ascending and descending stairs mode are not certain yet
class AscendStairsMode(ModeStrategy):
    """Mode for ascending stairs."""

    def get_parameters(self) -> ModeParameters:
        """Get parameters for ascending stairs."""
        return ModeParameters(
            scale=SCALE_LEVEL_MODE - 0.6,
            sigmoid_power=SIGMOID_POWER + 100,
            gain=AMPLITUDE_GAIN - 2,
        )


class DescendStairsMode(ModeStrategy):
    """Mode for descending stairs."""

    def get_parameters(self) -> ModeParameters:
        """Get parameters for descending stairs."""
        return ModeParameters(
            scale=SCALE_LEVEL_MODE - 0.5,
            sigmoid_power=SIGMOID_POWER + 100,
            gain=AMPLITUDE_GAIN + 0.5,
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

    def set_mode(self, mode: ModeStrategy):
        """Switch mode at runtime."""
        self._mode = mode

    @staticmethod
    def _compute_portrait_radius(signal: SensorSignal) -> float:
        """Calculate the portrait radius.

        :param SensorSignal signal: angle estimation in radians and quarature component velocity surrogate in radians per sec.

        :return: The portrait radius.
        :rtype: float
        """
        return np.sqrt(signal.angle_rad**2 + signal.velocity_rad_per_sec**2)

    def compute_amplitude(self, signal: SensorSignal) -> float:
        """Calculate the amplitude modulation factor based on the current signal and mode parameters.

        :param SensorSignal signal: angle estimation in radians and quarature component velocity surrogate in radians per sec.
        :return: Amplitude modulation factor.
        :rtype: float
        """
        params = self._mode.get_parameters()

        scaled_portrait_radius = (
            self._compute_portrait_radius(signal=signal) * params.scale
        )

        amplitude = self.apply_sigmoid_scaling(
            value=scaled_portrait_radius, power=params.sigmoid_power
        )
        return (amplitude * params.gain) * self.reverse_amplitude

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
