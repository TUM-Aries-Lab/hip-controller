"""Amplitude modulation to eliminate too much motor activity due to small sensor movements."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from hip_controller.definitions import (
    AMPLITUDE_GAIN,
    SCALE_LEVEL_MODE,
    SIGMOID_POWER,
    SensorSignal,
)
from hip_controller.utils.math_utils import apply_sigmoid_scaling


@dataclass
class ModeParameters:
    """Dataclass for mode parameters."""

    scale: float
    sigmoid_power: float
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
            self.reverse_amplitude = -1
        else:
            self.reverse_amplitude = 1

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

        amplitude = apply_sigmoid_scaling(scaled_portrait_radius, params.sigmoid_power)

        return (amplitude * params.gain) * self.reverse_amplitude
