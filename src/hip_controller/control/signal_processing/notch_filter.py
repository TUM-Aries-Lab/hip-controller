"""Notch filtering utilities for drift rejection.

This module provides a configurable notch (or DC blocker) filter implementation.
"""

from typing import TYPE_CHECKING

import numpy as np
from scipy import signal

if TYPE_CHECKING:
    from hip_controller.definitions import NotchConfig


class NotchFilter:
    """A step-by-step notch filter used for drift rejection.

    Implements a configurable notch filter that can act as a DC blocker
    (when center_freq_hz=0) or a standard notch filter.
    """

    def __init__(self, config: "NotchConfig") -> None:
        """Initialize the notch filter with configuration.

        :param NotchConfig config: Notch filter configuration.
        :return: None
        """
        self._config = config

        # Precompute filter coefficients and initial state
        if self._config.center_freq_hz == 0:
            # DC blocker: high-pass first-order IIR  y[n] = x[n] - x[n-1] + r*y[n-1]
            # r controls how close the pole is to the zero at DC.
            # r = 1 - (pi * BW / Fs) mirrors the same BW formula used by iirnotch.
            pole_radius = 1.0 - (
                np.pi * self._config.bandwidth_3db_hz / self._config.sample_rate_hz
            )
            self._b = np.array([1.0, -1.0])
            self._a = np.array([1.0, -pole_radius])
        else:
            quality_factor = self._config.center_freq_hz / self._config.bandwidth_3db_hz
            self._b, self._a = signal.iirnotch(
                self._config.center_freq_hz, quality_factor, self._config.sample_rate_hz
            )

        # Initial filter state
        self._zi = signal.lfilter_zi(self._b, self._a)

    def filter(self, raw_value: float) -> float:
        """Process one raw sample through the notch filter.

        :param float raw_value: Raw sensor sample.
        :return: Filtered sample.
        :rtype: float
        """
        y, self._zi = signal.lfilter(self._b, self._a, [raw_value], zi=self._zi)
        return float(y[0])

    def reset(self) -> None:
        """Reset the filter state to initial conditions."""
        self._zi = signal.lfilter_zi(self._b, self._a)
