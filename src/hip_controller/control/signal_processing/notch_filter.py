"""Notch filtering utilities for drift rejection.

This module provides a configurable notch (or DC blocker) filter implementation.
"""

import numpy as np
from scipy import signal

from hip_controller.definitions import NotchConfig


class NotchFilter:
    """A step-by-step notch filter used for drift rejection."""

    def __init__(self, config: NotchConfig) -> None:
        """Initialize the notch filter with configuration.

        :param NotchConfig config: Notch filter configuration.
        """
        self._config: NotchConfig = config

        if config.center_freq_hz == 0:
            pole_radius = 1.0 - (
                np.pi * config.bandwidth_3db_hz / config.sample_rate_hz
            )
            self._b = np.array([1.0, -1.0])
            self._a = np.array([1.0, -pole_radius])
        else:
            quality_factor = config.center_freq_hz / config.bandwidth_3db_hz
            self._b, self._a = signal.iirnotch(
                w0=config.center_freq_hz, Q=quality_factor, fs=config.sample_rate_hz
            )

        self._zi_base = signal.lfilter_zi(
            self._b, self._a
        )  # unit-step zi — scaled on first call
        self._zi: np.ndarray | None = None  # None = not yet initialised

    def filter(self, raw_value: float) -> float:
        """Process one raw sample through the notch filter.

        .. note::
            Filter coefficients are fixed at construction from ``sample_rate_hz``.
            The method signature is stateless with respect to ``time_difference`` —
            call at a consistent rate matching ``sample_rate_hz``.

        :param float raw_value: Raw sensor sample.
        :return: Filtered sample.
        :rtype: float
        """
        if self._zi is None:
            # Scale the unit-step zi by the first actual input so the filter
            # starts in steady-state for that value, avoiding the initial transient.
            self._zi = self._zi_base * raw_value

        y, self._zi = signal.lfilter(b=self._b, a=self._a, x=[raw_value], zi=self._zi)
        return float(y[0])

    def reset(self) -> None:
        """Reset the filter state — next call re-initialises from the first sample."""
        self._zi = None
