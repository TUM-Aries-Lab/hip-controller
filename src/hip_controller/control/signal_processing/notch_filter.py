"""Notch filtering utilities for drift rejection.

This module provides a configurable notch (or DC blocker) filter implementation.
"""

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class NotchConfig:
    """Configurations for the notch function."""

    center_freq_hz = 0
    bandwidth_3db_hz = 0.1
    sample_rate_hz = 100


class NotchFilter:
    """A step-by-step notch filter used for drift rejection."""

    def make_notch_filter(self, notch_config: NotchConfig):
        """Build notch filter coefficients and initial conditions.

        :param NotchConfig notch_config: Notch filter configuration.
        :return: Numerator, denominator, and initial state.
        :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        """
        if notch_config.center_freq_hz == 0:
            # DC blocker: high-pass first-order IIR  y[n] = x[n] - x[n-1] + r*y[n-1]
            # r controls how close the pole is to the zero at DC.
            # r = 1 - (pi * BW / Fs) mirrors the same BW formula used by iirnotch.
            pole_radius = 1.0 - (
                np.pi * notch_config.bandwidth_3db_hz / notch_config.sample_rate_hz
            )
            numerator_coeffs_b = np.array([1.0, -1.0])
            denominator_coeffs_a = np.array([1.0, -pole_radius])
        else:
            quality_factor = notch_config.center_freq_hz / notch_config.bandwidth_3db_hz
            numerator_coeffs_b, denominator_coeffs_a = signal.iirnotch(
                notch_config.center_freq_hz, quality_factor, notch_config.sample_rate_hz
            )

        zi = signal.lfilter_zi(numerator_coeffs_b, denominator_coeffs_a)
        return numerator_coeffs_b, denominator_coeffs_a, zi

    def filter(self, raw_value: float) -> float:
        """Process one raw sample through the notch filter.

        :param float raw_value: Raw sensor sample.
        :return: Filtered sample.
        :rtype: float
        """
        b, a, zi = self.make_notch_filter(notch_config=NotchConfig())
        y, zi = signal.lfilter(b, a, [raw_value], zi=zi)
        return float(y[0])
