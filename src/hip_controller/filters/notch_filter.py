"""Notch filtering utilities for drift rejection.

Notch-peak filter implementation following the MATLAB DSP Toolbox algorithm. The notch function from scipy.signal.iirnotch is not used here, because it is incompatible with 0 Hz center frequency. As a result, implementation of the matlab design equations is used instead.
Reference: https://www.mathworks.com/help/dsp/ref/notchpeakfilter.html
"""

from math import cos, pi, tan

import numpy as np
from scipy.signal import lfilter_zi

from hip_controller.definitions import NotchConfig


class NotchFilter:
    """Second-order IIR notch filter following the MATLAB Notch-Peak Filter algorithm.

    Implements the notch (and complementary peak) transfer functions::

        b  = 1 / (1 + tan(Δω/2))
        a1 = -2b·cos(ω0)
        a2 =  2b - 1

        Notch: H(z) = b·(1 - 2cos(ω0)z⁻¹ + z⁻²) / (1 + a1·z⁻¹ + a2·z⁻²)
        Peak:  H(z) = (1-b)·(1 - z⁻²)            / (1 + a1·z⁻¹ + a2·z⁻²)

    The two outputs are complementary: ``notch + peak = input`` at every sample.

    The filter is implemented as a Direct Form II Transposed biquad, which is
    numerically more stable than Direct Form I for fixed-point-like arithmetic.
    State is initialised lazily on the first call to steady-state conditions
    matching the first input sample, so the output starts at the correct value
    rather than ringing from zero.

    :param NotchConfig config: Notch filter configuration.
    """

    def __init__(self, config: NotchConfig) -> None:
        self._config = config
        self._b_notch, self._b_peak, self._a = self._compute_coefficients()

        # State vectors [z1, z2] for the shared denominator (Direct Form II Transposed)
        # None = uninitialised; set on first call using the first sample value.
        self._z: np.ndarray | None = None

    def filter(self, raw_value: float) -> float:
        """Filter one sample and return the notch output.

        :param float raw_value: Raw input sample.
        :return: Notch-filtered sample.
        :rtype: float
        """
        notch, _ = self._step(raw_value)
        return notch

    def filter_peak(self, raw_value: float) -> float:
        """Filter one sample and return the peak output.

        :param float raw_value: Raw input sample.
        :return: Peak-filtered sample (complementary to notch).
        :rtype: float
        """
        _, peak = self._step(raw_value)
        return peak

    def filter_both(self, raw_value: float) -> tuple[float, float]:
        """Filter one sample and return both notch and peak outputs.

        :param float raw_value: Raw input sample.
        :return: Tuple of ``(notch, peak)``.
        :rtype: tuple[float, float]
        """
        return self._step(raw_value)

    def filter_array(self, values: np.ndarray) -> np.ndarray:
        """Filter an array of samples sequentially, returning the notch output.

        The internal state is preserved across calls, so this is equivalent
        to calling :meth:`filter` in a loop.

        :param np.ndarray values: 1-D array of input samples.
        :return: 1-D array of notch-filtered samples.
        :rtype: np.ndarray
        """
        return np.array([self.filter(float(x)) for x in values])

    def reset(self) -> None:
        """Reset the filter to uninitialised state.

        The next call to :meth:`filter` will reinitialise the state to
        steady-state conditions for the new first sample.
        """
        self._z = None

    def _compute_coefficients(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute biquad coefficients from the MATLAB design equations.

        :return: Tuple of ``(b_notch, b_peak, a)`` coefficient arrays,
            each of length 3. The shared denominator ``a`` has ``a[0] = 1``.
        """
        w0 = 2.0 * pi * self._config.center_freq_hz / self._config.sample_rate_hz
        dw = 2.0 * pi * self._config.bandwidth_3db_hz / self._config.sample_rate_hz

        b = 1.0 / (1.0 + tan(dw / 2.0))  # bandwidth coefficient
        a1 = -2.0 * b * cos(w0)  # center frequency term
        a2 = 2.0 * b - 1.0  # combined bandwidth/frequency term

        # Notch numerator:  b · [1, -2cos(w0), 1]
        b_notch = np.array([b, -2.0 * b * cos(w0), b])
        # Peak numerator:   (1-b) · [1, 0, -1]
        b_peak = np.array([1.0 - b, 0.0, -(1.0 - b)])
        # Shared denominator: [1, a1, a2]
        a = np.array([1.0, a1, a2])

        return b_notch, b_peak, a

    def _step(self, x: float) -> tuple[float, float]:
        """Run one biquad step for both notch and peak.

        Uses Direct Form II Transposed recurrences::

            y    = b0·x + z1
            z1   = b1·x - a1·y + z2
            z2   = b2·x - a2·y
        """
        if self._z is None:
            """Initialise state to steady-state conditions for a constant input x0.

        For a constant input the output equals ``H(1) · x0``.  The notch has
        unity DC gain (H_notch(1) = 1) and the peak has zero DC gain
        (H_peak(1) = 0), so the steady-state notch output is x0.

        Using Direct Form II Transposed steady-state equations::

            y_ss  = H(1) · x0
            z1_ss = (b1 - a1·b0) · x0 / (1 + a1 + a2)  ... (solved from recurrence)

        In practice this is computed as: z = lfilter_zi(b, a) · x0, which is
        exactly what scipy does internally — but we call it only once here.
        """
            # Compute unit-step steady-state coefficients and scale by x0
            zi_notch = lfilter_zi(self._b_notch, self._a) * x
            # Store combined state: [notch_z1, notch_z2, peak_z1, peak_z2]
            zi_peak = lfilter_zi(self._b_peak, self._a) * x
            self._z = np.concatenate([zi_notch, zi_peak])

        b0n, b1n, b2n = self._b_notch
        b0p, b1p, b2p = self._b_peak
        _, a1, a2 = self._a  # a0 = 1, kept for clarity
        zn1, zn2, zp1, zp2 = self._z

        # Notch
        yn = b0n * x + zn1
        zn1 = b1n * x - a1 * yn + zn2
        zn2 = b2n * x - a2 * yn

        # Peak — shares the same denominator poles
        yp = b0p * x + zp1
        zp1 = b1p * x - a1 * yp + zp2
        zp2 = b2p * x - a2 * yp

        self._z = np.array([zn1, zn2, zp1, zp2])
        return float(yn), float(yp)
