"""The function implements a SOGI-FLL oscillator that locks onto the dominant oscillation in the input angle signal.

It does three main things:
    1.Filters the input angle into a clean sinusoidal component.
    2. Creates a quadrature signal (90° shifted) -> velocity.
    3. Adapts the internal frequency to match the cadence of the motion.
So effectively it behaves like a self-tuning oscillator that synchronizes to the gait signal.
"""

from math import exp, pi

from numpy import clip

from hip_controller.definitions import SogiFllConfig


class SogiFllFilter:
    """Second-Order Generalized Integrator which generates in-phase + quadrature, with Frequency-Locked Loop which (adapts internal frequency to cadence changes (SOGI-FLL)."""

    def __init__(self, config: SogiFllConfig) -> None:
        """Initialize the sogi fill filter.

        :param SogiFllConfig config: Configurations  of SOGI-FLL filter, including fixed and rarely changed parameters.
        :return: None
        """
        self._config: SogiFllConfig = config

        self._walking: bool = True
        # Persistent states
        self._va: float = 0.0
        self._vb: float = 0.0
        self._w_est: float = 2.0 * pi * self._config.initial_frequency_guess
        self._f_state: float = self._config.initial_frequency_guess
        self._lock_state: float = 0.0

    def stop_walking(self) -> None:
        """Stop walking.

        :return: None
        """
        self._walking = False

    def filter(self, theta: float, time_difference: float) -> tuple[float, float]:
        """Filter the inpit angle of one step into a clean sinusoidal component and return the surrogate angle and quadrature angular velocity.

        :param float theta: Drift-removed joint angle.
        :param time_difference: Sample period / elapsed time since last call [s].

        :return: Tuple of ``(theta, theta_quad)`` — in-phase filtered surrogate angle and 90°-shifted quadrature angular velocity.
        """
        w_min = 2.0 * pi * self._config.lower_cadence_bound
        w_max = 2.0 * pi * self._config.upper_cadence_bound

        # Smoothing coefficients (recomputed each call so dt changes are safe)
        a_f = exp(
            -2.0
            * pi
            * self._config.frequency_estimate_smoother_bandwidth
            * time_difference
        )
        a_lock = exp(
            -2.0 * pi * self._config.lock_state_smoother_bandwidth * time_difference
        )

        w0 = clip(a=self._w_est, a_min=w_min, a_max=w_max)

        # ---- SOGI --------------------------------------------------------
        if self._walking:
            e = theta - self._va
            self._va += time_difference * (
                w0 * self._vb + self._config.sogi_adaptation_gain * w0 * e
            )
            self._vb += time_difference * (-w0 * self._va)
        else:
            self._va *= self._config.decay_not_walking
            self._vb *= self._config.decay_not_walking
            e = 0.0

        theta = self._va
        theta_quad = self._vb

        # ---- Lock / confidence -------------------------------------------
        E = theta * theta + theta_quad * theta_quad
        self._lock_state = a_lock * self._lock_state + (1.0 - a_lock) * E
        lock = clip(
            a=(self._lock_state - self._config.lower_energy_threshold)
            / (
                self._config.upper_energy_threshold
                - self._config.lower_energy_threshold
                + self._config.numerical_safety_floor
            ),
            a_min=0.0,
            a_max=1.0,
        )

        # ---- FLL ---------------------------------------------------------
        if self._walking:
            mu = (e * theta_quad) / (E + self._config.numerical_safety_floor)
            self._w_est = (
                w0
                + time_difference * (self._config.fll_adaptation_gain * w0 * mu) * lock
            )
            self._w_est = clip(a=self._w_est, a_min=w_min, a_max=w_max)

            f_raw = self._w_est / (2.0 * pi)
            self._f_state = a_f * self._f_state + (1.0 - a_f) * f_raw
            self._w_est = (
                2.0
                * pi
                * clip(
                    a=self._f_state,
                    a_min=self._config.lower_cadence_bound,
                    a_max=self._config.upper_cadence_bound,
                )
            )
        else:
            self._f_state = (
                0.995 * self._f_state + 0.005 * self._config.initial_frequency_guess
            )
            self._w_est = 2.0 * pi * self._f_state

        return theta, theta_quad

    @property
    def estimated_frequency_hz(self) -> float:
        """Current FLL frequency estimate [Hz] — useful for monitoring/logging."""
        return self._f_state
