"""Persistent-state SOGI-FLL.

Compute surrogate angle and quadrature velocity. The FLL continuously
adapts the internal resonant frequency to track cadence changes within the
supplied bounds, while the lock signal gates adaptation so that frequency
updates are suppressed when the input energy is too low (standing still) or
too high (artefacts).
"""

from dataclasses import dataclass
from math import exp, pi

from numpy import clip


@dataclass(frozen=True)
class SogiFllConfig:
    """SOGI-FLL parameter set.

    :param float lower_cadence_bound: f_min
    :param float upper_cadence_bound: f_max
    :param float sogi_adaptation_gain: k_sogi
    :param float fll_adaptation_gain: k_fll
    :param float lower_energy_threshold: E_lo
    :param float upper_energy_threshold: E_hi
    :param float frequency_estimate_smoother_bandwidth: fc_f_smooth
    :param float lock_state_smoother_bandwidth: fc_lock
    :param float initial_frequency_guess: f_init
    :param float decay_not_walking: decay_notwalking
    :param float numerical_safety_floor: epsSmall
    """

    # cadence bounds (walking/running range)
    lower_cadence_bound: float = 0.2
    upper_cadence_bound: float = 4.0

    # FLL adaptation speed (sensor/noise dependent)
    sogi_adaptation_gain: float = 1.0
    fll_adaptation_gain: float = 1.0

    # lock thresholds (amplitude/noise dependent)
    lower_energy_threshold: float = 0.0001
    upper_energy_threshold: float = 0.01

    # Tune only if internal frequency becomes jittery or too laggy:
    # - decrease to 0.2 for smoother (more lag)
    # - increase to 0.5 for faster (more jitter)
    frequency_estimate_smoother_bandwidth: float = 0.30

    # Tune only if lock flickers or reacts too slowly:
    # - decrease (0.3) to reduce flicker
    # - increase (0.8-1.0) for faster start/stop response
    lock_state_smoother_bandwidth: float = 0.50

    # [Hz] initial guess (walking/running general default)
    # Tune only if you want faster lock at startup:
    # - set near typical cadence in your trials (walk ~1-2 Hz, run ~2-3 Hz)
    initial_frequency_guess: float = 1.4

    # % state decay when standing
    # Tune only if oscillator rings too long after stopping:
    # - faster decay: 0.995
    # - slower decay: 0.9995
    decay_not_walking: float = 0.999

    # numerical safety
    # Increase only if you see NaN/Inf in extreme low-motion segments (e.g., 1e-8)
    numerical_safety_floor: float = 1e-9


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

    def filter(self, theta: float, dt: float) -> tuple[float, float]:
        """Run one SOGI-FLL step and return the in-phase and quadrature signals.

        :param float theta: Drift-removed joint angle.
        :param dt: Sample period / elapsed time since last call [s].

        :return: Tuple of ``(theta, theta_quad)`` — in-phase filtered surrogate angle and 90°-shifted quadrature angular velocity.
        """
        w_min = 2.0 * pi * self._config.lower_cadence_bound
        w_max = 2.0 * pi * self._config.upper_cadence_bound

        # Smoothing coefficients (recomputed each call so dt changes are safe)
        a_f = exp(-2.0 * pi * self._config.frequency_estimate_smoother_bandwidth * dt)
        a_lock = exp(-2.0 * pi * self._config.lock_state_smoother_bandwidth * dt)

        w0 = clip(a=self._w_est, a_min=w_min, a_max=w_max)

        # ---- SOGI --------------------------------------------------------
        if self._walking:
            e = theta - self._va
            self._va += dt * (
                w0 * self._vb + self._config.sogi_adaptation_gain * w0 * e
            )
            self._vb += dt * (-w0 * self._va)
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
            self._w_est = w0 + dt * (self._config.fll_adaptation_gain * w0 * mu) * lock
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
