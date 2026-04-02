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
    """Second-Order Generalized Integrator (SOGI) + Frequency-Locked Loop (FLL).

    Produces in-phase and quadrature signals and tracks cadence-derived frequency.
    """

    def __init__(self, config: SogiFllConfig) -> None:
        """Initialize SOGI-FLL filter with configuration parameters.

        :param SogiFllConfig config: SOGI-FLL controller tuning parameters.
        :return: None
        """
        self._config: SogiFllConfig = config

        self._walking: bool = True

        # Persistent internal state (SOGI outputs)
        self._inphase: float = 0.0
        self._quadrature: float = 0.0

        # Estimated angular freq and smoothed frequency state
        self._omega_est: float = 2.0 * pi * self._config.initial_frequency_guess
        self._frequency_estimate: float = self._config.initial_frequency_guess

        # Lock confidence state (energy-based)
        self._confidence_state: float = 0.0

    def stop_walking(self) -> None:
        """Transition filter to non-walking mode (decay instead of tracking)."""
        self._walking = False

    def filter(
        self, raw_theta_rad: float, time_difference: float
    ) -> tuple[float, float]:
        """Process a single sample.

        :param float theta: Drift-corrected joint angle input.
        :param float time_difference: Time delta since last sample in seconds.

        :return: Tuple of (inphase_output, quadrature_output).
        """
        w_min = 2.0 * pi * self._config.lower_cadence_bound
        w_max = 2.0 * pi * self._config.upper_cadence_bound

        # Smoothed frequency updates
        alpha_frequency = exp(
            -2.0
            * pi
            * self._config.frequency_estimate_smoother_bandwidth
            * time_difference
        )
        alpha_confidence = exp(
            -2.0 * pi * self._config.lock_state_smoother_bandwidth * time_difference
        )

        omega_clipped = clip(a=self._omega_est, a_min=w_min, a_max=w_max)

        # ---- SOGI core update ------------------------------------------------
        if self._walking:
            phase_error = raw_theta_rad - self._inphase
            self._inphase += time_difference * (
                omega_clipped * self._quadrature
                + self._config.sogi_adaptation_gain * omega_clipped * phase_error
            )
            self._quadrature += time_difference * (-omega_clipped * self._inphase)
        else:
            self._inphase *= self._config.decay_not_walking
            self._quadrature *= self._config.decay_not_walking
            phase_error = 0.0

        inphase_output = self._inphase
        quadrature_output = self._quadrature

        # ---- energy-based lock confidence ---------------------------------------
        energy = inphase_output * inphase_output + quadrature_output * quadrature_output
        self._confidence_state = (
            alpha_confidence * self._confidence_state
            + (1.0 - alpha_confidence) * energy
        )
        confidence = clip(
            a=(self._confidence_state - self._config.lower_energy_threshold)
            / (
                self._config.upper_energy_threshold
                - self._config.lower_energy_threshold
                + self._config.numerical_safety_floor
            ),
            a_min=0.0,
            a_max=1.0,
        )

        # ---- FLL frequency adaptation -------------------------------------------
        if self._walking:
            adaptation_error = (phase_error * quadrature_output) / (
                energy + self._config.numerical_safety_floor
            )
            self._omega_est = (
                omega_clipped
                + time_difference
                * (self._config.fll_adaptation_gain * omega_clipped * adaptation_error)
                * confidence
            )
            self._omega_est = clip(a=self._omega_est, a_min=w_min, a_max=w_max)

            raw_frequency = self._omega_est / (2.0 * pi)
            self._frequency_estimate = (
                alpha_frequency * self._frequency_estimate
                + (1.0 - alpha_frequency) * raw_frequency
            )
            self._omega_est = (
                2.0
                * pi
                * clip(
                    a=self._frequency_estimate,
                    a_min=self._config.lower_cadence_bound,
                    a_max=self._config.upper_cadence_bound,
                )
            )
        else:
            self._frequency_estimate = (
                0.995 * self._frequency_estimate
                + 0.005 * self._config.initial_frequency_guess
            )
            self._omega_est = 2.0 * pi * self._frequency_estimate

        return inphase_output, quadrature_output

    @property
    def estimated_frequency_hz(self) -> float:
        """Current FLL frequency estimate [Hz]."""
        return self._frequency_estimate
