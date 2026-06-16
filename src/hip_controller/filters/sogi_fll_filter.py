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

# Number of samples (at the loop rate) after each ``start_walking()`` during
# which the FLL adaptation is skipped. The SOGI oscillator state is also
# zeroed at start_walking, so for the first ~0.8 s the SOGI rebuilds its
# in-phase / quadrature components from zero. During the rebuild, ``energy``
# is tiny but above ``lower_energy_threshold``, ``lock`` is small but
# non-zero, and the ratio ``(phase_error * quadrature) / energy`` is
# ill-conditioned -- the FLL would drift down by ~0.25 Hz before things
# stabilize. Holding adaptation off for this brief window keeps the FLL
# tuned to the preserved pre-pause frequency until the SOGI is locked, at
# which point the next stride's clean signal lets the FLL adapt to the
# user's actual cadence within a single cycle.
POST_RESUME_FLL_COOLDOWN_TICKS = 80

# Number of samples after each set_config() (i.e. every locomotion-mode swap)
# during which the FLL adaptation is held off. Unlike start_walking() the SOGI
# state is intentionally preserved across mode swaps, but the in-phase /
# quadrature components accumulated during the previous mode carry harmonic
# content tuned to that gait. Right after a swap the SOGI core still needs a
# few hundred ms to re-lock at the new k_sogi / fmin / fmax, and during that
# window the (phase_error * quadrature) / energy term can point the wrong way
# (observed empirically on DESCEND -> LEVEL, where the FLL drifted down ~50 mHz
# over 1.5 s before recovering). Holding adaptation off briefly lets the SOGI
# re-lock first, then the FLL adapts on a clean signal.
POST_MODE_SWITCH_FLL_COOLDOWN_TICKS = 50


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

        # FLL adaptation cooldown counter. Set to POST_RESUME_FLL_COOLDOWN_TICKS
        # on every start_walking() so the FLL holds its frequency estimate
        # for the duration of the SOGI rebuild. See the constant's docstring.
        self._fll_cooldown_ticks: int = 0

    def stop_walking(self) -> None:
        """Transition filter to non-walking mode (decay instead of tracking)."""
        self._walking = False

    def set_config(self, config: SogiFllConfig) -> None:
        """Swap the SOGI-FLL parameter set (used for per-locomotion-mode tuning).

        Only the configuration is replaced; the SOGI's state (in-phase,
        quadrature, omega_est, frequency_estimate, confidence_state) is
        preserved. The new bounds and gains take effect on the next sample.
        Crossing a mode boundary mid-walk therefore reuses the existing
        FLL lock rather than restarting from `initial_frequency_guess`.

        FLL adaptation is gated for ``POST_MODE_SWITCH_FLL_COOLDOWN_TICKS``
        samples after the swap so the SOGI core can re-lock at the new
        gains/bounds before the FLL starts chasing the new cadence. We
        ``max(...)`` rather than overwrite so an in-progress longer cooldown
        (e.g. POST_RESUME from a recent start_walking) isn't shortened.
        """
        self._config = config
        self._fll_cooldown_ticks = max(
            self._fll_cooldown_ticks, POST_MODE_SWITCH_FLL_COOLDOWN_TICKS
        )

    def start_walking(self) -> None:
        """Transition filter back to walking mode (resume tracking).

        Mirror of :meth:`stop_walking`. The frequency estimate is preserved
        (so the FLL re-engages already tuned to the user's cadence) but the
        SOGI oscillator state and confidence state are cleared. The FLL
        adaptation is also held off for ``POST_RESUME_FLL_COOLDOWN_TICKS``
        samples while the SOGI rebuilds -- otherwise the ill-conditioned
        ``(phase_error * vb) / energy`` term during rebuild aggressively
        drags the frequency estimate down. Once the cooldown elapses and
        the SOGI is locked, the FLL adapts on a clean, properly-phased
        signal and converges to the user's actual cadence within a cycle.
        """
        self._walking = True
        self._inphase = 0.0
        self._quadrature = 0.0
        self._confidence_state = 0.0
        self._fll_cooldown_ticks = POST_RESUME_FLL_COOLDOWN_TICKS

    @property
    def is_walking(self) -> bool:
        """True when SOGI/FLL is actively tracking; False during paused decay."""
        return self._walking

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
        if self._walking and self._fll_cooldown_ticks == 0:
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
        elif self._walking:
            # In post-resume cooldown: SOGI updates were already applied above,
            # but the FLL adaptation is held off so the frequency estimate
            # stays at its preserved pre-pause value while the SOGI rebuilds.
            self._fll_cooldown_ticks -= 1
        else:
            # FREEZE the frequency estimate during pause. The original
            # behavior here was a slow drift toward ``initial_frequency_guess``
            # (0.995 * current + 0.005 * f_init), which over multi-second
            # pauses pulled the FLL away from the user's actual cadence and
            # caused a mistuned re-lock on resume. With this freeze, the FLL
            # comes back online tuned to whatever frequency the user was
            # walking at just before they stopped.
            pass

        return inphase_output, quadrature_output

    def reset(self) -> None:
        """Reset the filter to a known initial condition."""
        # va, vb
        self._inphase: float = 0.0
        self._quadrature: float = 0.0

        # w_est, f_state, lock_state
        self._omega_est: float = 2.0 * pi * self._config.initial_frequency_guess
        self._frequency_estimate: float = self._config.initial_frequency_guess
        self._confidence_state: float = 0.0

        # walking + cooldown
        self._walking: bool = True
        self._fll_cooldown_ticks: int = 0

    @property
    def estimated_frequency_hz(self) -> float:
        """Current FLL frequency estimate [Hz]."""
        return self._frequency_estimate
