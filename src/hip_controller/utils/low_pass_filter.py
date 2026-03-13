"""Second-order low-pass filter — Strategy pattern for integration.

Simulink signal flow (from the labelled block diagram):
───────────────────────────────────────────────────────
  1. e1       = x - y                    (1st summer, y fed back from port 1)
  2. feedback = 2 * zt * q               (bottom multiplier, q = 1st integrator output)
  3. e2       = e1 - feedback            (2nd summer)
  4. wn_e2    = wn * e2                  (multiply block)
  5. q        = integral(wn_e2), IC=x0   [INTERNAL STATE — 1st integrator]
  6. yd       = wn * q                   [BETWEEN-STEP WIRE — port 2, NOT a stored state]
  7. y        = integral(yd),    IC=x0   [port 1 — 2nd integrator]
  8. y fed back to 1st summer (minus input)

State equations (continuous):
  dq/dt = wn * (x - y - 2*zt*q)
  dy/dt = wn * q  =  yd

Transfer function:
  H(s) = wn^2 / (s^2 + 2*zt*wn*s + wn^2)

Public API
──────────
  step(x, dt=None)
      One sample at a time. dt overrides cfg.dt for this step only.
      Use this in a real-time loop where the interval may vary each cycle.

  run(x_array, dt_array=None)
      Whole sequence at once; resets state first.
      dt_array may be a matching array of per-sample intervals, a single
      float to use uniformly, or None to fall back to cfg.dt every step.

  reset()
      Return both integrators to x0.

The solver strategy (Forward Euler / Backward Euler / Trapezoidal / RK4)
is selected via FilterDefinitions.solver_type and injected at construction.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from hip_controller.definitions import FilterConfig
from hip_controller.utils.low_pass_filter_solvers import (
    SolverStrategy,
    make_solver,
)

# ---------------------------------------------------------------------------
# State container — only the two integrator outputs are stored
# ---------------------------------------------------------------------------


@dataclass
class LowPassFilterState:
    """Internal states of the two Simulink integrators.

    Attributes
    ----------
    q : float
        1st integrator output — INTERNAL, never exposed as an output port.
        Satisfies  dq/dt = wn * (x - y - 2*zt*q).
        Used to compute yd and as the bottom-multiplier feedback signal.
    y : float
        2nd integrator output — Simulink output port 1.
        Satisfies  dy/dt = wn * q = yd.
        Also fed back to the 1st summer as the minus input.

    """

    q: float = 0.0  # 1st integrator state  (init from x0)
    y: float = 0.0  # 2nd integrator state  (init from x0)


# ---------------------------------------------------------------------------
# Filter class
# ---------------------------------------------------------------------------


class SecondOrderLPF:
    """Second-order low-pass filter with pluggable integration strategy.

    The filter owns:
        - Configuration  : FilterDefinitions (wn, zt, x0, dt, solver_type)
        - Internal state : LowPassFilterState (q, y)
        - Solver         : SolverStrategy instance (injected via solver_type)

    The solver owns nothing — it only defines HOW states advance each step.
    Swap solvers by changing cfg.solver_type; no other code changes.

    Usage — fixed timestep (real-time loop)
    ---------------------------------------
        cfg = FilterDefinitions(wn=20.0, zt=1.0, dt=0.01, solver_type=SolverType.RK4)
        lpf = SecondOrderLPF(cfg)

        yd, y = lpf.step(x)           # uses cfg.dt

    Usage — variable timestep (real-time loop)
    ------------------------------------------
        yd, y = lpf.step(x, dt=0.013)  # override dt for this step only

    Usage — whole sequence, fixed dt
    ---------------------------------
        y_array, yd_array = lpf.run(x_array)

    Usage — whole sequence, variable dt
    ------------------------------------
        y_array, yd_array = lpf.run(x_array, dt_array=timestamps_diff)
    """

    def __init__(self, config: FilterConfig) -> None:
        """Initialize the second-order low-pass filter."""
        self._config = config
        self._state = LowPassFilterState(q=config.x0, y=config.x0)
        self._solver: SolverStrategy = make_solver(config.solver_type)
        self._x = 0.0  # current input, stored so _deriv_fn can access it

    # ------------------------------------------------------------------
    # Signal path helpers — each mirrors one Simulink block / wire
    # ------------------------------------------------------------------

    def _compute_e1(self, x: float, y: float) -> float:
        """1st summer: e1 = x - y."""
        return x - y

    def _compute_feedback(self, q: float) -> float:
        """Bottom multiplier: feedback = 2 * zt * q."""
        return 2.0 * self._config.zt * q

    def _compute_e2(self, e1: float, feedback: float) -> float:
        """2nd summer: e2 = e1 - feedback."""
        return e1 - feedback

    def _compute_yd(self, q: float) -> float:
        """Multiply yd = wn * q.

        Between-step wire labelled 'yd' (port 2) in the diagram.
        Direct input to the 2nd integrator (= dy/dt).
        NOT a stored state — recomputed each step from q.
        """
        return self._config.wn * q

    def _deriv_fn(self, q: float, y: float) -> tuple[float, float]:
        """Compute derivative function (dq/dt, dy/dt) from trial states (q, y) passed to the solver.

        The solver may call this multiple times per step (RK4 calls it 4x),
        each time with different trial values of q and y.
        self._x holds the current filter input, set once per step() call.
        """
        e1 = self._compute_e1(self._x, y)
        feedback = self._compute_feedback(q)
        e2 = self._compute_e2(e1, feedback)
        dq_dt = self._config.wn * e2  # input to 1st integrator
        dy_dt = self._compute_yd(q)  # yd = wn*q = input to 2nd integrator
        return dq_dt, dy_dt

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self, x: float, dt: float = FilterConfig.dt) -> tuple[float, float]:
        """Advance the filter by one timestep.

        Parameters
        ----------
        x : float
            Filter input at the current timestep.
        dt : float, optional
            Timestep duration [s] for this step.
            If provided, overrides cfg.dt for this step only — the stored
            cfg.dt is NOT modified. Pass a new value every call when your
            sample intervals vary (e.g. timestamps from a sensor log).
            Defaults to cfg.dt when not provided.

        Returns
        -------
        yd : float
            Between-step computed wire = wn * q = dy/dt.
            Simulink output port 2. NOT a stored integrator state.
        y : float
            2nd integrator output.
            Simulink output port 1.

        Examples
        --------
        # Fixed timestep — uses cfg.dt:
        yd, y = lpf.step(sensor_value)

        # Variable timestep — override per call:
        yd, y = lpf.step(sensor_value, dt=elapsed_seconds)

        """
        self._x = x
        step_dt = dt

        q_out, y_out, q_next, y_next = self._solver.step(
            deriv_fn=self._deriv_fn,
            q=self._state.q,
            y=self._state.y,
            dt=step_dt,
        )

        self._state.q = q_next
        self._state.y = y_next

        # yd is a computed wire from q_out (output-side q, not next-state q)
        yd = self._compute_yd(q_out)

        return y_out, yd

    def run(
        self,
        x_array: Iterable[float],
        dt_array: float | Iterable[float] | None = None,
    ) -> tuple[list[float], list[float]]:
        """Filter an entire input sequence in one call.

        Resets both integrator states to x0 before processing, so each
        call to run() is independent and reproducible regardless of any
        prior step() calls.

        Parameters
        ----------
        x_array : iterable of float  (list, numpy array, or any sequence)
            Input signal samples, one value per timestep.
        dt_array : optional
            Timestep durations [s]. Three accepted forms:
              None          — use cfg.dt for every step  (fixed, default)
              float         — use this single value for every step  (fixed override)
              iterable      — per-sample intervals, must be same length as x_array

        Returns
        -------
        y_array  : list of float
            Port 1 output — the filtered signal.
        yd_array : list of float
            Port 2 output — derivative of the filtered signal (= dy/dt).

        Examples
        --------
        # Fixed dt from config:
        y_array, yd_array = lpf.run(x_array)

        # Fixed dt override (different from cfg.dt):
        y_array, yd_array = lpf.run(x_array, dt_array=0.013)

        # Variable dt from a timestamp array:
        timestamps = [0.0, 0.010, 0.023, 0.031, ...]
        dt_array   = [t1 - t0 for t0, t1 in zip(timestamps, timestamps[1:])]
        # Note: dt_array has one fewer element than timestamps.
        # x_array should also be trimmed to match: x_array = samples[1:]
        y_array, yd_array = lpf.run(x_array, dt_array=dt_array)

        # Only need the filtered output (ignore yd):
        y_array, _ = lpf.run(x_array)

        """
        self.reset()

        # Normalise dt_array into an iterator of floats
        x_list = list(x_array)
        n = len(x_list)

        if dt_array is None:
            # Use cfg.dt for every step
            dt_iter = [FilterConfig.dt] * n
        elif isinstance(dt_array, (int, float)):
            # Single scalar — broadcast to all steps
            dt_iter = [float(dt_array)] * n
        else:
            # Per-sample array — validate length
            dt_list = list(dt_array)
            if len(dt_list) != n:
                raise ValueError(
                    f"dt_array length ({len(dt_list)}) must match x_array length ({n})."
                )
            dt_iter = dt_list

        y_array: list[float] = []
        yd_array: list[float] = []

        for x, dt in zip(x_list, dt_iter, strict=False):
            yd, y = self.step(float(x), dt=dt)
            y_array.append(y)
            yd_array.append(yd)

        return y_array, yd_array

    def reset(self) -> None:
        """Reset both integrator states to x0."""
        self._state.q = self._config.x0
        self._state.y = self._config.x0

    @property
    def solver_name(self) -> str:
        """Return the name of the solver strategy for logging and debugging."""
        return repr(self._solver)
