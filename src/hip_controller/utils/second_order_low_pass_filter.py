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
  step(x, timestamp)
      One sample at a time. Advances using the provided timestamp.
      Use this in a real-time loop where timestamps are available.

  run(x_array, timestamp_array=None)
      Whole sequence at once; resets state first.
      timestamp_array may be a matching array of timestamps, or None to use
      default timing based on cfg.time_difference.

  reset()
      Return both integrators to x0.

The solver strategy (Forward Euler / Backward Euler / Trapezoidal / RK4)
is selected via FilterDefinitions.solver_type and injected at construction.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from hip_controller.definitions import FilterConfig, SolverType

# Type alias: deriv_fn(q, y) -> (dq_dt, dy_dt)
DerivFn = Callable[[float, float], tuple[float, float]]

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


class SecondOrderLowPassFilter:
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
        self._state = LowPassFilterState(
            q=config.initial_condition, y=config.initial_condition
        )
        self._solver: SolverStrategy = make_solver(config.solver_type)
        self._x = 0.0  # current input, stored so _deriv_fn can access it
        self._prev_timestamp: float | None = None

    # ------------------------------------------------------------------
    # Signal path helpers — each mirrors one Simulink block / wire
    # ------------------------------------------------------------------

    def _compute_e1(self, x: float, y: float) -> float:
        """1st summer: e1 = x - y."""
        return x - y

    def _compute_feedback(self, q: float) -> float:
        """Bottom multiplier: feedback = 2 * zt * q."""
        return 2.0 * self._config.damping_ratio * q

    def _compute_e2(self, e1: float, feedback: float) -> float:
        """2nd summer: e2 = e1 - feedback."""
        return e1 - feedback

    def _compute_yd(self, q: float) -> float:
        """Multiply yd = wn * q.

        Between-step wire labelled 'yd' (port 2) in the diagram.
        Direct input to the 2nd integrator (= dy/dt).
        NOT a stored state — recomputed each step from q.
        """
        return self._config.cut_off_frequency * q

    def _deriv_fn(self, q: float, y: float) -> tuple[float, float]:
        """Compute derivative function (dq/dt, dy/dt) from trial states (q, y) passed to the solver.

        The solver may call this multiple times per step (RK4 calls it 4x),
        each time with different trial values of q and y.
        self._x holds the current filter input, set once per step() call.
        """
        e1 = self._compute_e1(self._x, y)
        feedback = self._compute_feedback(q)
        e2 = self._compute_e2(e1, feedback)
        dq_dt = self._config.cut_off_frequency * e2  # input to 1st integrator
        dy_dt = self._compute_yd(q)  # yd = wn*q = input to 2nd integrator
        return dq_dt, dy_dt

    def step(self, x: float, timestamp: float) -> tuple[float, float]:
        """Advance the filter by one timestep.

        :param float x: Filter input at the current timestep.
        :param float timestamp:
            Current timestamp [s]. Used to calculate the time difference
            from the previous step. For the first step, uses cfg.time_difference.

        :return: [y, yd]. Filtered value y and its derivative yd.
        :rtype: tuple [float, float]


        Examples
        --------
        # First step — uses cfg.time_difference:
        y, yd = lpf.step(sensor_value, timestamp=0.0)

        # Subsequent steps — calculates dt from timestamps:
        y, yd = lpf.step(sensor_value, timestamp=0.013)

        """
        self._x = x

        if self._prev_timestamp is None:
            time_difference = self._config.time_difference
        else:
            time_difference = timestamp - self._prev_timestamp

        self._prev_timestamp = timestamp

        q_out, y_out, q_next, y_next = self._solver.step(
            deriv_fn=self._deriv_fn,
            q=self._state.q,
            y=self._state.y,
            time_difference=time_difference,
        )

        self._state.q = q_next
        self._state.y = y_next

        # yd is a computed wire from q_out (output-side q, not next-state q)
        yd = self._compute_yd(q_out)

        return y_out, yd

    def run(
        self,
        x_array: Iterable[float],
        timestamp_array: float | Iterable[float] | None = None,
    ) -> tuple[list[float], list[float]]:
        """Filter an entire input sequence in one call.

        Resets both integrator states to x0 before processing, so each
        call to run() is independent and reproducible regardless of any
        prior step() calls.

        Parameters
        ----------
        x_array : iterable of float  (list, numpy array, or any sequence)
            Input signal samples, one value per timestep.
        timestamp_array : optional
            Timestamps [s]. Three accepted forms:
              None          — generate timestamps starting from 0 with cfg.time_difference
              float         — starting timestamp, then increment by cfg.time_difference
              iterable      — per-sample timestamps, must be same length as x_array

        Returns
        -------
        y_array  : list of float
            Port 1 output — the filtered signal.
        yd_array : list of float
            Port 2 output — derivative of the filtered signal (= dy/dt).

        Examples
        --------
        # Generate timestamps from config:
        y_array, yd_array = lpf.run(x_array)

        # Fixed starting timestamp, then increment by cfg.time_difference:
        y_array, yd_array = lpf.run(x_array, timestamp_array=0.0)

        # Variable timestamps:
        timestamps = [0.0, 0.010, 0.023, 0.031, ...]
        y_array, yd_array = lpf.run(x_array, timestamp_array=timestamps)

        # Only need the filtered output (ignore yd):
        y_array, _ = lpf.run(x_array)

        """
        self.reset()

        # Normalise timestamp_array into a list of floats
        x_list = list(x_array)
        n = len(x_list)

        if timestamp_array is None:
            # Generate timestamps starting from 0 with cfg.time_difference
            timestamp_list = [i * self._config.time_difference for i in range(n)]
        elif isinstance(timestamp_array, (int, float)):
            # Starting timestamp, then increment by cfg.time_difference
            start_t = float(timestamp_array)
            timestamp_list = [
                start_t + i * self._config.time_difference for i in range(n)
            ]
        else:
            # Per-sample timestamps — validate length
            timestamp_list = list(timestamp_array)
            if len(timestamp_list) != n:
                raise ValueError(
                    f"timestamp_array length ({len(timestamp_list)}) must match x_array length ({n})."
                )
            # Check for None values
            if any(t is None for t in timestamp_list):
                raise ValueError("timestamp cannot be None")

        y_array: list[float] = []
        yd_array: list[float] = []

        for x, t in zip(x_list, timestamp_list, strict=False):
            yd, y = self.step(float(x), timestamp=t)
            y_array.append(y)
            yd_array.append(yd)

        return y_array, yd_array

    def reset(self) -> None:
        """Reset both integrator states to x0."""
        self._state.q = self._config.initial_condition
        self._state.y = self._config.initial_condition

    @property
    def solver_name(self) -> str:
        """Return the name of the solver strategy for logging and debugging."""
        return repr(self._solver)


"""Solver strategies for the second-order low-pass filter.

Design pattern: Strategy
────────────────────────
  - SolverStrategy (ABC) defines the shared interface.
  - Each concrete solver implements step(), which advances the two filter
    states (q, y) by one timestep given a derivative function.
  - The filter owns state; the solver only defines HOW states are advanced.
  - Solvers are interchangeable without changing any filter logic.

Interface contract
──────────────────
  Every solver implements:

    step(deriv_fn, q, y, dt) -> (q_out, y_out, q_next, y_next)

  Where:
    deriv_fn(q, y) -> (dq_dt, dy_dt)
        Pure derivative function provided by the filter.
        Solvers never know about wn, zt, x, or what the filter represents.

    q_out, y_out   : values to RETURN as outputs this step
    q_next, y_next : values to STORE as states for the next step

  NOTE: For continuous solvers (RK4),  q_out == q_next  (no feedthrough).
        For discrete solvers, q_out may differ from q_next (feedthrough).

Solver summary
──────────────
  Forward Euler  : output = state BEFORE update         → no feedthrough,  O(dt¹)
  Backward Euler : output = state + dt·u                → feedthrough,     O(dt¹)
  Trapezoidal    : output = state + dt/2·u              → feedthrough,     O(dt²)
  RK4            : output = state after 4-stage update  → no feedthrough,  O(dt⁴)
"""


class SolverStrategy(ABC):
    """Abstract base for all integration strategies.

    The filter calls step() each timestep without knowing which solver is
    active — this is the core of the Strategy pattern.
    """

    @abstractmethod
    def step(
        self,
        deriv_fn: DerivFn,
        q: float,
        y: float,
        time_difference: float,
    ) -> tuple[float, float, float, float]:
        """Advance states (q, y) by one timestep of the abstract method.

        :param DerivFn deriv_fn : callable(q, y) -> (dq_dt, dy_dt)
            Derivative function provided by the filter.
        :param float q: current 1st integrator state  (internal)
        :param float y : current 2nd integrator state  (output port 1)
        :param float dt : timestep [s]

        :return:
        q_out  : 1st state to use as output this step
        y_out  : 2nd state to use as output this step  (port 1)
        q_next : 1st state to store for the next step
        y_next : 2nd state to store for the next step

        :rtype: tuple[float, float, float, float]

        """
        ...

    def __repr__(self) -> str:
        """Return the class name for easy identification in logs and debugging."""
        return self.__class__.__name__


class ForwardEulerSolver(SolverStrategy):
    """Forward Euler (Forward Rectangular) integration.

    Maps to: Simulink discrete integrator — 'Forward Euler' method.
    Approximation: 1/s ≈ T/(z-1)

    Rule per integrator block:
        y(k)   = x(k)               ← output is the state BEFORE the update
        x(k+1) = x(k) + dt * u(k)  ← state advances AFTER output is read

    Properties:
        - No feedthrough: output does not depend on current input u(k).
        - 1st-order accurate: error ~ O(dt).
        - 1 derivative evaluation per step — simplest and cheapest.
        - Can become unstable if dt is too large relative to system bandwidth.
    """

    def step(
        self,
        deriv_fn: DerivFn,
        q: float,
        y: float,
        time_difference: float,
    ) -> tuple[float, float, float, float]:
        """Advance states (q, y) by one timestep of the forward euler method.

        :param DerivFn deriv_fn : callable(q, y) -> (dq_dt, dy_dt)
            Derivative function provided by the filter.
        :param float q: current 1st integrator state  (internal)
        :param float y : current 2nd integrator state  (output port 1)
        :param float dt : timestep [s]

        :return:
        q_out  : 1st state to use as output this step
        y_out  : 2nd state to use as output this step  (port 1)
        q_next : 1st state to store for the next step
        y_next : 2nd state to store for the next step

        """
        dq_dt, dy_dt = deriv_fn(q, y)  # slope at current states

        q_out = q  # output = state BEFORE update
        y_out = y

        q_next = q + time_difference * dq_dt  # state advances after output is read
        y_next = y + time_difference * dy_dt

        return q_out, y_out, q_next, y_next


class BackwardEulerSolver(SolverStrategy):
    """Backward Euler (Backward Rectangular) integration.

    Maps to: Simulink discrete integrator — 'Backward Euler' method.
    Approximation: 1/s ≈ T*z/(z-1)

    Rule per integrator block:
        y(k)   = x(k) + dt * u(k)  ← output INCLUDES current input (feedthrough)
        x(k+1) = y(k)              ← next state equals the output

    Properties:
        - Feedthrough: current input u(k) directly affects output y(k).
        - 1st-order accurate: error ~ O(dt), same order as Forward Euler.
        - A-stable: more stable than Forward Euler for stiff systems.
        - 1 derivative evaluation per step.
    """

    def step(
        self,
        deriv_fn: DerivFn,
        q: float,
        y: float,
        time_difference: float,
    ) -> tuple[float, float, float, float]:
        """Advance states (q, y) by one timestep of the backward euler method.

        :param DerivFn deriv_fn : callable(q, y) -> (dq_dt, dy_dt)
            Derivative function provided by the filter.
        :param float q: current 1st integrator state
        :param float y : current 2nd integrator state
        :param float dt : timestep [s]

        :return:
        q_out  : 1st state to use as output this step
        y_out  : 2nd state to use as output this step  (port 1)
        q_next : 1st state to store for the next step
        y_next : 2nd state to store for the next step

        """
        dq_dt, dy_dt = deriv_fn(q, y)  # slope at current states

        q_out = q + time_difference * dq_dt  # output = state + dt*u (feedthrough)
        y_out = y + time_difference * dy_dt

        q_next = q_out  # next state carries the updated value
        y_next = y_out

        return q_out, y_out, q_next, y_next


class TrapezoidalSolver(SolverStrategy):
    """Trapezoidal (Tustin / Bilinear) integration.

    Maps to: Simulink discrete integrator — 'Trapezoidal' method.
    Approximation: 1/s ≈ T/2 * (z+1)/(z-1)

    Rule per integrator block:
        y(k)   = x(k) + dt/2 * u(k)    ← output is midpoint between current and next
        x(k+1) = y(k) + dt/2 * u(k)    ← state advances a further dt/2
               = x(k) + dt  * u(k)     ← equivalent to a full Forward Euler state step

    Properties:
        - Feedthrough: current input affects output.
        - 2nd-order accurate: error ~ O(dt²) — best of the three discrete methods.
        - A-stable: suitable for stiff systems.
        - 1 derivative evaluation per step.
    """

    def step(
        self,
        deriv_fn: DerivFn,
        q: float,
        y: float,
        time_difference: float,
    ) -> tuple[float, float, float, float]:
        """Advance states (q, y) by one timestep of the trapezoidal method.

        :param DerivFn deriv_fn : callable(q, y) -> (dq_dt, dy_dt)
            Derivative function provided by the filter.
        :param float q: current 1st integrator state
        :param float y : current 2nd integrator state
        :param float dt : timestep [s]

        :return:
        q_out  : 1st state to use as output this step
        y_out  : 2nd state to use as output this step
        q_next : 1st state to store for the next step
        y_next : 2nd state to store for the next step

        """
        dq_dt, dy_dt = deriv_fn(q, y)  # slope at current states

        q_out = q + (time_difference / 2.0) * dq_dt  # output = midpoint
        y_out = y + (time_difference / 2.0) * dy_dt

        q_next = q_out + (time_difference / 2.0) * dq_dt  # = q + dt * dq_dt
        y_next = y_out + (time_difference / 2.0) * dy_dt  # = y + dt * dy_dt

        return q_out, y_out, q_next, y_next


# ---------------------------------------------------------------------------
# Concrete Strategy 4 — RK4
# ---------------------------------------------------------------------------


class RK4Solver(SolverStrategy):
    """Runge-Kutta 4th Order integration.

    Maps to: Simulink continuous integrator blocks + ode4 fixed-step solver.
    Use this when the Simulink integrator dialog reads:
        'Continuous-time integration of the input signal.'

    Algorithm — 4 slope evaluations, then a weighted average:
        k1 = deriv(q,              y             )        slope at start
        k2 = deriv(q + dt/2*k1_q, y + dt/2*k1_y )        midpoint via k1
        k3 = deriv(q + dt/2*k2_q, y + dt/2*k2_y )        midpoint via k2 (better)
        k4 = deriv(q + dt  *k3_q, y + dt  *k3_y )        endpoint via k3

        q_next = q + dt/6 * (k1_q + 2*k2_q + 2*k3_q + k4_q)
        y_next = y + dt/6 * (k1_y + 2*k2_y + 2*k3_y + k4_y)

    Midpoint stages are weighted x2 because they capture more of the curve's
    shape across the full interval than the single endpoint stage.

    Properties:
        - No feedthrough: output = updated state.
        - 4th-order accurate: error ~ O(dt⁴).
        - 4 derivative evaluations per step (negligible cost in practice).
        - Faithful representation of a Simulink continuous-time model.
    """

    def step(
        self,
        deriv_fn: DerivFn,
        q: float,
        y: float,
        time_difference: float,
    ) -> tuple[float, float, float, float]:
        """Advance states (q, y) by one timestep of the RK4 method.

        :param DerivFn deriv_fn : callable(q, y) -> (dq_dt, dy_dt)
            Derivative function provided by the filter.
        :param float q: current 1st integrator state  (internal)
        :param float y : current 2nd integrator state  (output port 1)
        :param float dt : timestep [s]

        :return:
        q_out  : 1st state to use as output this step
        y_out  : 2nd state to use as output this step  (port 1)
        q_next : 1st state to store for the next step
        y_next : 2nd state to store for the next step

        """
        # Stage 1 — slope at start
        k1_q, k1_y = deriv_fn(q, y)

        # Stage 2 — slope at midpoint using k1
        k2_q, k2_y = deriv_fn(
            q + 0.5 * time_difference * k1_q, y + 0.5 * time_difference * k1_y
        )

        # Stage 3 — slope at midpoint using k2 (refined estimate)
        k3_q, k3_y = deriv_fn(
            q + 0.5 * time_difference * k2_q, y + 0.5 * time_difference * k2_y
        )

        # Stage 4 — slope at endpoint using k3
        k4_q, k4_y = deriv_fn(q + time_difference * k3_q, y + time_difference * k3_y)

        # Weighted average — midpoint stages count double
        q_next = q + (time_difference / 6.0) * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
        y_next = y + (time_difference / 6.0) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)

        # No feedthrough: output = updated state
        return q_next, y_next, q_next, y_next


def make_solver(solver_type: SolverType) -> SolverStrategy:
    """Return the correct SolverStrategy for a given SolverType enum value.

    Example:
        from definitions import SolverType
        solver = make_solver(SolverType.RK4)

    """
    _map = {
        SolverType.FORWARD_EULER: ForwardEulerSolver,
        SolverType.BACKWARD_EULER: BackwardEulerSolver,
        SolverType.TRAPEZOIDAL: TrapezoidalSolver,
        SolverType.RUNGE_KUTTA: RK4Solver,
    }
    cls = _map.get(solver_type)
    if cls is None:
        raise ValueError(f"Unknown solver type: {solver_type}")
    return cls()
