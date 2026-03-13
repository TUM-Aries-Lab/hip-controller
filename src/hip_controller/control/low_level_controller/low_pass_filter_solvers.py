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

from abc import ABC, abstractmethod
from collections.abc import Callable

from hip_controller.definitions import SolverType

# Type alias: deriv_fn(q, y) -> (dq_dt, dy_dt)
DerivFn = Callable[[float, float], tuple[float, float]]


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


# ---------------------------------------------------------------------------
# Concrete Strategy 2 — Backward Euler
# ---------------------------------------------------------------------------


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

        q_out = q + time_difference * dq_dt  # output = state + dt*u (feedthrough)
        y_out = y + time_difference * dy_dt

        q_next = q_out  # next state carries the updated value
        y_next = y_out

        return q_out, y_out, q_next, y_next


# ---------------------------------------------------------------------------
# Concrete Strategy 3 — Trapezoidal
# ---------------------------------------------------------------------------


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
        SolverType.RK4: RK4Solver,
    }
    cls = _map.get(solver_type)
    if cls is None:
        raise ValueError(f"Unknown solver type: {solver_type}")
    return cls()
