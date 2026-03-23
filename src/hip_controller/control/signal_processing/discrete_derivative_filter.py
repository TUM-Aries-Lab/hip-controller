"""Compute discrete derivative of angle.

:param float angle: Input angle [rad].
:param float dt: Time elapsed since last sample [s].
:return: Angular velocity [rad/s].
:rtype: float
"""


class DiscreteDerivativeFilter:
    """Discrete derivative matching the Simulink 'Discrete Derivative' block.

    Implements the backward-difference formula::

        y[k] = K * (u[k] - u[k-1]) / Ts

    which is the ``du/dt`` Simulink block with configurable gain *K*. The
    output is zero on the first call (no prior sample available), mirroring
    Simulink's default initial-condition behaviour.

    :param K: Derivative gain (default 1.0, matching the Simulink block default).
    """

    def __init__(self, K: float = 1.0) -> None:
        self._K = K
        self._u_prev: float | None = None  # None signals "first call"

    def filter(self, u: float, dt: float) -> float:
        """Compute the derivative of *u* over the elapsed time *dt*.

        :param u: Current input sample.
        :param dt: Elapsed time since the previous sample [s]. Must be > 0.
        :return: Estimated derivative ``K * (u - u_prev) / dt`` [unit/s].
            Returns 0.0 on the very first call (no prior sample).
        :raises ValueError: If *dt* is not strictly positive.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt!r}.")

        if self._u_prev is None:
            # First call — no history yet; output zero as Simulink does.
            self._u_prev = u
            return 0.0

        derivative = self._K * (u - self._u_prev) / dt
        self._u_prev = u
        return derivative

    def reset(self, u_init: float = 0.0) -> None:
        """Reset the filter to a known initial condition.

        :param u_init: Value to treat as the previous sample after reset.
        """
        self._u_prev = u_init
