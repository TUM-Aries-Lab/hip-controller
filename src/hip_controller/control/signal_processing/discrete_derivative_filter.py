"""Discrete-time derivative of the input."""


class DiscreteDerivativeFilter:
    """Discrete-time derivative filter.

    Implements the backward-difference formula::

        y[k] = K * (u[k] - u[k-1]) / Ts
    """

    def __init__(self) -> None:
        """Initialize the discrete derivative filter."""
        self._gain_value_k = 1.0
        # Returns 0.0 on the very first call (no prior sample).
        self._initial_condition = 0.0

        self._theta_prev: float | None = None  # None signals first call
        self._derivative_prev: float = 0.0

    def filter(self, theta: float, time_difference: float) -> float:
        """Compute the derivative of *u* over the elapsed time *dt*.

        :param float theta: Current input angle theta.
        :param time_difference: Elapsed time since the previous sample [s]. Must be > 0.
        :return: Estimated derivative ``K * (u - u_prev) / Ts`` [unit/s].

        :raises ValueError: If *time_difference* is not strictly positive.
        """
        if self._theta_prev is None:
            # First call with no history yet; output initial condition.
            self._theta_prev = theta
            return self._initial_condition

        if time_difference < 0.0:
            raise ValueError(
                f"Time difference must be positive, got {time_difference!r}."
            )

        elif time_difference == 0.0:
            return self._derivative_prev

        derivative = self._gain_value_k * (theta - self._theta_prev) / time_difference

        self._theta_prev = theta
        self._derivative_prev = derivative

        return derivative

    def reset(self, theta_init: float = 0.0) -> None:
        """Reset the filter to a known initial condition.

        :param theta_init: Value to treat as the previous sample after reset.
        """
        self._theta_prev = theta_init
