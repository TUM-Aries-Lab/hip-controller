"""State space representation for dynamical systems."""

import numpy as np
from loguru import logger
from numpy.typing import NDArray


class StateSpaceLinear:
    """A discrete-time state-space model representation."""

    def __init__(
        self,
        A: NDArray,
        B: NDArray | None = None,
        C: NDArray | None = None,
        D: NDArray | None = None,
    ):
        """Initialize the state-space model.

        :param A: State transition matrix
        :param B: Control input matrix
        :param C: Observation matrix
        :param D: Direct transmission matrix
        """
        self.A = A
        self.B = np.zeros((self.A.shape[0], 1)) if B is None else B
        self.C = np.eye(self.A.shape[0]) if C is None else C
        self.D = np.zeros((self.C.shape[0], self.B.shape[1])) if D is None else D

        if self.A.shape[0] != self.B.shape[0]:
            msg = (
                f"A and B matrices must have the same number of rows. "
                f"{self.A.shape[0]} != {self.B.shape[0]}"
            )
            logger.error(msg)
            raise ValueError(msg)

    def step(self, x: NDArray, u: NDArray | None = None) -> NDArray:
        """Step the state-space model by one step.

        :param x: Current state
        :param u: Control input
        :return: Next state
        """
        if u is None:
            return self.A @ x

        return self.A @ x + self.B @ u

    def __repr__(self) -> str:
        """Return a string representation of the state space."""
        return f"A:{self.A} \nB:{self.B} \nC:{self.C} \nD:{self.D}"
