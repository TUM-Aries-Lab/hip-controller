"""State space representation for dynamical systems."""

import numpy as np
from loguru import logger


class StateSpaceLinear:
    """A discrete-time state-space model representation."""

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray | None = None,
        C: np.ndarray | None = None,
        D: np.ndarray | None = None,
    ):
        """Initialize the state-space model.

        :param A: State transition matrix
        :param B: Control input matrix
        :param C: Observation matrix
        :param D: Direct transmission matrix
        """
        if B is None:
            B = np.zeros((A.shape[0], 1))
        if A.shape[0] != B.shape[0]:
            msg = (
                f"A and B matrices must have the same number of rows. "
                f"{A.shape[0]} != {B.shape[0]}"
            )
            logger.error(msg)
            raise ValueError(msg)
        self.A = A
        self.B = B

        if C is None:
            C = np.eye(A.shape[0])
        self.C = C

        if D is None:
            D = np.zeros((C.shape[0], B.shape[1]))
        self.D = D

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Step the state-space model by one step.

        :param x: Current state
        :param u: Control input
        :return: Next state
        """
        return self.A @ x + self.B @ u
