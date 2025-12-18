"""Mid-level control functions."""

import numpy as np
from numpy.typing import NDArray

from hip_controller.control.state_space import StateSpaceLinear
from hip_controller.definitions import MEASUREMENT_NOISE, PROCESS_NOISE
from hip_controller.math_utils import symmetrize_matrix


class KalmanFilter:
    """Kalman filter implementation."""

    def __init__(
        self,
        state_space: StateSpaceLinear,
        initial_x: np.ndarray,
        initial_covariance: np.ndarray,
        process_noise: NDArray | None = None,
        measurement_noise: NDArray | None = None,
    ) -> None:
        """Initialize the Kalman Filter.

        :param state_space: linear state space model
        :param initial_x: Initial state estimate
        :param initial_covariance: Initial error covariance
        :param process_noise: Process noise covariance
        :param measurement_noise: Measurement noise covariance
        :return: None
        """
        self.state_space = state_space
        if process_noise is None:
            process_noise = PROCESS_NOISE * np.eye(len(state_space.A))
        self.Q: np.ndarray = process_noise

        if measurement_noise is None:
            measurement_noise = MEASUREMENT_NOISE * np.eye(len(state_space.C))
        self.R: np.ndarray = measurement_noise

        self.x: np.ndarray = initial_x
        self.cov: np.ndarray = initial_covariance

    def predict(self, u: NDArray | None = None) -> None:
        """Predict the next state and error covariance.

        :param u: Control input
        """
        self.x = self.state_space.step(x=self.x, u=u)
        cov = self.state_space.A @ self.cov @ self.state_space.A.T + self.Q
        self.cov = symmetrize_matrix(cov)

    def update(self, z: NDArray) -> NDArray:
        """Update the state estimate with measurement z.

        :param z: Measurement
        :return: Updated state estimate and state covariance
        """
        y = z - self.state_space.C @ self.x

        S = self.state_space.C @ self.cov @ self.state_space.C.T + self.R
        K = self.cov @ self.state_space.C.T @ np.linalg.inv(S)
        self.x = self.x + K @ y

        cov = (np.eye(self.cov.shape[0]) - K @ self.state_space.C) @ self.cov
        self.cov = symmetrize_matrix(cov)

        return z - self.state_space.C @ self.x
