"""Mid-level control functions."""

import numpy as np
from numpy.typing import NDArray

from hip_controller.definitions import KalmanFilterConfig
from hip_controller.utils.math_utils import symmetrize_matrix


class KalmanFilter:
    """Kalman filter implementation."""

    def __init__(
        self,
        config: KalmanFilterConfig,
    ) -> None:
        """Initialize the Kalman Filter.

        :param config: Kalman filter configuration.
        :return: None
        """
        self.config = config  # save initial state for resets
        self.state_space = config.state_space
        self.Q: NDArray = config.process_noise
        self.R: NDArray = config.measurement_noise
        self.x: NDArray = config.initial_state
        self.cov: NDArray = config.initial_covariance

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

    def filter(self, angle_rad: float, time_difference: float) -> float:
        """Execute one filter step, containing a prediction step and an update step.

        :param angle_rad: Raw angle in rad.
        :param time_difference: Difference dt between current timestamp and previous timestamp.
        :return: Drift-compensated angle in rad.
        """
        # modify the state transition matrix with the given time step
        self.state_space.A[0, 1] = time_difference
        self.predict(u=None)
        self.update(z=np.array([angle_rad]))
        return float(self.x[0])

    def reset(self) -> None:
        """Reset the Kalman filter to its initial condition.

        :return: None
        """
        self.x = self.config.initial_state
        self.cov = self.config.initial_covariance
