"""Low-level control functions."""

from numpy.typing import NDArray

from hip_controller.definitions import A, B, C, D, Q, R


class KalmanFilter:
    """Kalman filter."""

    def __init__(self, state_init: NDArray, P_init: NDArray):
        self.A: NDArray = A  # dynamics model
        self.B: NDArray = B  # control model

        self.C: NDArray = C
        self.D: NDArray = D

        self.Q: NDArray = Q  # dynamics covariance
        self.R: NDArray = R  # control covariance

        self.state: NDArray = state_init
        self.P: NDArray = P_init

    def predict(self, state: NDArray) -> NDArray:
        """Prediction step."""
        self.state = self.A @ state + self.B @ state
        return self.state

    def update(self, measurement: NDArray) -> NDArray:
        """Update step."""
        return self.state
