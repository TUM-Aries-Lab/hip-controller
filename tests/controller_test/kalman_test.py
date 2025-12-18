"""Test the mid-level control module."""

import numpy as np

from src.hip_controller.control.kalman import KalmanFilter
from src.hip_controller.control.state_space import StateSpaceLinear


def test_kalman_filter_initialization() -> None:
    """Test the initialization of the Kalman filter."""
    # Arrange
    dt = 0.01
    A = np.array(
        [
            [1.0, dt],
            [0.0, 1.0],
        ]
    )
    C = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    ss = StateSpaceLinear(A=A, C=C)

    initial_state = np.array(
        [
            [0.0],
            [0.0],
        ]
    )
    initial_covariance = np.eye(2)

    # Act
    kf = KalmanFilter(
        state_space=ss,
        initial_x=initial_state,
        initial_covariance=initial_covariance,
    )
    for _i in range(10):
        kf.predict()
        _ = kf.update(z=np.array([[0.0]]))

    # Assert
    assert isinstance(kf, KalmanFilter)
    assert kf.cov[0, 0] < initial_covariance[0, 0]
    assert kf.cov[1, 1] < initial_covariance[1, 1]
