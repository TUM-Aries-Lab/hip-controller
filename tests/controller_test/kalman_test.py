"""Test the mid-level control module."""

import numpy as np

from src.hip_controller.control.kalman import KalmanFilter
from src.hip_controller.control.state_space import StateSpaceLinear


def test_kalman_filter_initialization() -> None:
    """Test the initialization of the Kalman filter.

    :return: None
    """
    # Arrange
    dt = 0.01
    A = np.array(
        [
            [1.0, dt],
            [0.0, 1.0],
        ]
    )
    C = np.eye(2)
    ss = StateSpaceLinear(A=A, C=C)

    # Act
    kf = KalmanFilter(
        state_space=ss,
        initial_x=np.zeros((2, 1)),
        initial_covariance=np.eye(2),
    )
    for _i in range(10):
        kf.predict()
        _ = kf.update(z=np.array([[0.0]]))

    # Assert
    assert isinstance(kf, KalmanFilter)
    assert np.all(np.diag(kf.cov) < np.diag(np.eye(2)))
