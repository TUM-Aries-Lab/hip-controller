"""Test the mid-level control module."""

import numpy as np

from src.hip_controller.control.kalman import KalmanFilter
from src.hip_controller.control.state_space import StateSpaceLinear
from src.hip_controller.definitions import MEASUREMENT_NOISE, PROCESS_NOISE


def test_kalman_filter_initialization() -> None:
    """Test the initialization of the Kalman filter."""
    A = np.eye(2)
    B = np.array([[1], [0]])
    ss = StateSpaceLinear(A=A, B=B)

    exp_Q = PROCESS_NOISE * np.eye(2)
    exp_R = MEASUREMENT_NOISE * np.eye(2)
    initial_state = np.array([[1.0], [1.0]])
    initial_covariance = np.eye(2)

    kf = KalmanFilter(
        state_space=ss,
        initial_x=initial_state,
        initial_covariance=initial_covariance,
    )

    assert isinstance(kf, KalmanFilter)
    assert np.array_equal(kf.state_space.A, ss.A)
    assert np.array_equal(kf.state_space.B, ss.B)
    assert np.array_equal(kf.state_space.C, ss.C)
    assert np.array_equal(kf.Q, exp_Q)
    assert np.array_equal(kf.R, exp_R)
    assert np.array_equal(kf.cov, initial_covariance)
    assert np.array_equal(kf.x, initial_state)
