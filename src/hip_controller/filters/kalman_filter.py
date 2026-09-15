"""Kalman filter for the angle stage of the preprocessing pipeline."""

import numpy as np
from numpy.typing import NDArray

from hip_controller.definitions import KalmanFilterConfig
from hip_controller.utils.math_utils import symmetrize_matrix
from hip_controller.utils.state_space import StateSpaceLinear


class KalmanFilter:
    """Kalman filter over a constant-velocity model of the hip angle.

    The state is ``[angle, angular velocity]`` and only the angle is measured,
    so the filter smooths the angle while estimating the velocity that explains
    it. :meth:`filter` wraps one predict/update pair into the
    ``(angle_rad, time_difference) -> angle_rad`` shape the pipeline's
    filtering stage expects.
    """

    def __init__(self, config: KalmanFilterConfig) -> None:
        """Initialize the Kalman Filter.

        The model and the initial estimate are copied out of ``config``. The
        filter writes the measured sample period into ``A`` on every step, and
        configs are shared -- one :class:`BasicConfig` serves both limbs -- so
        holding the config's own arrays would make the two legs overwrite each
        other's model.

        :param KalmanFilterConfig config: Model, noise covariances and initial
            estimate.
        :return: None
        """
        self.config = config  # kept so reset() can restore the initial estimate
        self.state_space = StateSpaceLinear(
            A=np.array(config.state_space.A, dtype=float),
            B=np.array(config.state_space.B, dtype=float),
            C=np.array(config.state_space.C, dtype=float),
            D=np.array(config.state_space.D, dtype=float),
        )
        self.Q: NDArray = np.array(config.process_noise, dtype=float)
        self.R: NDArray = np.array(config.measurement_noise, dtype=float)
        self.x: NDArray = np.array(config.initial_state, dtype=float)
        self.cov: NDArray = np.array(config.initial_covariance, dtype=float)

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
        """Execute one filter step: predict, then update with the measurement.

        :param float angle_rad: Raw angle [rad].
        :param float time_difference: Elapsed time since the previous sample [s].
        :return: Filtered angle [rad].
        :rtype: float
        """
        # The constant-velocity model propagates angle by velocity * dt, so the
        # real sample period goes into A before predicting. This writes into
        # this instance's own copy of A -- see __init__.
        self.state_space.A[0, 1] = time_difference
        self.predict(u=None)
        self.update(z=np.array([angle_rad]))
        return float(self.x[0])

    def reset(self) -> None:
        """Reset the filter to its initial condition.

        :return: None
        """
        self.x = np.array(self.config.initial_state, dtype=float)
        self.cov = np.array(self.config.initial_covariance, dtype=float)
