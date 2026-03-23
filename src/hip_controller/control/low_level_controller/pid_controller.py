"""A pid controller for the motor."""

from hip_controller.definitions import FilterConfig, PIDConfig
from hip_controller.utils.second_order_low_pass_filter import SecondOrderLowPassFilter


class PIDController:
    """PID controller.

    The derivative term is NOT the derivative of the error. The previous velocity motor command is filtered by a second-order low-pass filter and subtracted. This is velocity feedback damping smoothly resists fast changes in the output without differentiating noisy sensor data.
    """

    def __init__(self, pid_config: PIDConfig, filter_config: FilterConfig) -> None:
        """Initialize the pid controller.

        :param PIDConfig pid_config: Configurations of the pid controller, containing proportional gain kp, integral gain ki, derivative gain kd, and min, max clamp on the output.
        :param FilterConfig filter_config: Configurations of the second-order low-pass filter, containing cut-off frequency, damping ratio, initial condition, filter solver type. Time difference should not be initialized here.
        :return: None
        """
        self.config: PIDConfig = pid_config
        self.low_pass_filter: SecondOrderLowPassFilter = SecondOrderLowPassFilter(
            config=filter_config
        )

        self._integral: float = 0.0
        self._prev_velocity: float = 0.0  # previous velocity output fed into LPF
        self._prev_timestamp: float

    def pid_tuning(
        self, timestamp: float, motor_reference: float, motor_position: float
    ) -> float:
        """Compute the velocity command for one control cycle using the PID controller.

        :param float motor_reference: Desired target position motor command of reference motion.
        :param float motor_position: Current measured motor position of actual motion.
        :param float dt:

        :return: Commanded velocity, clamped to output_limits if set.
        :rtype: float
        """
        #  calculate how much the motors have to move
        error = motor_reference - motor_position

        proportional = self.config.proportional_gain * error

        time_difference = timestamp - self._prev_timestamp
        self._integral += error * time_difference
        integral = self.config.integral_gain * self._integral

        _, filtered_velocity = self.low_pass_filter.step(
            x=self._prev_velocity, timestamp=timestamp
        )

        derivative = self.config.derivative_gain * filtered_velocity
        output = proportional + integral - derivative

        output = self._clamp(output, self.config.output_limits)

        # Store velocity for next cycle's D term
        self._prev_velocity = output
        self._prev_timestamp = timestamp
        return output

    def reset(self) -> None:
        """Reset internal state integral and velocity feedback.

        :return: None
        """
        self._integral = 0.0
        self._prev_velocity = 0.0

    @staticmethod
    def _clamp(value: float, limits: tuple[float, float] | None) -> float:
        """Restricts a value so it can never go outside a defined range.

        :param float value: Value.
        :param tuple[float, float] limits: Limits.
        :return: Restricted value.
        :rtype: float
        """
        if limits is None:
            return value
        low, high = limits
        return max(low, min(high, value))
