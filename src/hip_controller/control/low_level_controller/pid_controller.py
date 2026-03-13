"""A pid controller which takes the motor position as input."""


class MotorTorque:
    """Torque function of the motor."""

    def __init__(self) -> None:
        """Initialize the motor torque."""
        self.exosuit_switch = False
        self.tension_switch = False


class PIDController:
    """PID controller for the motor."""

    def __init__(self, motor_reference: float, motor_position: float) -> None:
        pass
