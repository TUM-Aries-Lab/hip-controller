"""Test the main program."""

from hip_controller.app import WalkOnController
from hip_controller.control.high_level import MotionState


def test_controller():
    """Test the main function."""
    # Arrange

    # Act
    controller = WalkOnController(freq=1.0)
    controller.step(theta=0.0, theta_dot=0.0)

    # Assert


def test_setters_update_state_fields_and_state_enum():
    """Test the state setter methods of the controller."""
    controller = WalkOnController(freq=1.0)

    # Angle max setter
    controller.curr_angle = 0.75
    controller.set_state_angle_max()
    assert controller.state == MotionState.ANGLE_MAX
    assert controller.angle_max == 0.75

    # Angle min setter
    controller.curr_angle = -0.5
    controller.set_state_angle_min()
    assert controller.state == MotionState.ANGLE_MIN
    assert controller.angle_min == -0.5

    # Velocity max setter
    controller.curr_velocity = 1.23
    controller.set_state_velocity_max()
    assert controller.state == MotionState.VELOCITY_MAX
    assert controller.velocity_max == 1.23

    # Velocity min setter
    controller.curr_velocity = -2.34
    controller.set_state_velocity_min()
    assert controller.state == MotionState.VELOCITY_MIN
    assert controller.velocity_min == -2.34

    # Initial setter resets dt and sets state
    controller.dt = 5.0
    controller.set_state_initial()
    assert controller.state == MotionState.INITIAL
    assert controller.dt == 0.0
