"""Test the main program."""

from hip_controller.app import WalkOnController


def test_controller():
    """Test the main function."""
    # Arrange

    # Act
    controller = WalkOnController(freq=1.0)
    controller.step(theta=0.0, theta_dot=0.0)

    # Assert
