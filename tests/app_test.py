"""Test the main program."""

from hip_controller.app import WalkOnController


def test_controller():
    """Test the main function."""
    # Arrange

    # Act
    controller = WalkOnController(plot=False)
    controller.step(angle=0.0, velocity=0.0, timestamp=0.0)

    # Assert
