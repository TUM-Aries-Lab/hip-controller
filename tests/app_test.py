"""Test the main program."""

from hip_controller.app import SensorSignal, WalkOnController


def test_controller():
    """Test the main function."""
    # Arrange

    # Act
    controller = WalkOnController(plot=False)
    controller.step(
        curr_signal=SensorSignal(angle_rad=0.0, velocity_rad_per_sec=0.0), timestamp=0.0
    )

    # Assert
