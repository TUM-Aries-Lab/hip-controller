"""Test the main program."""

from hip_controller.app import main


def test_main():
    """Test the main function."""
    assert main() is None
