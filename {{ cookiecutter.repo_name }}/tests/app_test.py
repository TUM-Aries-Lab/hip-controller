"""Test the main program."""

from {{ cookiecutter.module_name }}.app import main


def test_main():
    """Test the main function."""
    assert main() is None
