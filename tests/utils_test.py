"""Test the utils module."""

from pathlib import Path
from tempfile import TemporaryDirectory

from hip_controller.definitions import DEFAULT_LOG_FILENAME, LogLevel
from hip_controller.utils import setup_logger


def test_logger_init() -> None:
    """Test logger initialization."""
    with TemporaryDirectory() as log_dir:
        log_dir_path = Path(log_dir)
        log_filepath = setup_logger(filename=DEFAULT_LOG_FILENAME, log_dir=log_dir_path)
        assert Path(log_filepath).exists()
    assert not Path(log_filepath).exists()


def test_log_level() -> None:
    """Test the log level.

    :return: None
    """
    # Act
    log_levels = list(LogLevel())

    # Assert
    assert type(log_levels) is list
