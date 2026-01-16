"""Test the utils module."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from hip_controller.definitions import DEFAULT_LOG_FILENAME, LogLevel
from hip_controller.math_utils import symmetrize_matrix
from hip_controller.utils import setup_logger


def test_logger_init() -> None:
    """Test logger initialization."""
    with TemporaryDirectory() as log_dir:
        log_dir_path = Path(log_dir)
        log_filepath = setup_logger(filename=DEFAULT_LOG_FILENAME, log_dir=log_dir_path)
        assert Path(log_filepath).exists()
    assert not Path(log_filepath).exists()


def test_log_level() -> None:
    """Test the log level."""
    # Act
    log_levels = list(LogLevel())

    # Assert
    assert type(log_levels) is list


def test_symmetrize_matrix():
    """Test symmetrize_matrix function."""
    # asymmetrical square matrix -> symmetrized
    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected = np.array([[1.0, 2.5], [2.5, 4.0]])
    np.testing.assert_allclose(symmetrize_matrix(m), expected)

    # already symmetric remains unchanged
    s = np.array([[1.0, 2.0], [2.0, 1.0]])
    np.testing.assert_allclose(symmetrize_matrix(s), s)

    # non-square matrix raises ValueError
    with pytest.raises(ValueError):
        symmetrize_matrix(np.array([[1, 2, 3], [4, 5, 6]]))
