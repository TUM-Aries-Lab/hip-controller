"""Test the Math utils module."""

import numpy as np
import pytest

from hip_controller.math_utils import symmetrize_matrix


@pytest.mark.parametrize(
    "matrix, expected",
    [
        # asymmetrical square matrix -> symmetrized
        (
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[1.0, 2.5], [2.5, 4.0]]),
        ),
        # already symmetric remains unchanged
        (
            np.array([[1.0, 2.0], [2.0, 1.0]]),
            np.array([[1.0, 2.0], [2.0, 1.0]]),
        ),
    ],
)
def test_symmetrize_matrix_valid(matrix: np.ndarray, expected: np.ndarray) -> None:
    """Est ``symmetrize_matrix`` with valid square matrices.

    This test verifies that:
    - An asymmetric square matrix is correctly symmetrized.
    - An already symmetric matrix remains unchanged.

    :param matrix: Input square matrix.
    :type matrix: numpy.ndarray
    :param expected: Expected symmetrized matrix.
    :type expected: numpy.ndarray
    :return: None
    """
    np.testing.assert_allclose(symmetrize_matrix(matrix), expected)


def test_symmetrize_matrix_non_square_raises() -> None:
    """Test that ``symmetrize_matrix`` raises ``ValueError`` for non-square matrices.

    :return: None
    :raises ValueError: If the input matrix is not square.
    """
    with pytest.raises(ValueError):
        symmetrize_matrix(np.array([[1, 2, 3], [4, 5, 6]]))
