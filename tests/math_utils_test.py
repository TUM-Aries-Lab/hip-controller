"""Test the Math utils module."""

import numpy as np
import pytest

from hip_controller.math_utils import (
    hit_zero_crossing_from_lower,
    hit_zero_crossing_from_upper,
    symmetrize_matrix,
)


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


@pytest.mark.parametrize(
    "hz_prev, hz_curr, hz_expected",
    [
        # valid zero-crossings from upper to lower
        (0.1, -0.1, True),
        (0.0, -0.1, True),
        # not a zero-crossing from upper
        (0.0, 0.0, False),
        (-1.0, 0.0, False),
        (2.0, 1.0, False),
        (1.0, 0.0, False),
    ],
)
def test_hit_zero_crossing_from_upper(
    hz_prev: float,
    hz_curr: float,
    hz_expected: bool,
) -> None:
    """Test ``hit_zero_crossing_from_upper`` using parametrized inputs.

    A zero-crossing from upper to lower occurs when the signal
    transitions from a positive or zero value to a strictly
    negative value.

    :param hz_prev: Previous signal value for zero-crossing test.
    :param hz_curr: Current signal value for zero-crossing test.
    :param hz_expected: Expected detection result.

    :return: None
    """
    assert (
        hit_zero_crossing_from_upper(
            prev=hz_prev,
            curr=hz_curr,
        )
        is hz_expected
    )


@pytest.mark.parametrize(
    "hz_prev, hz_curr, hz_expected",
    [
        # valid zero-crossings from lower to upper
        (-0.1, 0.1, True),
        (0.0, 0.1, True),
        # not a zero-crossing from lower
        (0.0, 0.0, False),
        (1.0, 2.0, False),
        (-1.0, -1.0, False),
        (-1.0, 0.0, False),
    ],
)
def test_hit_zero_crossing_from_lower(
    hz_prev: float,
    hz_curr: float,
    hz_expected: bool,
) -> None:
    """Test ``hit_zero_crossing_from_lower`` using parametrized inputs.

    A zero-crossing from lower to upper occurs when the signal
    transitions from a negative or zero value to a strictly
    positive value.

    :param hz_prev: Previous signal value for zero-crossing test.
    :param hz_curr: Current signal value for zero-crossing test.
    :param hz_expected: Expected detection result.
    :return: None
    """
    assert (
        hit_zero_crossing_from_lower(
            prev=hz_prev,
            curr=hz_curr,
        )
        is hz_expected
    )
