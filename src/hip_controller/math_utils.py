"""Math utilities for the hip controller."""

import numpy as np
from loguru import logger
from numpy.typing import NDArray


def symmetrize_matrix(matrix: NDArray) -> NDArray:
    """Symmetrize a matrix.

    :param matrix: A square matrix represented as a numpy array.
    """
    if np.shape(matrix)[0] != np.shape(matrix)[1]:
        dim = matrix.shape
        msg = f"Input matrix must be square. Matrix has dimensions: {dim[0]}x{dim[1]}."
        logger.error(msg)
        raise ValueError(msg)

    return (matrix + matrix.T) / 2


def hit_zero_crossing_from_upper(curr: float, prev: float) -> bool:
    """Detect zero-crossing from upper to lower.

    Checks if a value transitions from non-negative to negative.

    :param curr: Current value.
    :param prev: Previous value.
    :return: True if zero-crossing from upper to lower detected, False otherwise.
    """
    return prev >= 0 and curr < 0


def hit_zero_crossing_from_lower(curr: float, prev: float) -> bool:
    """Detect zero-crossing from lower to upper.

    Checks if a value transitions from non-positive to positive.

    :param curr: Current value.
    :param prev: Previous value.
    :return: True if zero-crossing from lower to upper detected, False otherwise.
    """
    return prev <= 0 and curr > 0
