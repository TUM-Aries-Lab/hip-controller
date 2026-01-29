"""Math utilities for the hip controller."""

import numpy as np
from loguru import logger
from numpy.typing import NDArray


def symmetrize_matrix(matrix: NDArray) -> NDArray:
    """Symmetrize a matrix.

    :param matrix: A square matrix represented as a numpy array.
    :return: A symmetrized matrix.
    :raises ValueError: If the input matrix is not square.
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
    return prev >= 0 > curr


def hit_zero_crossing_from_lower(curr: float, prev: float) -> bool:
    """Detect zero-crossing from lower to upper.

    Checks if a value transitions from non-positive to positive.

    :param curr: Current value.
    :param prev: Previous value.
    :return: True if zero-crossing from lower to upper detected, False otherwise.
    """
    return prev <= 0 and curr > 0


def normalize(val_max: float, val_min: float, val_curr: float) -> float:
    """Normalize value relative to bounded range.

    Computes a normalized steady-state value by removing the midpoint offset of
    the provided maximum and minimum bounds from the current value. This transforms
    a signal bounded by [val_min, val_max] to be centered at zero, useful for
    normalizing joint angle and velocity signals for the gait phase calculation.


    :param float val_max:
        Upper bound (maximum value) of the expected signal range.
    :param float val_min:
        Lower bound (minimum value) of the expected signal range.
    :param float val_curr:
        Current value of the signal.
    :return:
        Steady-state value relative to the range midpoint. Zero when val_curr
        equals the midpoint of [val_min, val_max].
    :rtype: float
    """
    return val_curr - ((val_max + val_min) / 2.0)
