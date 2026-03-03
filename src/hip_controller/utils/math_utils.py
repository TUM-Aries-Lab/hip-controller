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


def hit_crossing_falling(curr: float, prev: float, offset: float = 0) -> bool:
    """Detect certain crossing from upper to lower.

    Checks if a value transitions from non-negative to negative.

    :param curr: Current value.
    :param prev: Previous value.
    :param offset: Position of the crossing. Default by zero-crossing.
    :return: True if crossing from upper to lower detected, False otherwise.
    """
    return prev >= offset > curr


def hit_crossing_rising(curr: float, prev: float, offset: float = 0) -> bool:
    """Detect certain crossing from lower to upper.

    Checks if a value transitions from non-positive to positive.

    :param curr: Current value.
    :param prev: Previous value.
    :param offset: Position of the crossing. Default by zero-crossing.
    :return: True if crossing from lower to upper detected, False otherwise.
    """
    return prev <= offset and curr > offset


def align(center_val: float, curr_val: float) -> float:
    """Normalize value relative to bounded range.

    Computes a normalized steady-state value by removing the midpoint offset of
    the provided maximum and minimum bounds from the current value. This transforms
    a signal bounded by [val_min, val_max] to be centered at zero, useful for
    normalizing joint angle and velocity signals for the gait phase calculation.



    :param float val_curr:
        Current value of the signal.
    :return:
        Steady-state value relative to the range center value.
    :rtype: float
    """
    return curr_val - center_val


def calculate_center_value(val_max: float, val_min: float) -> float:
    """Calculate a centered value of max and min values.

    :param float val_max:
        Upper bound (maximum value) of the expected signal range.
    :param float val_min:
        Lower bound (minimum value) of the expected signal range.

    :return: midpoint. Zero when val_curr
        equals the midpoint of [val_min, val_max].
        :rtype: float
    """
    return (val_max + val_min) / 2.0
