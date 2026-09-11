"""Qt-free helpers for preparing CSV data for plotting.

These live outside :mod:`hip_controller.plotter.csv_inspector` so they can be
imported — and unit-tested — without pulling in PyQt6. The inspector module
defines Qt widget subclasses at module level, so importing it requires a
working Qt installation and, on Linux, the system libraries Qt links against.
A headless CI runner has neither, which made these helpers untestable there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

# Column names matched case-insensitively against these prefixes are treated
# as timestamp columns and excluded from the plottable signal list, because
# the X axis is synthesized from the sample frequency.
_TIME_COLUMN_PREFIXES: tuple[str, ...] = ("time", "timestamp", "t (")


def discover_plottable_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return the subset of CSV columns that should appear as selectable signals.

    Keeps only numeric columns and drops anything whose header looks like a
    timestamp, because the time axis is synthesized from the sample frequency
    rather than read from the file.

    :param pandas.DataFrame dataframe: parsed CSV.
    :return: ordered list of plottable column names.
    :rtype: list[str]
    """
    plottable: list[str] = []
    for col in dataframe.columns:
        if not is_numeric_dtype(dataframe[col]):
            continue
        lowered = str(col).lower().strip()
        if any(lowered.startswith(prefix) for prefix in _TIME_COLUMN_PREFIXES):
            continue
        plottable.append(str(col))
    return plottable


def synthesize_time_vector(n_samples: int, frequency_hz: int) -> np.ndarray:
    """Synthesize a uniform time vector (seconds) from a sample count and frequency.

    :param int n_samples: number of rows in the CSV.
    :param int frequency_hz: sampling frequency (samples per second).
    :return: 1-D array of timestamps in seconds, length ``n_samples``.
    :rtype: numpy.ndarray
    :raises ValueError: if ``frequency_hz`` is non-positive.
    """
    if frequency_hz <= 0:
        raise ValueError(f"frequency_hz must be positive, got {frequency_hz}.")
    return np.arange(n_samples, dtype=np.float64) / float(frequency_hz)
