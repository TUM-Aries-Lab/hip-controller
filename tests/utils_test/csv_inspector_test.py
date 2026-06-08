"""Tests for the pure helpers in :mod:`hip_controller.plotter.csv_inspector`.

The GUI itself (``CSVInspectorWindow``) is not unit-tested because it requires
a Qt event loop and a display; it is annotated ``# pragma: no cover`` for the
same reason as ``live_phase_portrait.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pytest import raises

from hip_controller.plotter.csv_inspector import (
    discover_plottable_columns,
    synthesize_time_vector,
)


def test_discover_plottable_columns_filters_time_and_strings() -> None:
    """Time-like and non-numeric columns must be excluded."""
    df = pd.DataFrame(
        {
            "time (s)": [0.0, 0.01, 0.02],
            "angle_left (rad)": [0.1, 0.2, 0.3],
            "vel_left (rad/s)": [1.0, 1.1, 1.2],
            "label": ["a", "b", "c"],
        },
    )
    assert discover_plottable_columns(df) == [
        "angle_left (rad)",
        "vel_left (rad/s)",
    ]


def test_discover_plottable_columns_preserves_csv_column_order() -> None:
    """The output order must follow the CSV's column order, not be re-sorted."""
    df = pd.DataFrame(
        {
            "b_signal": [1.0, 2.0],
            "a_signal": [3.0, 4.0],
            "Timestamp": [0.0, 0.1],
        },
    )
    assert discover_plottable_columns(df) == ["b_signal", "a_signal"]


def test_discover_plottable_columns_handles_empty_dataframe() -> None:
    """An empty CSV yields an empty signal list without raising."""
    assert discover_plottable_columns(pd.DataFrame()) == []


def test_synthesize_time_vector_uses_inverse_frequency() -> None:
    """t[i] must equal i / frequency_hz."""
    time_sec = synthesize_time_vector(n_samples=4, frequency_hz=100)
    np.testing.assert_allclose(time_sec, [0.0, 0.01, 0.02, 0.03])


def test_synthesize_time_vector_length_matches_n_samples() -> None:
    """The returned vector must have exactly n_samples entries."""
    assert synthesize_time_vector(n_samples=250, frequency_hz=50).shape == (250,)


def test_synthesize_time_vector_rejects_non_positive_frequency() -> None:
    """Zero or negative frequency must raise ValueError."""
    with raises(ValueError):
        synthesize_time_vector(n_samples=10, frequency_hz=0)
    with raises(ValueError):
        synthesize_time_vector(n_samples=10, frequency_hz=-100)
