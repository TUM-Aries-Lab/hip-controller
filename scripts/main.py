"""Simulate multiple scripts."""

"""
Comparison plotting utility for WalkOnController preprocessing validation.

Given a CSV file, input column(s), and an expected output column, this module:
  1. Replays the CSV row-by-row through a callable (e.g. preprocessor.filter)
  2. Extracts the actual output via a user-supplied accessor
  3. Plots three panels: actual vs expected | input vs actual | input vs expected
"""



from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from matplotlib.figure import Figure
from loguru import logger

# Re-use shared palette and helpers from evaluation_plots if available,
# otherwise fall back to inline defaults.


C_RIGHT = "#E05C5C"
C_LEFT  = "#4A90D9"
C_FILT  = "#7DCEA0"





# ── Colours specific to this module ─────────────────────────────────────────
C_ACTUAL    = C_RIGHT    # actual output from the callable
C_EXPECTED  = C_FILT     # expected output read from the CSV
C_INPUT     = C_LEFT     # raw input signal


def plot_preprocessor_comparison(
    csv_path: str | Path,
    time_col: str,
    input_col: str,
    expected_output_col: str,
    build_signal: Callable[[float, float], Any],
    run_callable: Callable[[Any], Any],
    extract_output: Callable[[Any], float],
    title: str = "Preprocessor Comparison",
    save_path: str | Path | None = None,
) -> Figure:
    """
    Replay a CSV through a preprocessing callable and compare actual vs expected output.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    for col in (time_col, input_col, expected_output_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {csv_path.name}.\n"
                             f"Available columns: {list(df.columns)}")

    # ── Replay ───────────────────────────────────────────────────────────────
    actual_outputs: list[float] = []
    for _, row in df.iterrows():
        signal = build_signal(row[time_col], row[input_col])
        result = run_callable(signal)
        actual_outputs.append(extract_output(result))

    t        = df[time_col]
    input_   = df[input_col]
    expected = df[expected_output_col]
    actual   = pd.Series(actual_outputs, index=df.index)

    residual = actual - expected

    # ── Layout ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 11))
    gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.48)

    # Panel 1 – Actual vs Expected
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, expected, color=C_EXPECTED, lw=1.0, label=f"Expected  [{expected_output_col}]")
    ax1.plot(t, actual,   color=C_ACTUAL,   lw=1.0, ls="--", label="Actual output")
    ax1.set_ylabel(_infer_unit(expected_output_col))
    ax1.set_title("Actual vs Expected Output")
    ax1.legend(fontsize=8, loc="upper right")

    # Panel 2 – Input vs Actual output
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2_r = ax2.twinx()
    ax2.plot(t, input_,  color=C_INPUT,  lw=0.9, label=f"Input  [{input_col}]")
    ax2_r.plot(t, actual, color=C_ACTUAL, lw=0.9, ls="--", label="Actual output")
    ax2.set_ylabel(_infer_unit(input_col), color=C_INPUT)
    ax2_r.set_ylabel(_infer_unit(expected_output_col), color=C_ACTUAL)
    ax2.tick_params(axis="y", labelcolor=C_INPUT)
    ax2_r.tick_params(axis="y", labelcolor=C_ACTUAL)
    ax2.set_title("Input vs Actual Output")



    # Panel 3 – Input vs Expected output
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3_r = ax3.twinx()
    ax3.plot(t, input_,   color=C_INPUT,    lw=0.9, label=f"Input  [{input_col}]")
    ax3_r.plot(t, expected, color=C_EXPECTED, lw=0.9, ls="--", label=f"Expected  [{expected_output_col}]")
    ax3.set_ylabel(_infer_unit(input_col), color=C_INPUT)
    ax3_r.set_ylabel(_infer_unit(expected_output_col), color=C_EXPECTED)
    ax3.tick_params(axis="y", labelcolor=C_INPUT)
    ax3_r.tick_params(axis="y", labelcolor=C_EXPECTED)
    ax3.set_title("Input vs Expected Output")



    # Panel 4 – Residual (actual − expected)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(t, residual, color="grey", lw=0.8)
    ax4.axhline(0, color="lightgrey", lw=0.6, ls="--")
    ax4.set_ylabel(_infer_unit(expected_output_col))
    ax4.set_xlabel("Time (s)")
    ax4.set_title("Residual  (Actual - Expected)")

    fig.suptitle(f"{title}\n{csv_path.name}", fontsize=11)
    plt.show()
    return fig


# ── Internal helper ──────────────────────────────────────────────────────────

def _infer_unit(col_name: str) -> str:
    """Extract a short axis label from a column name, e.g. 'angle_right (rad)' → 'rad'."""
    import re
    m = re.search(r"\(([^)]+)\)", col_name)
    return m.group(1) if m else col_name


# ============================================================================
# Example
# ============================================================================

if __name__ == "__main__":
    from hip_controller.control.signal_processing.sensor_preprocessor import SensorPreprocessor, PreprocessorConfig
    from hip_controller.definitions import SensorSignal

    preprocessor = SensorPreprocessor(PreprocessorConfig())

    plot_preprocessor_comparison(
        csv_path            = "hip-controller/scripts/evaluation_output/normal_walk/normal_walk_1_2/AB06_normal_walk_1_1-2_angle.csv",
        time_col            = "time (s)",
        input_col           = "angle_right (rad)",
        expected_output_col = "filtered_velocity_right (rad/s)",
        build_signal        = lambda t, a: SensorSignal(timestamp=t, angle_rad=a, velocity_rad_per_sec=0.0),
        run_callable        = preprocessor.filter,
        extract_output      = lambda sig: sig.velocity_rad_per_sec,
        title               = "Right Hip — Filter Velocity: Actual vs Expected",
        save_path           = "evaluation_plots/filter_comparison_right.png")
