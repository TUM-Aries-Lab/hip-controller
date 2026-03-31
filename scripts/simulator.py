"""Simulator for different kinds of data."""

import signal  # pragma: no cover
from collections.abc import Callable
from pathlib import Path  # pragma: no cover
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import ticker
from pyqtgraph import QtCore, QtWidgets  # pragma: no cover

from hip_controller.control.app import ExoController  # pragma: no cover
from hip_controller.filters.second_order_low_pass_filter import (
    SecondOrderLowPassFilter,
)
from hip_controller.definitions import (
    DEFAULT_LOG_LEVEL,
    BasicConfig,
    LowPassFilterConfig,
    SolverType,
)

from src.hip_controller.plotter.csv_player import CSVPlayer
from dataclasses import dataclass
from scripts.script import ScriptPlayer, ComparisonData
from scripts.live_comparison_plot import TimePlotterComparisonWindow
from hip_controller.utils.utils import setup_logger



def simulate_comparison_dynamic(
    input_name: str,
    expected_output_name: str,
    func: Callable[[float], tuple[float, Any]],
    path: Path,
) -> None:
    """Simulate the comparison of the output and the expected output.

    :param str input_name: The column name of the input variable.
    :param str expected_output_name: The column name of the output variable.
    :param Callable[[float], tuple[float, Any]] func: Function to call and compare.
    :param str csv_path: Path to the CSV file used for simulated real-time playback. The user could pass in the path of a file as well.
    :return: None


    Usage Example:
        from hip_controller.definitions import DATA_DIR
        simulate( input_name="motion_mapping_key", output_name="motion_mapping_value", func=MotionMapping().spline, path=DATA_DIR / "sensor_data" / "look_up_table_2026_02_25.csv")
        -----------
        lpf = SecondOrderLPF(config=FilterConfig(wn=20.0, zt=1.0, x0=0.0, dt=0.01, solver_type=SolverType.RK4))
        if __name__ == "__main__":
            simulate(input_name="x", expected_output_name="y", func=lpf.step, path=TESTING_DIR / "controller_test/low_level_testing/low_level_testing_data/second_order_lpf_2026_03_06.csv")
    """
    app = QtWidgets.QApplication([])

    player = ScriptPlayer(csv_path=path)

    timer = QtCore.QTimer()
    plotter = TimePlotterComparisonWindow(
        separated=False, input_name=input_name, output_name=expected_output_name
    )
    plotter.show()

    def update() -> None:
        """Update the controller with the next line of CSV data."""
        if not player.has_next_line():
            timer.stop()
            return

        data_to_compare: ComparisonData = player.get_data_from_csv(
            input_name=input_name, expected_output_name=expected_output_name
        )

        # Function to simulate. Takes the first output if there were multiple
        output, _ = func(data_to_compare.given_input)

        plotter.update_plots(
            timestamp=data_to_compare.timestamp,
            input=data_to_compare.given_input,
            output=output,
            expected_output=data_to_compare.expected_output,
        )

    def sigint_handler(signal, frame) -> None:
        """Handle SIGINT (Ctrl+C) gracefully."""
        timer.stop()
        app.quit()

    timer.timeout.connect(slot=update)
    signal.signal(signal.SIGINT, sigint_handler)
    timer.start(0)
    app.exec()


def simulate_controller_with_data(
    log_level: str = DEFAULT_LOG_LEVEL,
    stderr_level: str = DEFAULT_LOG_LEVEL,
    csv_path: Path = BasicConfig.read_data_from_path,
) -> None:  # pragma: no cover
    """Run the main pipeline.

    :param log_level: The log level to use.
    :param stderr_level: The std err level to use.
    :param str csv_path: Path to the CSV file used for simulated real-time playback. The user could pass in the path of a file as well.
    :return: None
    """
    setup_logger(log_level=log_level, stderr_level=stderr_level)

    app = QtWidgets.QApplication([])

    player = CSVPlayer(csv_path)
    controller = ExoController()
    timer = QtCore.QTimer()

    def update() -> None:
        """Update the controller with the next line of CSV data."""
        if not player.has_next_line():
            timer.stop()
            return

        sensordata = player.get_sensor_data_from_csv()

        # setInterval in miliseconds. Update each 10ms
        timer.setInterval(10)

        controller.step(sensor_data=sensordata)

    def sigint_handler(signal, frame) -> None:
        """Handle SIGINT (Ctrl+C) gracefully."""
        logger.success("Keyboard interrupted with ^C.")
        timer.stop()
        app.quit()

    timer.timeout.connect(slot=update)
    signal.signal(signal.SIGINT, sigint_handler)
    timer.start(0)
    app.exec()


def demonstrate_random_lpf_static() -> None:  # pragma no cover
    """Demonstrate to test the 2nd order low pass filter with small random data and show the plot."""
    wn = 100.0
    zt = 1.0
    x0 = 0.0
    dt = 0.01

    # Create filter instance
    cfg = LowPassFilterConfig(
        cut_off_frequency=wn,
        damping_ratio=zt,
        initial_condition=x0,
        solver_type=SolverType.RUNGE_KUTTA,
    )
    lpf = SecondOrderLowPassFilter(config=cfg)

    # ── Test signal: step from 0 → 1 at t=0.1s, with added noise ──
    t = np.arange(0, 1.0, dt)
    signal = np.where(t >= 0.1, 1.0, 0.0)
    noise = np.random.normal(0, 0.05, size=len(t))
    noisy = signal + noise

    # Run through the filter
    y_filtered, yd_filtered = lpf.run(noisy)

    # ── Plot ──
    plt.figure(figsize=(10, 5))
    plt.plot(t, noisy, color="#aaaaaa", linewidth=0.8, label="Noisy input x")
    plt.plot(
        t,
        signal,
        color="#2196F3",
        linewidth=1.5,
        linestyle="--",
        label="Clean step (reference)",
    )
    plt.plot(t, y_filtered, color="#E91E63", linewidth=2.0, label="Filtered output y")
    plt.plot(t, yd_filtered, color="#5DD471", linewidth=2.0, label="Filtered output yd")

    plt.title(f"Second-Order Low-Pass Filter  |  ωₙ={wn} rad/s, ζ={zt}, x₀={x0}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()
    logger.info("Plot saved to lpf_plot.png")

    # ── Print first 10 output values ──
    logger.info("\nFirst 10 filtered output values:")
    for i, y in enumerate(y_filtered[:10]):
        logger.info(f"  step {i:02d}  x={noisy[i]:+.4f}  →  y={y:.6f}")


def plot_notch_filter_debug(
    csv_path: Path,
    timestamp_col: str,
    raw_input_col: str,
    expected_col: str,
    actual_results: list[float],
) -> None:
    """Plot notch filter input, expected output, actual output, and their difference.

        :param str csv_path: Path to the CSV file.
        :param str timestamp_col: Column name for timestamps.
        :param str raw_angle_col: Column name for raw angle (filter input).
        :param str expected_col: Column name for expected filtered output.
        :param list[float] actual_results: Actual filter outputs collected during the test.

        # Example
            - plot_notch_filter_debug(csv_path=DATA_PRE_PROCESSING, timestamp_col=KinematicsDataColumnName.TIMESTAMP, raw_angle_col=KinematicsDataColumnName.RAW_ANG_LEFT, expected_col=KinematicsDataColumnName.NO_DRIFT_ANG_NOTCH, actual_results=actual_results,
    )
    """
    df = pd.read_csv(csv_path)

    # Test skips row 0 (no dt), so align to same length
    timestamps = df[timestamp_col].values[1:]
    raw = df[raw_input_col].values[1:]
    expected = df[expected_col].values[1:]
    actual = np.array(actual_results)
    diff = actual - expected

    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(14, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.suptitle("Notch Filter Debug", fontsize=13, fontweight="bold")

    # ---- Top: signals -------------------------------------------------------
    ax1.plot(timestamps, raw, label="Input (raw angle)", alpha=0.5, linewidth=0.8)
    ax1.plot(timestamps, expected, label="Expected output", linewidth=1.2)
    ax1.plot(timestamps, actual, label="Actual output", linestyle="--", linewidth=1.0)
    ax1.set_ylabel("Angle [rad]")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # ---- Bottom: difference -------------------------------------------------
    ax2.plot(
        timestamps, diff, color="crimson", linewidth=0.8, label="Actual - Expected"
    )
    ax2.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax2.set_ylabel("Error [rad]")
    ax2.set_xlabel("Time [s]")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))

    plt.tight_layout()
    plt.show()
