"""Simulator for different kinds of data."""

import signal  # pragma: no cover
from collections.abc import Callable
from pathlib import Path  # pragma: no cover
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from pyqtgraph import QtCore, QtWidgets  # pragma: no cover

from hip_controller.definitions import FilterConfig, SolverType

# pragma: no cover
from hip_controller.plotter.csv_player import CSVPlayer  # pragma: no cover
from hip_controller.plotter.data_comparison_plot import TimePlotterComparisonWindow
from hip_controller.utils.low_pass_filter import SecondOrderLPF


def simulate_comparison_dynamic(
    input_name: str,
    expected_output_name: str,
    func: Callable[[float], tuple[float, Any]],
    path: Path,
) -> None:  # pragma: no cover
    """Simulate the comparison of the output and the expected output.

    :param str input_name: The column name of the input variable.
    :param str output_name: The column name of the output variable.
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

    player = CSVPlayer(csv_path=path)

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

        timestamp, input, expected_output = player.get_data_from_csv(
            input_name=input_name, expected_output_name=expected_output_name
        )

        # Function to simulate. Takes the first output if there were multiple
        output, _ = func(input)

        plotter.update_plots(
            timestamp=timestamp,
            input=input,
            output=output,
            expected_output=expected_output,
        )

    def sigint_handler(signal, frame) -> None:
        """Handle SIGINT (Ctrl+C) gracefully."""
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
    cfg = FilterConfig(wn=wn, zt=zt, x0=x0, dt=dt, solver_type=SolverType.RK4)
    lpf = SecondOrderLPF(config=cfg)

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
