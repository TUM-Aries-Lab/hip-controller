"""Simulator for different kinds of data."""

import signal  # pragma: no cover
from pathlib import Path  # pragma: no cover

from pyqtgraph import QtCore, QtWidgets  # pragma: no cover

from hip_controller.control.mid_level_controller.mid_level import MotionMapping

# pragma: no cover
from hip_controller.plotter.csv_player import CSVPlayer  # pragma: no cover
from hip_controller.plotter.time_plot import TimePlotterComparisonWindow


def simulate(
    input_name: str,
    output_name: str,
    path: Path,
) -> None:  # pragma: no cover
    """Simulate the comparison of the output and the expected output.

    :param str input_name: The column name of the input variable.
    :param str output_name: The column name of the output variable.
    :param str csv_path: Path to the CSV file used for simulated real-time playback. The user could pass in the path of a file as well.
    :return: None

    Usage Example: simulate( input_name="motion_mapping_key", output_name="motion_mapping_value", path=DATA_DIR / "sensor_data" / "look_up_table_2026_02_25.csv")
    """
    app = QtWidgets.QApplication([])

    player = CSVPlayer(csv_path=path)
    lookup = MotionMapping()
    timer = QtCore.QTimer()
    plotter = TimePlotterComparisonWindow(separated=False)
    plotter.show()

    def update() -> None:
        """Update the controller with the next line of CSV data."""
        if not player.has_next_line():
            timer.stop()
            return

        timestamp, input, _ = player.get_data_from_csv(
            input_name=input_name, output_name=output_name
        )

        # Function to simulate. Here is for example the spline function
        output = lookup.spline(input)
        plotter.update_plots(
            timestamp=timestamp, first_input=input, second_input=output
        )

    def sigint_handler(signal, frame) -> None:
        """Handle SIGINT (Ctrl+C) gracefully."""
        timer.stop()
        app.quit()

    timer.timeout.connect(slot=update)
    signal.signal(signal.SIGINT, sigint_handler)
    timer.start(0)
    app.exec()
