"""Time-domain plotting utilities for real-time visualization of two signals."""

# pragma: no cover
from collections import deque

import pyqtgraph as pg
from PyQt6 import QtWidgets

from hip_controller.definitions import PlotConfig


class TimePlotterComparisonWindow(QtWidgets.QMainWindow):  # pragma: no cover
    """PyQt6 window that displays two real-time plots."""

    def __init__(self, separated: bool, input_name: str, output_name: str) -> None:
        """Initialize the real-time plotting window.

        :param bool separated: Whether the data should be showed in a separated way, or not.
        :return: None
        """
        # Initialize the QMainWindow base class
        super().__init__()

        # Set global PyQtGraph appearance (white background, black axes)
        pg.setConfigOption(opt="background", value="w")
        pg.setConfigOption(opt="foreground", value="k")
        pg.setConfigOption(opt="antialias", value=True)

        # If the graph is separated or not
        self.separated: bool = separated

        # Deques for the time plot
        self.time_buf = deque()

        self.input_buf = deque()
        self.output_buf = deque()

        self.expected_output_buf = deque()

        self._sample_counter = 0
        self._draw_every = 5

        # Build all UI components

        # Create a central widget to hold everything
        central = QtWidgets.QWidget()

        # these settings take no keyword arguments
        # Vertical layout: left plot + right plot
        layout = QtWidgets.QVBoxLayout(central)

        # Set window title
        self.setWindowTitle(f"Time domain plotter comparing {output_name}")

        # Set the central widget of the window
        self.setCentralWidget(central)

        # ---- Up graph  ----
        self.time_plot_up = pg.PlotWidget()

        # Label the axes
        self.time_plot_up.setLabel("bottom", "Time", units="s")

        # Add legend
        self.time_plot_up.addLegend()

        # PlotWidget.setXRange(r, padding) is a wrapper that forwards arguments down to ViewBox.setXRange(min, max, padding)
        # At runtime, ViewBox expects min and max. As a result, min and max are passed positionally here, instead of keywords.
        # This works for all the function calls setXRange, setYRange
        self.time_plot_up.setYRange(
            PlotConfig.time_plot_ymin, PlotConfig.time_plot_ymax
        )

        # Create the curve that will be updated in real time
        self.input_curve = self.time_plot_up.plot(
            pen=pg.mkPen(
                color="#E2E2E2",
                width=PlotConfig.time_plot_curve_width,
            ),
            name="input_" + input_name,
            antialias=False,
        )

        self.my_output_curve = self.time_plot_up.plot(
            pen=pg.mkPen(
                color=PlotConfig.time_plot_curve_color,
                width=PlotConfig.time_plot_curve_width,
            ),
            name="my_" + output_name,
            antialias=False,
        )

        layout.addWidget(self.time_plot_up)

        # ---- Down graph  ----

        if self.separated:
            self.time_plot_down = pg.PlotWidget()

            # Label the axes
            self.time_plot_down.setLabel("bottom", "Time", units="s")

            # PlotWidget.setXRange(r, padding) is a wrapper that forwards arguments down to ViewBox.setXRange(min, max, padding)
            # At runtime, ViewBox expects min and max. As a result, min and max are passed positionally here, instead of keywords.
            # This works for all the function calls setXRange, setYRange
            self.time_plot_down.setYRange(
                PlotConfig.time_plot_ymin, PlotConfig.time_plot_ymax
            )

            # Create the curve that will be updated in real time
            self.expected_output_curve = self.time_plot_down.plot(
                pen=pg.mkPen(
                    color="g",
                    width=PlotConfig.time_plot_curve_width,
                ),
                name="expected_" + output_name,
                antialias=False,
            )

            # Add down plot to the layout
            layout.addWidget(self.time_plot_down)
            self.resize(1000, 1000)
        else:
            # If not separated, two curves both on the upper graph
            self.expected_output_curve = self.time_plot_up.plot(
                pen=pg.mkPen(
                    color="g",
                    width=PlotConfig.time_plot_curve_width,
                ),
                name="expected_" + output_name,
                antialias=False,
            )

            self.resize(1000, 500)

        self._sample_counter = 0
        self._draw_every = PlotConfig.draw_sample_frequency

    def update_plots(
        self, timestamp: float, input: float, output: float, expected_output: float
    ) -> None:
        """Update both plots using new data.

        :param float timestamp: Timestamp.

        :param float input:  Input.
        :param float output: Output.
        :param float expected_output: Expected output.

        :return: None
        :rtype: None
        """
        # ---- store incoming data ----

        # Append the newest time and signal value to the buffers
        self.time_buf.append(timestamp)
        self.input_buf.append(input)
        self.output_buf.append(output)
        self.expected_output_buf.append(expected_output)

        # ---- draw every certain batch of sample ----
        self._sample_counter += 1
        if self._sample_counter < self._draw_every:
            return

        self._sample_counter = 0

        # ---- sliding time window ----

        # Update the plotted curve with the buffered data
        self.input_curve.setData(self.time_buf, self.input_buf)
        self.my_output_curve.setData(self.time_buf, self.output_buf)
        self.expected_output_curve.setData(self.time_buf, self.expected_output_buf)

        # Automatically move the visible x-range to follow time
        self.time_plot_up.setXRange(
            timestamp + PlotConfig.time_plot_window_follow,
            timestamp + PlotConfig.time_plot_window_lead_sec,
        )

        if self.separated:
            self.time_plot_down.setXRange(
                timestamp + PlotConfig.time_plot_window_follow,
                timestamp + PlotConfig.time_plot_window_lead_sec,
            )
