"""Time-domain plotting utilities for real-time visualization of two signals."""

# pragma: no cover
from collections import deque

import pyqtgraph as pg
from PyQt6 import QtWidgets

from hip_controller.definitions import ConfigPlot


class PlotterWindow(QtWidgets.QMainWindow):  # pragma: no cover
    """PyQt6 window that displays two real-time plots."""

    def __init__(self, separated: bool) -> None:
        """Initialize the real-time plotting window.

        :param bool left: Whether this plot is for the left or right limb, used for titles and labels.
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
        self.left_buf = deque()
        self.right_buf = deque()

        # Build all UI components
        self._init_ui()

        self._sample_counter = 0
        self._draw_every = ConfigPlot.DRAW_SAMPLE_FREQUENCY

    def _init_ui(self) -> None:
        """Create and configure all Qt and PyQtGraph widgets.

        This method is called once during initialization and should
        not contain any real-time logic.

        :param bool left: Whether this plot is for the left or right limb, used for titles and labels.
        :return: None
        """
        # Create a central widget to hold everything
        central = QtWidgets.QWidget()

        # these settings take no keyword arguments
        # Vertical layout: left plot + right plot
        layout = QtWidgets.QVBoxLayout(central)

        # Set window title
        self.setWindowTitle("Time domain plotter comparing two signals")

        # Set the central widget of the window
        self.setCentralWidget(central)

        # ---- Up graph  ----
        self.time_plot_up = pg.PlotWidget(title="key")

        # Label the axes
        self.time_plot_up.setLabel("bottom", "Time", units="s")

        # PlotWidget.setXRange(r, padding) is a wrapper that forwards arguments down to ViewBox.setXRange(min, max, padding)
        # At runtime, ViewBox expects min and max. As a result, min and max are passed positionally here, instead of keywords.
        # This works for all the function calls setXRange, setYRange
        self.time_plot_up.setYRange(
            ConfigPlot.TIME_PLOT_YMIN, ConfigPlot.TIME_PLOT_YMAX
        )

        # Create the curve that will be updated in real time
        self.time_curve_up = self.time_plot_up.plot(
            pen=pg.mkPen(
                color=ConfigPlot.TIME_PLOT_CURVE_COLOR,
                width=ConfigPlot.TIME_PLOT_CURVE_WIDTH,
                name=ConfigPlot.TIME_PLOT_CURVE_NAME,
            ),
            antialias=False,
        )

        layout.addWidget(self.time_plot_up)

        # ---- Down graph  ----

        if self.separated:
            self.time_plot_down = pg.PlotWidget(title="value")

            # Label the axes
            self.time_plot_down.setLabel("bottom", "Time", units="s")

            # PlotWidget.setXRange(r, padding) is a wrapper that forwards arguments down to ViewBox.setXRange(min, max, padding)
            # At runtime, ViewBox expects min and max. As a result, min and max are passed positionally here, instead of keywords.
            # This works for all the function calls setXRange, setYRange
            self.time_plot_down.setYRange(
                ConfigPlot.TIME_PLOT_YMIN, ConfigPlot.TIME_PLOT_YMAX
            )

            # Create the curve that will be updated in real time
            self.time_curve_down = self.time_plot_down.plot(
                pen=pg.mkPen(
                    color="g",
                    width=ConfigPlot.TIME_PLOT_CURVE_WIDTH,
                    name=ConfigPlot.TIME_PLOT_CURVE_NAME,
                ),
                antialias=False,
            )

            # Add down plot to the layout
            layout.addWidget(self.time_plot_down)
            self.resize(1000, 1000)
        else:
            # If not separated, two curves both on the upper graph
            self.time_curve_down = self.time_plot_up.plot(
                pen=pg.mkPen(
                    color="g",
                    width=ConfigPlot.TIME_PLOT_CURVE_WIDTH,
                    name=ConfigPlot.TIME_PLOT_CURVE_NAME,
                ),
                antialias=False,
            )

            self.resize(1000, 500)

        self._sample_counter = 0
        self._draw_every = ConfigPlot.DRAW_SAMPLE_FREQUENCY

    def update_plots(
        self, timestamp: float, left_input: float, right_input: float
    ) -> None:
        """Update both plots using new data.

        :param float timestamp: Time stamp.
        :param float sinusoidal: Output signal plotted in the time-domain view.
        :param float angle: Normalized angle in rad.
        :param float velocity: Normalized angular velocity in rad/s.
        """
        # ---- store incoming data ----

        # Append the newest time and signal value to the buffers
        self.time_buf.append(timestamp)
        self.left_buf.append(left_input)
        self.right_buf.append(right_input)

        # ---- draw every certain batch of sample ----
        self._sample_counter += 1
        if self._sample_counter < self._draw_every:
            return

        self._sample_counter = 0

        # ---- sliding time window ----

        # Update the plotted curve with the buffered data
        self.time_curve_up.setData(self.time_buf, self.left_buf)
        self.time_curve_down.setData(self.time_buf, self.right_buf)

        # Automatically move the visible x-range to follow time
        self.time_plot_up.setXRange(
            timestamp + ConfigPlot.TIME_PLOT_WINDOW_FOLLOW,
            timestamp + ConfigPlot.TIME_PLOT_WINDOW_LEAD_SEC,
        )

        if self.separated:
            self.time_plot_down.setXRange(
                timestamp + ConfigPlot.TIME_PLOT_WINDOW_FOLLOW,
                timestamp + ConfigPlot.TIME_PLOT_WINDOW_LEAD_SEC,
            )
