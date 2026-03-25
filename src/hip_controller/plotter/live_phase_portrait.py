"""Plotter using pyqt6."""

# pragma: no cover
from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets

from hip_controller.definitions import PlotConfig, SensorSignal


class PortraitWindow(QtWidgets.QMainWindow):  # pragma: no cover
    """PyQt6 window that displays two real-time plots.

    1) Left plot:
       - Time vs gait phase
       - Sliding time window

    2) Right plot:
       - Phase portrait (angle vs velocity)
       - Fading trail showing temporal evolution

    Data is advanced by a QTimer to keep the GUI responsive.
    """

    def __init__(self, left: bool) -> None:
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

        # Deques for the left (time) plot
        self.time_buf = deque(maxlen=PlotConfig.time_plot_size)
        self.signal_buf = deque(maxlen=PlotConfig.time_plot_size)

        # Deques for the right (phase portrait)
        self.angle_buf = deque(maxlen=PlotConfig.phase_plot_size)
        self.vel_buf = deque(maxlen=PlotConfig.phase_plot_size)

        # Build all UI components
        self._init_ui(left=left)

    def _init_ui(self, left: bool) -> None:
        """Create and configure all Qt and PyQtGraph widgets.

        This method is called once during initialization and should
        not contain any real-time logic.

        :param bool left: Whether this plot is for the left or right limb, used for titles and labels.
        :return: None
        """
        # Create a central widget to hold everything
        central = QtWidgets.QWidget()

        # these settings take no keyword arguments
        # Horizontal layout: left plot + right plot
        layout = QtWidgets.QHBoxLayout(central)

        # Set the central widget of the window
        self.setCentralWidget(central)

        # ---- Left: time-series graph of sinusoidal behavior of the gait phase ----
        self.time_plot = pg.PlotWidget(title="Sin-wave of the gait phase vs Time")

        # Label the axes
        self.time_plot.setLabel("bottom", "Time", units="s")
        self.time_plot.setLabel("left", "<math>-sin(Φ) </math>")

        # PlotWidget.setXRange(r, padding) is a wrapper that forwards arguments down to ViewBox.setXRange(min, max, padding)
        # At runtime, ViewBox expects min and max. As a result, min and max are passed positionally here, instead of keywords.
        # This works for all the function calls setXRange, setYRange
        self.time_plot.setYRange(PlotConfig.time_plot_ymin, PlotConfig.time_plot_ymax)

        # Create the curve that will be updated in real time
        self.time_curve = self.time_plot.plot(
            pen=pg.mkPen(
                color=PlotConfig.time_plot_curve_color,
                width=PlotConfig.time_plot_curve_width,
                name=PlotConfig.time_plot_curve_name,
            )
        )

        # ---- Right: phase portrait ----
        self.phase_plot = pg.PlotWidget(title="Phase Portrait of angle vs velocity")

        # Create the phase portrait plot widget
        self.phase_plot.setLabel("bottom", PlotConfig.phase_plot_axis_angle)
        self.phase_plot.setLabel("left", PlotConfig.phase_plot_axis_velocity)
        self.phase_plot.setAspectLocked(True)

        # Setup for the manual window range
        self.phase_plot.enableAutoRange(x=False, y=False)
        self._phase_max_radius = 0.0

        # Scatter plot for fading phase trajectory
        self.phase_scatter = pg.ScatterPlotItem(size=PlotConfig.phase_plot_scatter_size)
        self.phase_plot.addItem(self.phase_scatter)

        # Line connecting origin (0,0) to newest phase point
        self.phase_vector_line = self.phase_plot.plot(
            [0.0, 0.0],
            [0.0, 0.0],
            pen=pg.mkPen(
                color=PlotConfig.phase_plot_line_color,
                width=PlotConfig.phase_plot_line_width,
            ),
        )

        # Add both plots to the layout
        layout.addWidget(self.time_plot)
        layout.addWidget(self.phase_plot)

        # Set window title and initial size
        self.setWindowTitle(f"Real-Time Phase Portrait ({'Left' if left else 'Right'})")
        self.resize(
            PlotConfig.graph_width, PlotConfig.graph_height
        )  #  resize() takes no keyword arguments

        self._sample_counter = 0
        self._draw_every = PlotConfig.draw_sample_frequency

    def update_plots(
        self, timestamp: float, reference_motor: float, steady: SensorSignal
    ) -> None:
        """Update both plots using new data.

        :param float timestamp: Time stamp.
        :param float reference_motor: Output signal plotted in the time-domain view.
        :param float steady: Normalized angle in rad and normalized angular velocity in rad/s.
        """
        # ---- store incoming data ----

        # Append the newest time and signal value to the buffers
        self.time_buf.append(timestamp)
        self.signal_buf.append(reference_motor)

        # Append the newest phase point
        self.angle_buf.append(steady.angle_rad)
        self.vel_buf.append(steady.velocity_rad_per_sec)

        # ---- draw every certain batch of sample ----
        self._sample_counter += 1
        if self._sample_counter < self._draw_every:
            return

        self._sample_counter = 0

        # ---- sliding time window ----

        # Update the plotted curve with the buffered data
        self.time_curve.setData(self.time_buf, self.signal_buf)

        # Automatically move the visible x-range to follow time
        self.time_plot.setXRange(
            timestamp + PlotConfig.time_plot_window_follow,
            timestamp + PlotConfig.time_plot_window_lead_sec,
        )

        # ---- phase portrait with fading ----

        n = len(self.angle_buf)

        # Generate fading transparency values (old → transparent)
        alphas = np.linspace(start=0, stop=255, num=n).astype(int)

        # Build scatter points with per-point transparency
        spots = [
            dict(
                pos=(self.angle_buf[i], self.vel_buf[i]),
                brush=pg.mkBrush(
                    PlotConfig.phase_plot_scatter_color_r,
                    PlotConfig.phase_plot_scatter_color_g,
                    PlotConfig.phase_plot_scatter_color_b,
                    alphas[i],
                ),
            )
            for i in range(n)
        ]

        # Reset range of the window

        # Compute max absolute radius of current point
        r_current = max(abs(steady.angle_rad), abs(steady.velocity_rad_per_sec))

        # Update stored global max only if bigger
        if r_current > self._phase_max_radius:
            self._phase_max_radius = r_current

            R = PlotConfig.phase_plot_window_margin * self._phase_max_radius

            # Always symmetric around zero
            self.phase_plot.setXRange(-R, R)
            self.phase_plot.setYRange(-R, R)

        # Update the phase portrait scatter plot
        self.phase_scatter.setData(spots)

        # Update the line connecting (0,0) to the current phase point
        self.phase_vector_line.setData(
            [0.0, steady.angle_rad], [0.0, steady.velocity_rad_per_sec]
        )
