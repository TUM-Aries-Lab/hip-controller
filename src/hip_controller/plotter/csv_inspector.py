"""Modular CSV plotter for the hip controller.

Provides a Simulink-Data-Inspector-style GUI for inspecting CSV recordings:

- Vertically stacked subplots with a linked (shared) time axis.
- A *single* signal panel on the left: click a subplot to make it "active",
  then tick which CSV columns appear in that subplot.
- Each subplot has a small overlaid toolbar in its upper-right corner that
  switches mouse-interaction modes:

    * Pan          -- left-drag translates the view
    * T-Zoom       -- left-drag pans; wheel zooms X only (default)
    * Zoom         -- left-drag draws a zoom rectangle; wheel zooms X and Y
    * Pick         -- click a data point to read its value in the status bar

- The X axis is synthesized from the sampling frequency
  (``BasicConfig.frequency`` by default), so the CSV need not carry its own
  time column.

Public entry point: :func:`plot`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pyqtgraph as pg
from loguru import logger
from pandas.api.types import is_numeric_dtype
from PyQt6 import QtCore, QtGui, QtWidgets

from hip_controller.definitions import BasicConfig

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


class _ColorSwatch(QtWidgets.QPushButton):  # pragma: no cover
    """Small color square that opens a color picker when clicked."""

    color_changed = QtCore.pyqtSignal(QtGui.QColor)

    def __init__(
        self, initial: QtGui.QColor, parent: QtWidgets.QWidget | None = None
    ) -> None:
        """Build a swatch displaying ``initial`` and emitting on user changes."""
        super().__init__(parent)
        self._color: QtGui.QColor = QtGui.QColor(initial)
        self.setFixedSize(18, 18)
        self.setToolTip("Click to change this signal's line color.")
        self._refresh_style()
        self.clicked.connect(self._on_clicked)

    def color(self) -> QtGui.QColor:
        """Return the swatch's current color."""
        return QtGui.QColor(self._color)

    def set_color(self, color: QtGui.QColor) -> None:
        """Set the swatch color without emitting ``color_changed``."""
        self._color = QtGui.QColor(color)
        self._refresh_style()

    def _refresh_style(self) -> None:
        rgba = self._color
        self.setStyleSheet(
            f"background-color: rgba({rgba.red()}, {rgba.green()}, "
            f"{rgba.blue()}, {rgba.alpha()});"
            "border: 1px solid #555; border-radius: 2px;",
        )

    def _on_clicked(self) -> None:
        picked = QtWidgets.QColorDialog.getColor(
            self._color,
            self,
            "Pick line color",
        )
        if picked.isValid():
            self._color = picked
            self._refresh_style()
            self.color_changed.emit(picked)


class _SubplotWidget(QtWidgets.QFrame):  # pragma: no cover
    """One subplot: a ``pyqtgraph.PlotWidget`` plus an overlaid mode toolbar.

    Owns its curves and legend so the parent window only has to manage
    high-level layout (how many subplots and which columns go where).

    Signals:

    * ``activated()`` -- emitted on any user interaction inside this subplot;
      the parent uses it to know which subplot the side-panel checkboxes
      should target.
    * ``point_picked(time_sec, value, name)`` -- emitted in Pick mode when the
      user clicks near a data point; the parent displays the readout.
    """

    MODE_PAN: str = "pan"
    MODE_TIME_ZOOM: str = "time_zoom"
    MODE_GENERAL_ZOOM: str = "general_zoom"
    MODE_PICKER: str = "picker"

    activated = QtCore.pyqtSignal()
    point_picked = QtCore.pyqtSignal(float, float, str)

    def __init__(
        self,
        index: int,
        initial_mode: str = MODE_TIME_ZOOM,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Build one subplot with its own plot widget and mode toolbar.

        :param int index: zero-based subplot index, used for the title.
        :param str initial_mode: starting mouse-interaction mode.
        :param QtWidgets.QWidget parent: optional Qt parent.
        """
        super().__init__(parent)
        self.setObjectName("subplotFrame")
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        self._index: int = index
        self._mode: str = initial_mode

        self.plot_widget: pg.PlotWidget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("bottom", "time", units="s")
        self.plot_widget.setTitle(f"Subplot {index + 1}")
        self._legend: pg.LegendItem = self.plot_widget.addLegend(offset=(10, 10))

        # Curves currently displayed: column name → PlotDataItem.
        self.curves: dict[str, pg.PlotDataItem] = {}

        # Marker shown in Pick mode.
        self._pick_marker: pg.ScatterPlotItem | None = None
        self._pick_label: pg.TextItem | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot_widget)

        self._toolbar, self._mode_buttons = self._build_mode_toolbar()
        self._toolbar.setParent(self)
        self._toolbar.raise_()

        self.set_active(False)
        # scene() is typed as Optional in PyQt stubs and sigMouseClicked is a
        # pyqtgraph-specific signal that PyQt's stubs don't know about.
        scene = self.plot_widget.scene()
        assert scene is not None
        scene.sigMouseClicked.connect(self._on_scene_clicked)  # pyright: ignore[reportAttributeAccessIssue]
        self.set_mode(self._mode)

    # --- toolbar construction ----------------------------------------

    def _build_mode_toolbar(
        self,
    ) -> tuple[QtWidgets.QFrame, dict[str, QtWidgets.QToolButton]]:
        """Build the floating mode toolbar shown in the plot's upper-right corner."""
        bar = QtWidgets.QFrame()
        bar.setObjectName("modeBar")
        bar.setStyleSheet(
            "#modeBar { background: rgba(255, 255, 255, 220); "
            "border: 1px solid #888; border-radius: 4px; }"
            "QToolButton { padding: 2px 6px; }"
            "QToolButton:checked { background: #cfe1f7; border: 1px solid #4a90e2; "
            "border-radius: 3px; }",
        )
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(3, 3, 3, 3)
        row.setSpacing(2)

        buttons: dict[str, QtWidgets.QToolButton] = {}
        group = QtWidgets.QButtonGroup(bar)
        group.setExclusive(True)

        entries: list[tuple[str, str, str]] = [
            (self.MODE_PAN, "Pan", "Pan: left-drag translates the view."),
            (
                self.MODE_TIME_ZOOM,
                "T-Zoom",
                "Time-only zoom: wheel zooms the X axis; Y auto-fits (default).",
            ),
            (
                self.MODE_GENERAL_ZOOM,
                "Zoom",
                "General zoom: left-drag draws a zoom rectangle; wheel zooms X and Y.",
            ),
            (
                self.MODE_PICKER,
                "Pick",
                "Data cursor: click near a curve point to read its value.",
            ),
        ]
        for mode, label, tooltip in entries:
            btn = QtWidgets.QToolButton()
            btn.setText(label)
            btn.setToolTip(tooltip)
            btn.setCheckable(True)
            btn.setAutoRaise(True)
            btn.clicked.connect(self.activated.emit)
            btn.clicked.connect(lambda _checked, m=mode: self.set_mode(m))
            group.addButton(btn)
            row.addWidget(btn)
            buttons[mode] = btn
        return bar, buttons

    # --- curve management --------------------------------------------

    def add_curve(
        self,
        name: str,
        time_sec: np.ndarray,
        y_values: np.ndarray,
        pen: QtGui.QPen,
    ) -> None:
        """Plot one column on this subplot (no-op if already present)."""
        if name in self.curves:
            return
        item = self.plot_widget.plot(time_sec, y_values, pen=pen, name=name)
        self.curves[name] = item
        self._update_left_label()

    def remove_curve(self, name: str) -> None:
        """Remove one column from this subplot (no-op if absent)."""
        item = self.curves.pop(name, None)
        if item is None:
            return
        self.plot_widget.removeItem(item)
        try:
            self._legend.removeItem(name)
        except (KeyError, AttributeError):
            # Older pyqtgraph builds may raise if the entry is already gone.
            pass
        self._clear_pick_marker()
        self._update_left_label()

    def remove_all_curves(self) -> None:
        """Remove every curve currently shown on this subplot."""
        for name in list(self.curves.keys()):
            self.remove_curve(name)

    def set_curve_pen(self, name: str, pen: QtGui.QPen) -> None:
        """Update an existing curve's pen (color/width) in place."""
        item = self.curves.get(name)
        if item is None:
            return
        item.setPen(pen)
        # Re-stamp the legend sample so its swatch reflects the new pen.
        try:
            self._legend.removeItem(name)
        except (KeyError, AttributeError):
            pass
        self._legend.addItem(item, name)

    def _update_left_label(self) -> None:
        """Show the column name on the Y axis when exactly one curve is plotted."""
        if len(self.curves) == 1:
            self.plot_widget.setLabel("left", next(iter(self.curves)))
        else:
            self.plot_widget.setLabel("left", "")

    # --- mode handling -----------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Switch this subplot's mouse-interaction mode.

        :param str mode: one of the ``MODE_*`` class constants.
        """
        self._mode = mode
        btn = self._mode_buttons.get(mode)
        if btn is not None and not btn.isChecked():
            btn.setChecked(True)
        self._apply_mode()
        if mode != self.MODE_PICKER:
            self._clear_pick_marker()

    def _apply_mode(self) -> None:
        """Configure ViewBox and cursor to match ``self._mode``."""
        plot_item = self.plot_widget.getPlotItem()
        assert plot_item is not None
        view_box = plot_item.getViewBox()
        assert view_box is not None
        viewport = self.plot_widget.viewport()
        assert viewport is not None
        if self._mode == self.MODE_PAN:
            view_box.setMouseMode(pg.ViewBox.PanMode)
            view_box.setMouseEnabled(x=True, y=True)
            viewport.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        elif self._mode == self.MODE_TIME_ZOOM:
            view_box.setMouseMode(pg.ViewBox.PanMode)
            view_box.setMouseEnabled(x=True, y=False)
            # Intentionally do NOT re-enable Y auto-range here: switching INTO
            # T-Zoom should preserve whatever Y range the user has set in
            # another mode. Y is just locked from mouse input, not refit.
            viewport.setCursor(QtCore.Qt.CursorShape.SizeHorCursor)
        elif self._mode == self.MODE_GENERAL_ZOOM:
            view_box.setMouseMode(pg.ViewBox.RectMode)
            view_box.setMouseEnabled(x=True, y=True)
            viewport.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        elif self._mode == self.MODE_PICKER:
            view_box.setMouseMode(pg.ViewBox.PanMode)
            view_box.setMouseEnabled(x=False, y=False)
            viewport.setCursor(QtCore.Qt.CursorShape.CrossCursor)

    # --- active styling ----------------------------------------------

    def set_active(self, active: bool) -> None:
        """Toggle the visual highlight that marks the active subplot."""
        if active:
            self.setStyleSheet(
                "#subplotFrame { border: 2px solid #4a90e2; border-radius: 3px; }",
            )
        else:
            self.setStyleSheet(
                "#subplotFrame { border: 1px solid #cccccc; border-radius: 3px; }",
            )

    # --- click handling (activate + picker) --------------------------

    def _on_scene_clicked(self, event: Any) -> None:
        """Activate this subplot on any click; in Pick mode, report the nearest point.

        ``event`` is a ``pg.GraphicsScene.mouseEvents.MouseClickEvent`` at
        runtime, but that internal pyqtgraph type isn't exposed via stubs, so
        we accept ``Any`` rather than chasing a private import.
        """
        self.activated.emit()
        if self._mode != self.MODE_PICKER:
            return
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        plot_item = self.plot_widget.getPlotItem()
        assert plot_item is not None
        view_box = plot_item.getViewBox()
        assert view_box is not None
        scene_pos = event.scenePos()
        if not self.plot_widget.sceneBoundingRect().contains(scene_pos):
            return
        view_point = view_box.mapSceneToView(scene_pos)
        nearest = self._find_nearest_point(
            click_x=float(view_point.x()),
            click_y=float(view_point.y()),
        )
        if nearest is None:
            return
        time_sec, value, name = nearest
        self._show_pick_marker(time_sec=time_sec, value=value, name=name)
        self.point_picked.emit(time_sec, value, name)

    def _find_nearest_point(
        self,
        click_x: float,
        click_y: float,
    ) -> tuple[float, float, str] | None:
        """Return ``(x, y, curve_name)`` for the data point closest to the click."""
        plot_item = self.plot_widget.getPlotItem()
        assert plot_item is not None
        view_box = plot_item.getViewBox()
        assert view_box is not None
        pixel_w, pixel_h = view_box.viewPixelSize()
        pixel_w = pixel_w or 1.0
        pixel_h = pixel_h or 1.0

        best: tuple[float, float, str] | None = None
        best_dist = float("inf")
        for name, item in self.curves.items():
            data = item.getData()
            if data is None:
                continue
            x_data, y_data = data
            if x_data is None or y_data is None or len(x_data) == 0:
                continue
            idx = int(np.argmin(np.abs(x_data - click_x)))
            x_val = float(x_data[idx])
            y_val = float(y_data[idx])
            dx = (x_val - click_x) / pixel_w
            dy = (y_val - click_y) / pixel_h
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best = (x_val, y_val, name)
        return best

    def _show_pick_marker(self, time_sec: float, value: float, name: str) -> None:
        """Draw / move the picker marker and its text label at the given point."""
        if self._pick_marker is None:
            self._pick_marker = pg.ScatterPlotItem(
                size=12,
                pen=pg.mkPen("k", width=1),
                brush=pg.mkBrush(255, 80, 80, 220),
            )
            self.plot_widget.addItem(self._pick_marker)
        self._pick_marker.setData([time_sec], [value])

        if self._pick_label is None:
            self._pick_label = pg.TextItem(anchor=(0.0, 1.0), color="k")
            self.plot_widget.addItem(self._pick_label)
        self._pick_label.setText(f"{name}\nt={time_sec:.3f}s, y={value:.4g}")
        self._pick_label.setPos(time_sec, value)

    def _clear_pick_marker(self) -> None:
        """Remove the picker marker / label from this subplot if present."""
        if self._pick_marker is not None:
            self.plot_widget.removeItem(self._pick_marker)
            self._pick_marker = None
        if self._pick_label is not None:
            self.plot_widget.removeItem(self._pick_label)
            self._pick_label = None

    # --- layout ------------------------------------------------------

    def resizeEvent(self, a0: QtGui.QResizeEvent | None) -> None:  # noqa: N802
        """Keep the mode toolbar anchored to the upper-right corner.

        ``a0`` is named to match the PyQt6 base-class signature so the
        override is recognized by the type checker.
        """
        super().resizeEvent(a0)
        self._toolbar.adjustSize()
        margin = 6
        x = self.width() - self._toolbar.width() - margin
        self._toolbar.move(max(0, x), margin)


class CSVInspectorWindow(QtWidgets.QMainWindow):  # pragma: no cover
    """Main window for the modular CSV plotter.

    The user picks the number of subplots from the left panel, clicks a
    subplot to make it active, then ticks which CSV columns appear in it.
    All subplots share their X axis, so panning / zooming time stays in
    sync.
    """

    _MAX_SUBPLOTS: int = 8

    # Class-level reference list that keeps every open inspector window alive
    # so the garbage collector doesn't reap one when the "Open in New Window"
    # handler returns. Cleared per-window in closeEvent.
    _open_windows: ClassVar[list[CSVInspectorWindow]] = []

    def __init__(
        self,
        csv_path: Path,
        frequency_hz: int = BasicConfig.frequency,
        time_only_zoom: bool = True,
    ) -> None:
        """Build the GUI for one CSV file.

        :param pathlib.Path csv_path: path to a CSV file with a header row.
        :param int frequency_hz: sampling frequency in Hz used to synthesize
            the time axis. Defaults to ``BasicConfig.frequency``.
        :param bool time_only_zoom: starting mouse-interaction mode for every
            subplot. ``True`` (default) selects the time-only zoom mode, which
            matches the Simulink Data Inspector feel. ``False`` selects the
            general (X+Y) zoom mode.
        """
        super().__init__()

        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        pg.setConfigOption("antialias", True)

        self._csv_path: Path = Path(csv_path)
        self._frequency_hz: int = int(frequency_hz)
        self._initial_mode: str = (
            _SubplotWidget.MODE_TIME_ZOOM
            if time_only_zoom
            else _SubplotWidget.MODE_GENERAL_ZOOM
        )

        self._dataframe: pd.DataFrame = pd.DataFrame()
        self._columns: list[str] = []
        self._time_sec: np.ndarray = np.empty(0, dtype=np.float64)
        self._column_colors: dict[str, QtGui.QColor] = {}
        self._subplot_signals: list[set[str]] = []
        self._subplots: list[_SubplotWidget] = []
        self._signal_checkboxes: dict[str, QtWidgets.QCheckBox] = {}
        self._signal_swatches: dict[str, _ColorSwatch] = {}
        self._active_index: int = 0

        self._load_csv(self._csv_path)
        self._init_column_colors()
        self._subplot_signals = [set(self._columns[: min(2, len(self._columns))])]

        self.resize(1200, 800)
        self.setWindowTitle(f"CSV Inspector — {self._csv_path.name}")
        status_bar = self.statusBar()
        assert status_bar is not None
        status_bar.showMessage("Ready.")

        self._build_menu()
        self._build_layout()
        self._apply_layout()

        # Register so a strong reference outlives the constructing scope.
        CSVInspectorWindow._open_windows.append(self)

    # --- data loading -------------------------------------------------

    def _init_column_colors(self) -> None:
        """Assign a stable default color to each column from pyqtgraph's palette."""
        self._column_colors = {}
        hues = max(len(self._columns), 6)
        for idx, col in enumerate(self._columns):
            self._column_colors[col] = pg.intColor(idx, hues=hues)

    def _load_csv(self, csv_path: Path) -> None:
        """Read a CSV from disk and refresh ``_columns`` / ``_time_sec``.

        :raises ValueError: if the CSV exposes no numeric, non-time columns.
        """
        logger.info(f"Loading CSV '{csv_path}'.")
        self._dataframe = pd.read_csv(csv_path)
        self._columns = discover_plottable_columns(self._dataframe)
        if not self._columns:
            raise ValueError(
                f"CSV '{csv_path}' has no numeric (non-time) columns to plot.",
            )
        self._time_sec = synthesize_time_vector(
            n_samples=len(self._dataframe),
            frequency_hz=self._frequency_hz,
        )

    # --- UI construction ----------------------------------------------

    def _build_menu(self) -> None:
        """Construct the File and View menus.

        QMainWindow.menuBar(), QMenuBar.addMenu(), and QMenu.addAction() are
        all typed as Optional in the PyQt stubs even though they always return
        a real object on a QMainWindow that owns a menu bar. Asserts narrow
        the types for the checker without changing runtime behavior.
        """
        menu = self.menuBar()
        assert menu is not None
        file_menu = menu.addMenu("&File")
        assert file_menu is not None

        open_action = file_menu.addAction("&Open CSV…")
        assert open_action is not None
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open_csv)

        open_new_action = file_menu.addAction("Open CSV in &New Window…")
        assert open_new_action is not None
        open_new_action.setShortcut("Ctrl+Shift+O")
        open_new_action.triggered.connect(self._on_open_csv_new_window)

        file_menu.addSeparator()
        quit_action = file_menu.addAction("&Quit")
        assert quit_action is not None
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)

        view_menu = menu.addMenu("&View")
        assert view_menu is not None
        reset_action = view_menu.addAction("Reset view (auto-range)")
        assert reset_action is not None
        reset_action.setShortcut("Ctrl+R")
        reset_action.triggered.connect(self._on_reset_view)

    def _build_layout(self) -> None:
        """Build the central plot area and the left-side signal panel."""
        self._plot_area = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.setCentralWidget(self._plot_area)

        dock = QtWidgets.QDockWidget("Signals", self)
        dock.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | QtCore.Qt.DockWidgetArea.RightDockWidgetArea,
        )

        panel = QtWidgets.QWidget(dock)
        outer = QtWidgets.QVBoxLayout(panel)
        outer.setContentsMargins(8, 8, 8, 8)

        outer.addWidget(QtWidgets.QLabel(f"<b>File:</b> {self._csv_path.name}"))
        outer.addWidget(
            QtWidgets.QLabel(
                f"<b>X axis:</b> time (s) synthesized at {self._frequency_hz} Hz",
            ),
        )

        count_row = QtWidgets.QHBoxLayout()
        count_row.addWidget(QtWidgets.QLabel("Number of subplots:"))
        self._count_spin = QtWidgets.QSpinBox()
        self._count_spin.setRange(1, self._MAX_SUBPLOTS)
        self._count_spin.setValue(len(self._subplot_signals))
        self._count_spin.valueChanged.connect(self._on_subplot_count_changed)
        count_row.addWidget(self._count_spin)
        count_row.addStretch(1)
        outer.addLayout(count_row)

        outer.addWidget(_make_separator())

        self._active_label = QtWidgets.QLabel()
        self._active_label.setStyleSheet("font-weight: bold;")
        outer.addWidget(self._active_label)

        outer.addWidget(
            QtWidgets.QLabel("Tick a column to add it to the active subplot:"),
        )

        signals_box = QtWidgets.QGroupBox("Signals")
        signals_layout = QtWidgets.QVBoxLayout(signals_box)
        self._populate_signal_rows(signals_layout)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(signals_box)
        outer.addWidget(scroll, stretch=1)

        dock.setWidget(panel)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    # --- layout sync --------------------------------------------------

    def _apply_layout(self) -> None:
        """Reconcile the GUI with the current ``_subplot_signals`` state."""
        n_subplots = len(self._subplot_signals)
        self._sync_subplots(n_subplots)
        self._active_index = min(self._active_index, n_subplots - 1)
        self._refresh_curves()
        self._refresh_signal_checkboxes()
        self._refresh_active_indicator()
        for subplot in self._subplots:
            subplot.plot_widget.enableAutoRange(axis="x", enable=True)

    def _sync_subplots(self, n_subplots: int) -> None:
        """Add or remove subplot widgets to match ``n_subplots`` and re-link X axes."""
        while len(self._subplots) > n_subplots:
            subplot = self._subplots.pop()
            subplot.remove_all_curves()
            subplot.setParent(None)
            subplot.deleteLater()

        while len(self._subplots) < n_subplots:
            idx = len(self._subplots)
            subplot = _SubplotWidget(index=idx, initial_mode=self._initial_mode)
            subplot.activated.connect(
                lambda i=idx: self._on_subplot_activated(i),
            )
            subplot.point_picked.connect(self._on_point_picked)
            self._plot_area.addWidget(subplot)
            self._subplots.append(subplot)

        if self._subplots:
            base_view = self._subplots[0].plot_widget
            for subplot in self._subplots[1:]:
                subplot.plot_widget.setXLink(base_view)

    def _refresh_curves(self) -> None:
        """Reconcile curves on every subplot against ``_subplot_signals``."""
        for i, subplot in enumerate(self._subplots):
            desired = self._subplot_signals[i]
            for stale in set(subplot.curves.keys()) - desired:
                subplot.remove_curve(stale)
            for col in sorted(desired):
                if col in subplot.curves:
                    continue
                y_values = self._dataframe[col].to_numpy(dtype=np.float64)
                subplot.add_curve(
                    name=col,
                    time_sec=self._time_sec,
                    y_values=y_values,
                    pen=self._pen_for_column(col),
                )

    def _refresh_signal_checkboxes(self) -> None:
        """Sync the side-panel checkboxes to the active subplot's signal set."""
        if not self._subplots:
            return
        active_set = self._subplot_signals[self._active_index]
        for col, cb in self._signal_checkboxes.items():
            cb.blockSignals(True)
            cb.setChecked(col in active_set)
            cb.blockSignals(False)

    def _refresh_active_indicator(self) -> None:
        """Update the 'Editing: Subplot N' label and per-subplot border highlight."""
        for i, subplot in enumerate(self._subplots):
            subplot.set_active(i == self._active_index)
        if self._subplots:
            self._active_label.setText(f"Editing: Subplot {self._active_index + 1}")
        else:
            self._active_label.setText("")

    def _pen_for_column(self, column: str) -> QtGui.QPen:
        """Return the pen for ``column`` using its currently-selected color."""
        color = self._column_colors.get(column) or QtGui.QColor("#888888")
        return pg.mkPen(color=color, width=2)

    def _populate_signal_rows(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Build one [color-swatch][checkbox] row per column into ``layout``."""
        for col in self._columns:
            row = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)

            swatch = _ColorSwatch(self._column_colors[col])
            swatch.color_changed.connect(
                lambda color, name=col: self._on_color_changed(name, color),
            )
            row_layout.addWidget(swatch)

            cb = QtWidgets.QCheckBox(col)
            cb.toggled.connect(
                lambda checked, name=col: self._on_signal_toggled(name, checked),
            )
            row_layout.addWidget(cb, stretch=1)

            layout.addWidget(row)
            self._signal_checkboxes[col] = cb
            self._signal_swatches[col] = swatch
        layout.addStretch(1)

    # --- slots --------------------------------------------------------

    def _on_subplot_count_changed(self, value: int) -> None:
        """Handle the subplot-count spin box: grow or shrink the model."""
        current = len(self._subplot_signals)
        if value > current:
            for _ in range(value - current):
                self._subplot_signals.append(set())
        else:
            self._subplot_signals = self._subplot_signals[:value]
        self._apply_layout()

    def _on_signal_toggled(self, column: str, checked: bool) -> None:
        """Add or remove ``column`` from the active subplot."""
        if not self._subplots:
            return
        target = self._subplot_signals[self._active_index]
        if checked:
            target.add(column)
        else:
            target.discard(column)
        self._refresh_curves()

    def _on_subplot_activated(self, index: int) -> None:
        """Make ``index`` the active subplot (the one the side panel edits)."""
        if index == self._active_index:
            return
        self._active_index = index
        self._refresh_signal_checkboxes()
        self._refresh_active_indicator()

    def _on_point_picked(self, time_sec: float, value: float, name: str) -> None:
        """Display the picked data point in the status bar."""
        status_bar = self.statusBar()
        assert status_bar is not None
        status_bar.showMessage(
            f"{name}    t = {time_sec:.4f} s    y = {value:.6g}",
        )

    def _on_color_changed(self, column: str, color: QtGui.QColor) -> None:
        """Update the stored color for ``column`` and restyle every curve using it."""
        self._column_colors[column] = QtGui.QColor(color)
        pen = self._pen_for_column(column)
        for subplot in self._subplots:
            subplot.set_curve_pen(column, pen)
        swatch = self._signal_swatches.get(column)
        if swatch is not None:
            swatch.set_color(color)

    def _on_reset_view(self) -> None:
        """Auto-range both axes on every subplot."""
        for subplot in self._subplots:
            subplot.plot_widget.enableAutoRange(axis="x", enable=True)
            subplot.plot_widget.enableAutoRange(axis="y", enable=True)

    def _on_open_csv_new_window(self) -> None:
        """Open a CSV in a *new* inspector window (this one stays open).

        Useful for comparing two or more recordings side-by-side. The new
        window is appended to ``CSVInspectorWindow._open_windows`` so it
        survives past the end of this method.
        """
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open CSV in New Window",
            str(self._csv_path.parent),
            "CSV files (*.csv)",
        )
        if not path_str:
            return
        try:
            new_window = CSVInspectorWindow(
                csv_path=Path(path_str),
                frequency_hz=self._frequency_hz,
                time_only_zoom=self._initial_mode == _SubplotWidget.MODE_TIME_ZOOM,
            )
        except (ValueError, OSError, pd.errors.ParserError) as exc:
            QtWidgets.QMessageBox.critical(self, "Failed to load CSV", str(exc))
            return
        new_window.show()

    def closeEvent(self, a0: QtGui.QCloseEvent | None) -> None:  # noqa: N802
        """Drop ourselves from the global open-windows list on close.

        ``a0`` is named to match the PyQt6 base-class signature so the
        override is recognized by the type checker.
        """
        try:
            CSVInspectorWindow._open_windows.remove(self)
        except ValueError:
            pass
        super().closeEvent(a0)

    def _on_open_csv(self) -> None:
        """Open a new CSV in the running window via a file dialog."""
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open CSV",
            str(self._csv_path.parent),
            "CSV files (*.csv)",
        )
        if not path_str:
            return
        new_path = Path(path_str)
        try:
            self._load_csv(new_path)
        except (ValueError, OSError, pd.errors.ParserError) as exc:
            QtWidgets.QMessageBox.critical(self, "Failed to load CSV", str(exc))
            return

        self._csv_path = new_path
        self.setWindowTitle(f"CSV Inspector — {new_path.name}")

        for subplot in self._subplots:
            subplot.remove_all_curves()
            subplot.setParent(None)
            subplot.deleteLater()
        self._subplots.clear()

        self._init_column_colors()
        self._subplot_signals = [set(self._columns[: min(2, len(self._columns))])]
        self._active_index = 0
        self._count_spin.blockSignals(True)
        self._count_spin.setValue(1)
        self._count_spin.blockSignals(False)
        self._rebuild_signal_checkboxes()
        self._apply_layout()

    def _rebuild_signal_checkboxes(self) -> None:
        """Rebuild the side-panel signal rows against the current columns."""
        signals_box = self._find_signals_groupbox()
        if signals_box is None:
            return
        layout = signals_box.layout()
        # _build_layout() always installs a QVBoxLayout here, but QGroupBox.layout()
        # is typed as Optional[QLayout]. Narrow it for both pyright and runtime.
        if not isinstance(layout, QtWidgets.QVBoxLayout):
            return
        while layout.count():
            item = layout.takeAt(0)
            if item is None:
                break
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._signal_checkboxes.clear()
        self._signal_swatches.clear()
        self._populate_signal_rows(layout)

    def _find_signals_groupbox(self) -> QtWidgets.QGroupBox | None:
        """Locate the 'Signals' group box inside the side-panel dock."""
        for dock in self.findChildren(QtWidgets.QDockWidget):
            for box in dock.findChildren(QtWidgets.QGroupBox):
                if box.title() == "Signals":
                    return box
        return None


def _make_separator() -> QtWidgets.QFrame:  # pragma: no cover
    """Return a thin horizontal divider for use in the side panel."""
    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
    line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
    return line


def plot(
    csv_path: str | Path,
    frequency_hz: int = BasicConfig.frequency,
    time_only_zoom: bool = True,
) -> None:
    """Open the modular CSV inspector for the given file.

    The time axis is synthesized as ``numpy.arange(n_samples) / frequency_hz``;
    no time column is required in the CSV.

    :param csv_path: path to a CSV file with a header row.
    :type csv_path: str or pathlib.Path
    :param int frequency_hz: sampling frequency in Hz used to synthesize the
        time axis. Defaults to ``BasicConfig.frequency``.
    :param bool time_only_zoom: starting interaction mode for every subplot.
        ``True`` (default) is the Simulink-Data-Inspector-style time-only
        zoom; ``False`` is general (X + Y) zoom. Either mode can also be
        switched per subplot from its in-plot toolbar.
    """
    path = Path(csv_path)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = CSVInspectorWindow(
        csv_path=path,
        frequency_hz=frequency_hz,
        time_only_zoom=time_only_zoom,
    )
    window.show()
    app.exec()
