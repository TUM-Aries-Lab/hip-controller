"""Test the plotter module."""

from sys import argv

from PyQt6.QtWidgets import QApplication
from pytest import fixture

from hip_controller.definitions import ConfigPlot
from hip_controller.utils.plotter import PortraitWindow


@fixture(scope="session", autouse=True)
def qapp():
    """Prepare for the test of qt."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(argv)
    return app


def test_plotter_initial_state():
    """Test the initialization of the plotter."""
    window = PortraitWindow(left=True)

    assert len(window.time_buf) == 0
    assert len(window.angle_buf) == 0
    assert window._sample_counter == 0


def test_plotter_buffer_updates():
    """Test the buffer update of the plotter."""
    window = PortraitWindow(left=True)

    window._draw_every = 1

    window.update_plots(
        timestamp=0.0,
        sinusoidal=1.0,
        angle=0.5,
        velocity=0.2,
    )

    assert len(window.time_buf) == 1
    assert len(window.signal_buf) == 1
    assert len(window.angle_buf) == 1
    assert len(window.vel_buf) == 1


def test_plotter_batching_behavior():
    """Test the batching behavior of the plotter."""
    window = PortraitWindow(left=True)

    window._draw_every = 3

    window.update_plots(0.0, 1.0, 0.5, 0.2)
    window.update_plots(0.1, 1.1, 0.6, 0.3)

    assert window._sample_counter == 2

    window.update_plots(0.2, 1.2, 0.7, 0.4)

    assert window._sample_counter == 0


def test_phase_plot_buffer_limit():
    """Test the buffer limit of the plotter."""
    window = PortraitWindow(left=True)

    window._draw_every = 1

    max_len = ConfigPlot.PHASE_PLOT_SIZE

    for i in range(max_len + 5):
        window.update_plots(i, 0.0, float(i), float(i))

    assert len(window.angle_buf) == max_len
    assert len(window.vel_buf) == max_len
