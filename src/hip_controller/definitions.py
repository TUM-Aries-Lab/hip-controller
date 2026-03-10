"""Common definitions for this module."""

from dataclasses import asdict, dataclass
from math import pi
from pathlib import Path

import numpy as np

np.set_printoptions(precision=3, floatmode="fixed", suppress=True)

# --- Directories ---

ROOT_DIR: Path = Path(__file__).resolve().parents[2]
# Use the file location to determine the project root reliably.
# This works regardless of the current working directory,
# whereas Path("src").parent depends on where the script is executed from.
DATA_DIR: Path = ROOT_DIR / "data"
TESTING_DIR: Path = ROOT_DIR / "tests"
RECORDINGS_DIR: Path = DATA_DIR / "recordings"
LOG_DIR: Path = DATA_DIR / "logs"


@dataclass(frozen=True)
class BasicConfig:
    """Basic configurations for the hip controller."""

    # if the graph is displayed or not
    left_limb_plot: bool = True
    right_limb_plot: bool = True

    # if the wiring settings are reversed or not
    left_limb_reverse: bool = False
    right_limb_reverse: bool = True

    # either read data from imu or read data from csv file using csv player
    read_from_imu: bool = False

    # the path where data is read from
    read_data_from_path: Path = (
        DATA_DIR / "sensor_data" / "data_input_filtered_2026_01_09.csv"
    )


# Default encoding
ENCODING: str = "utf-8"

DATE_FORMAT = "%Y-%m-%d_%H-%M-%S"


@dataclass
class LogLevel:
    """Log level."""

    trace: str = "TRACE"
    debug: str = "DEBUG"
    info: str = "INFO"
    success: str = "SUCCESS"
    warning: str = "WARNING"
    error: str = "ERROR"
    critical: str = "CRITICAL"

    def __iter__(self):
        """Iterate over log levels."""
        return iter(asdict(self).values())


DEFAULT_LOG_LEVEL = LogLevel.info
DEFAULT_LOG_FILENAME = "log_file"

# ---high level---
# centering & normalization
VALUE_NEAR_ZERO = 1e-6

# stride event detector
STRIDE_EVENT_COUNTER_TIME = 0.3099  # in s
STRIDE_EVENT_HIT_CROSSING_OFFSET = -0.1

# ---mid level---
LAG_COMPENSATION = 0  # Lag correction

# Amplitude modulation
SCALE_LEVEL_MODE = 1
SIGMOID_POWER = 50
AMPLITUDE_GAIN = -6.5  # Motor position desidered amplitude (rad)

# Kalman filter definitions
PROCESS_NOISE = 2e-2
MEASUREMENT_NOISE = 0.75


# Cubic Spline Interpolation
@dataclass(frozen=True)
class LookUpTable:
    """Stores breakpoint and table data of the motion mapping."""

    breakpoints = np.array(
        [-1, -0.8, -0.6, -0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.6, 0.8, 1],
        dtype=np.float64,
    )

    tabledata = np.array(
        [-1, -0.8, -0.6, -0.4, -0.2, -0.1, 0, 0, 0.005, 0.01, 0.015, 0.02, 0.025],
        dtype=np.float64,
    )


# S Gait stopping threshold
STOP_THRESHOLD = 0.5


@dataclass(frozen=True)
class StateChangeTimeThreshold:
    """TMIN and TMAX in seconds."""

    tmin: float = 0.0
    tmax: float = 0.6


@dataclass(frozen=True)
class PositionLimitation:
    """Limitations of position steady states."""

    # both are []
    upper = 600 * pi / 180
    lower = -600 * pi / 180


@dataclass
class SensorSignal:
    """Container for timestamp, angle and velocity measurements from the sensor.

    Represents a single snapshot of kinematic data (angle and velocity) read from
    the hip joint sensor at a specific point in time. Used throughout the control
    system to maintain consistent representation of joint state.

    :angle_rad: hip angle of the lower limb in radians.
    :velocity_rad_per_sec: hip angle velocity of the lower limb in radians per second.
    """

    angle_rad: float = 0.0
    velocity_rad_per_sec: float = 0.0


@dataclass
class ExosuitData:
    """Container for the measurements from the sensor of both lower limbs.

    :timestamp: current timestamp.
    :left: Signal state of the left lower limb.
    :right: Signal state of the right lower limb.
    """

    timestamp: float
    left: SensorSignal
    right: SensorSignal


@dataclass
class RecordedSensorData:
    """Names of columns of recorded sensor data."""

    timestamp: str = "time (s)"
    ang_left: str = "angle_left (rad)"
    vel_left: str = "vel_left (rad/s)"
    ang_right: str = "angle_right (rad)"
    vel_right: str = "vel_right (rad/s)"

    fake_frequency_hz: int = 100


@dataclass(frozen=True)
class ConfigPlot:
    """Plot Configurations for the hip controller."""

    graph_width = 1000
    graph_height = 500

    draw_sample_frequency = 10

    # left time-series graph config
    time_plot_size: int = 150
    time_plot_window_size_sec = (
        2  # This should align with the plot number size with frequency of the samples
    )

    # Motor command has a range between about [-10.472, 10.472]
    time_plot_ymin = -11
    time_plot_ymax = 11

    time_plot_curve_color = (244, 96, 144)
    time_plot_curve_width = 2
    time_plot_curve_name = "Reference motion motor command"
    time_plot_window_lead_sec = 0.5
    time_plot_window_follow = time_plot_window_lead_sec - time_plot_window_size_sec

    # right phase portrait config
    phase_plot_size: int = 150
    phase_plot_window_margin = 1.1

    phase_plot_axis_angle = "Angle (rad)"
    phase_plot_axis_velocity = "Velocity (rad/s)"
    phase_plot_scatter_size = 5

    # Since the spots are fading transparently, RGB values are be given separately
    phase_plot_scatter_color_r = 56
    phase_plot_scatter_color_g = 136
    phase_plot_scatter_color_b = 56
    phase_plot_line_width = 2
    phase_plot_line_color = "g"
