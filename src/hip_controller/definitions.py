"""Common definitions for this module."""

from dataclasses import asdict, dataclass
from math import pi
from pathlib import Path

import numpy as np

np.set_printoptions(precision=3, floatmode="fixed", suppress=True)


@dataclass(frozen=True)
class ConfigPlot:
    """Plot Configurations for the hip controller."""

    # if the graph is displayed or not
    left_limb_plot: bool = True
    right_limb_plot: bool = False

    GRAPH_WIDTH = 1000
    GRAPH_HEIGHT = 500

    DRAW_SAMPLE_FREQUENCY = 10

    # left time-series graph config
    TIME_PLOT_SIZE: int = 150
    TIME_PLOT_WINDOW_SIZE_SEC = (
        2  # This should align with the plot number size with frequency of the samples
    )

    # sin-wave is always between 1 and -1
    TIME_PLOT_YMIN = -1.1
    TIME_PLOT_YMAX = 1.1

    TIME_PLOT_CURVE_COLOR = (244, 96, 144)
    TIME_PLOT_CURVE_WIDTH = 2
    TIME_PLOT_CURVE_NAME = "<math>-sin(Φ) with lag correction </math>"
    TIME_PLOT_WINDOW_LEAD_SEC = 0.5
    TIME_PLOT_WINDOW_FOLLOW = TIME_PLOT_WINDOW_LEAD_SEC - TIME_PLOT_WINDOW_SIZE_SEC

    # right phase portrait config
    PHASE_PLOT_SIZE: int = 150
    PHASE_PLOT_WINDOW_MARGIN = 1.1

    PHASE_PLOT_AXIS_ANGLE = "Angle (rad)"
    PHASE_PLOT_AXIS_VELOCITY = "Velocity (rad/s)"
    PHASE_PLOT_SCATTER_SIZE = 5

    # Since the spots are fading transparently, RGB values are be given separately
    PHASE_PLOT_SCATTER_COLOR_R = 56
    PHASE_PLOT_SCATTER_COLOR_G = 136
    PHASE_PLOT_SCATTER_COLOR_B = 56
    PHASE_PLOT_LINE_WIDTH = 2
    PHASE_PLOT_LINE_COLOR = "g"


# --- Directories ---

ROOT_DIR: Path = Path(__file__).resolve().parents[2]
# Use the file location to determine the project root reliably.
# This works regardless of the current working directory,
# whereas Path("src").parent depends on where the script is executed from.
DATA_DIR: Path = ROOT_DIR / "data"
TESTING_DIR: Path = ROOT_DIR / "tests"
RECORDINGS_DIR: Path = DATA_DIR / "recordings"
LOG_DIR: Path = DATA_DIR / "logs"


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
STRIDE_EVENT_COUNTER_TIME = 0.3099  # in ms

# ---mid level---
LAG_CORRECTION = pi / 7

# Amplitude modulation
AMPLITUDE_GAIN = -6.5
SIGMOID_POWER = 50
SCALE_LEVEL_GROUND = 1

# Kalman filter definitions
PROCESS_NOISE = 2e-2
MEASUREMENT_NOISE = 0.75


# Cubic Spline Interpolation
@dataclass(frozen=True)
class MotionMapping:
    """Stores breakpoint and table data of the motion mapping."""

    BREAKPOINTS = np.array(
        [-1, -0.8, -0.6, -0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.6, 0.8, 1],
        dtype=np.float64,
    )

    TABLE = np.array(
        [-1, -0.8, -0.6, -0.4, -0.2, -0.1, 0, 0, 0.005, 0.01, 0.015, 0.02, 0.025],
        dtype=np.float64,
    )


# S Gait stopping threshold
STOP_THRESHOLD = 0.5


@dataclass(frozen=True)
class StateChangeTimeThreshold:
    """TMIN and TMAX in seconds."""

    TMIN: float = 0.0
    TMAX: float = 0.6


@dataclass(frozen=True)
class PositionLimitation:
    """Limitations of position steady states."""

    # both are []
    UPPER = 10.0
    LOWER = -10.0


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
