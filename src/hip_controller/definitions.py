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


# Kalman filter definitions
PROCESS_NOISE = 2e-2
MEASUREMENT_NOISE = 0.75

# S Gait stopping threshold
STOP_THRESHOLD = 0.5


# centering & normalization
LAG_CORRECTION = pi / 7
VALUE_NEAR_ZERO = 1e-6


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
class RecordedSensorData:
    """Names of columns of recorded sensor data."""

    # You could try "data_input_filtered_2026_01_09.csv" or "data_kinematics_2026_02_16.csv"
    FILEPATH: Path = DATA_DIR / "sensor_data" / "data_input_filtered_2026_01_09.csv"

    TIMESTAMP: str = "time (s)"
    ANG_LEFT: str = "angle_left (rad)"
    VEL_LEFT: str = "vel_left (rad/s)"
    ANG_RIGHT: str = "angle_right (rad)"
    VEL_RIGHT: str = "vel_right (rad/s)"

    FREQUENCY_HZ: int = 100
