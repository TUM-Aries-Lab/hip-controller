"""Common definitions for this module."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

np.set_printoptions(precision=3, floatmode="fixed", suppress=True)


# --- Directories ---
ROOT_DIR: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = ROOT_DIR / "data"
RECORDINGS_DIR: Path = DATA_DIR / "recordings"
LOG_DIR: Path = DATA_DIR / "logs"
SENSOR_DATA_DIR: Path = DATA_DIR / "sensor_data"
TESTING_HIGH_LEVEL_DIR: Path = SENSOR_DATA_DIR / "high_level_testing"


@dataclass
class HighLevelData:
    """High level data for testing."""

    DATA_ZERO_CROSSING: Path = (
        TESTING_HIGH_LEVEL_DIR / "zero_crossing_left_2026_01_09.csv"
    )

    DATA_VALID_TRIGGER: Path = (
        TESTING_HIGH_LEVEL_DIR / "valid_trigger_left_2026_01_15.csv"
    )

    DATA_EXTREMA_VALUES: Path = TESTING_HIGH_LEVEL_DIR / "extrema_2026_01_26.csv"

    DATA_VEL_SS: Path = TESTING_HIGH_LEVEL_DIR / "vel_ss_2026_01_26.csv"
    DATA_GAIT_PHASE: Path = TESTING_HIGH_LEVEL_DIR / "gait_phase_left_2026_01_21.csv"


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

# High-level controller

# centering & normalization


@dataclass(frozen=True)
class StateChangeTimeThreshold:
    """TMIN and TMAX in seconds."""

    TMIN: float = 0.0
    TMAX: float = 0.6


VALUE_ZERO = 0.0
VALUE_NEAR_ZERO = 1e-6


@dataclass(frozen=True)
class PositionLimitation:
    """Limitations of position steady states."""

    # both are []
    UPPER = 10
    LOWER = -10


# csv column names for testing
@dataclass
class ColumnName:
    """Names of columns."""

    TIMESTAMP: str = "time (s)"
    ANGLE: str = "angle_left (rad)"
    VELOCITY: str = "vel_left (rad/s)"

    TRIGG_VEL_MAX: str = "vel_max_trigg_left"
    TRIGG_ANG_MAX: str = "ang_max_trigg_left"
    TRIGG_VEL_MIN: str = "vel_min_trigg_left"
    TRIGG_ANG_MIN: str = "ang_min_trigg_left"

    VALID_TRIGG_VEL_MAX: str = "valid_vel_max_left"
    VALID_TRIGG_ANG_MAX: str = "valid_ang_max_left"
    VALID_TRIGG_VEL_MIN: str = "valid_vel_min_left"
    VALID_TRIGG_ANG_MIN: str = "valid_ang_min_left"

    VALUE_VEL_MAX: str = "vel_max_left (rad/s)"
    VALUE_VEL_MIN: str = "vel_min_left (rad/s)"
    VEL_GAMMA_T: str = "vel_gamma_t"
    VEL_SUM_MINMAX: str = "sum_vel_minmax"
    VEL_SS: str = "vel_ss"

    VALUE_ANG_MAX: str = "ang_max_left (rad)"
    VALUE_ANG_MIN: str = "ang_min_left (rad)"

    POS_SS: str = "pos_ss"
    Z_T: str = "z_t"

    GAIT_PHASE: str = "gait_phase_left"
