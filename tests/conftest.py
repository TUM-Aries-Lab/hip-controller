"""Code to help initialize pytest."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from src.hip_controller.definitions import TESTING_DIR

# Add the src directory to the path so that the quaternion_ekf package can be imported
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(my_path, "../src"))


# Offers path and column strings for testing.
@dataclass
class HighLevelData:
    """High level data for testing."""

    TESTING_HIGH_LEVEL_DIR: Path = (
        TESTING_DIR
        / "controller_test"
        / "high_level_controller"
        / "high_level_testing_data"
    )
    DATA_ZERO_CROSSING: Path = (
        TESTING_HIGH_LEVEL_DIR / "zero_crossing_left_2026_01_09.csv"
    )

    DATA_VALID_TRIGGER: Path = (
        TESTING_HIGH_LEVEL_DIR / "valid_trigger_left_2026_01_15.csv"
    )

    DATA_EXTREMA_VALUES: Path = TESTING_HIGH_LEVEL_DIR / "extrema_2026_01_26.csv"

    DATA_VEL_SS: Path = TESTING_HIGH_LEVEL_DIR / "vel_ss_2026_01_26.csv"
    DATA_ANG_SS: Path = TESTING_HIGH_LEVEL_DIR / "ang_ss_2026_01_26.csv"
    DATA_GAIT_PHASE: Path = TESTING_HIGH_LEVEL_DIR / "gait_phase_left_2026_01_21.csv"


@dataclass
class CSVColumnName:
    """Names of columns for csv files for high-level controller testing."""

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
    VEL_STEADY_STATE: str = "vel_ss"

    VALUE_ANG_MAX: str = "ang_max_left (rad)"
    VALUE_ANG_MIN: str = "ang_min_left (rad)"
    ANG_GAMMA_T: str = "ang_gamma_t"
    ANG_STEADY_STATE: str = "ang_ss"

    POSTION_STEADY_STATE: str = "pos_ss"
    Z_T: str = "z_t"

    GAIT_PHASE: str = "gait_phase_left"
