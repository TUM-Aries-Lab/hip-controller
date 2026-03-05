"""Code to help initialize pytest."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from hip_controller.definitions import TESTING_DIR

# Add the src directory to the path so that the quaternion_ekf package can be imported
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(my_path, "../src"))

REL_TOL = 1e-9


# Offers path and column strings for testing.
@dataclass
class HighLevelData:
    """High level data for testing."""

    TESTING_HIGH_LEVEL_DIR: Path = (
        TESTING_DIR
        / "controller_test"
        / "high_level_testing"
        / "high_level_testing_data"
    )
    DATA_ZERO_CROSSING: Path = (
        TESTING_HIGH_LEVEL_DIR / "zero_crossing_left_2026_01_09.csv"
    )

    DATA_VALID_TRIGGER: Path = (
        TESTING_HIGH_LEVEL_DIR / "valid_trigger_left_2026_01_15.csv"
    )

    DATA_EXTREMA_VALUES: Path = TESTING_HIGH_LEVEL_DIR / "extrema_2026_01_26.csv"

    DATA_STRIDE_EVENT_DETECTOR: Path = (
        TESTING_HIGH_LEVEL_DIR / "stride_event_detector_2026_02_26.csv"
    )

    DATA_HIGH_LEVEL: Path = TESTING_HIGH_LEVEL_DIR / "gait_phase_left_2026_03_03.csv"


@dataclass
class MidLevelData:
    """Mid level data for testing."""

    TESTING_MID_LEVEL_DIR: Path = (
        TESTING_DIR / "controller_test" / "mid_level_testing" / "mid_level_testing_data"
    )

    DATA_MOTION_MAPPING: Path = TESTING_MID_LEVEL_DIR / "look_up_table_2026_02_25.csv"

    DATA_AMPLITUDE_MODULATION: Path = (
        TESTING_MID_LEVEL_DIR / "amplitude_modulation_2026_03_03.csv"
    )

    DATA_REFERENCE_MOTION: Path = (
        TESTING_MID_LEVEL_DIR / "reference_motion_2026_03_05.csv"
    )


@dataclass
class KinematicsDataColumnName:
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
    VALUE_ANG_MAX: str = "ang_max_left (rad)"
    VALUE_ANG_MIN: str = "ang_min_left (rad)"

    CENTER_ANG: str = "center_ang"
    CENTER_VEL: str = "center_vel"

    VEL_STEADY_STATE: str = "normalized_centered_velocity"
    ANG_STEADY_STATE: str = "normalized_centered_angle"
    RESCALE_FACTOR: str = "gamma"

    # Stride event detector
    NOT_INITIALIZED: str = "before_initialization"
    ENABLE_TRIGGER: str = "enable_trigger"
    ENABLE_DETECTOR: str = "enable_detector"
    VALID_STRIDE: str = "valid_stride"
    STRIDE_EVENT: str = "stride_event"

    # Amplitude Modulation
    RADIUS: str = "portrait_radius"
    SCALED_RADIUS: str = "scaled_portrait_radius"
    SIGMOID_RADIUS: str = "after_sigmoid"
    AMPLITUDE: str = "amplitude"

    # Reference Motion
    GAIT_PHASE: str = "gait_phase_left"
    SIN_WAVE: str = "sinusoidal_behavior"
    REFERENCE_MOTION: str = "reference_motion"

    # Cubic Spline Interpolation
    MAPPING_KEY: str = "motion_mapping_key"
    MAPPING_VALUE: str = "motion_mapping_value"
