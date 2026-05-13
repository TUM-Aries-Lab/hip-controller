"""Code to help initialize pytest."""

import os
import sys
from hip_controller.definitions import StrEnum
from pathlib import Path

from hip_controller.definitions import TESTING_DIR

# Add the src directory to the path so that the quaternion_ekf package can be imported
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(my_path, "../src"))


REL_TOL = 1e-9


# File folder path for testing
TESTING_CONTROLLER_DIR = TESTING_DIR / "controller_test" / "testing_data"


# General testing files
DATA_REFERENCE_MOTION_RIGHT: Path = (
    TESTING_CONTROLLER_DIR / "reference_motion_2026_03_06.csv"
)
# Preprocessing testing files
DATA_PRE_PROCESSING: Path = TESTING_CONTROLLER_DIR / "filtering_2026_03_19.csv"
# High level controller testing files
DATA_ZERO_CROSSING: Path = TESTING_CONTROLLER_DIR / "zero_crossing_left_2026_01_09.csv"

DATA_VALID_TRIGGER: Path = TESTING_CONTROLLER_DIR / "valid_trigger_left_2026_01_15.csv"

DATA_EXTREMA_VALUES: Path = TESTING_CONTROLLER_DIR / "extrema_2026_01_26.csv"

DATA_STRIDE_EVENT_DETECTOR: Path = (
    TESTING_CONTROLLER_DIR / "stride_event_detector_2026_02_26.csv"
)
DATA_HIGH_LEVEL: Path = TESTING_CONTROLLER_DIR / "gait_phase_left_2026_03_03.csv"

# Mid level controller testing files
DATA_MOTION_MAPPING: Path = TESTING_CONTROLLER_DIR / "look_up_table_2026_02_25.csv"

DATA_AMPLITUDE_MODULATION: Path = (
    TESTING_CONTROLLER_DIR / "amplitude_modulation_2026_03_03.csv"
)

DATA_REFERENCE_MOTION_LEFT: Path = (
    TESTING_CONTROLLER_DIR / "reference_motion_2026_03_05.csv"
)

# Low level controller testing files
DATA_SECOND_ORDER_LOW_PASS_FILTER: Path = (
    TESTING_CONTROLLER_DIR / "second_order_lpf_2026_03_06.csv"
)


class KinematicsDataColumnName(StrEnum):
    """Names of columns for csv files for high-level controller testing."""

    TIMESTAMP = "time (s)"
    ANGLE_LEFT = "angle_left (rad)"
    VELOCITY_LEFT = "vel_left (rad/s)"

    TRIGG_VEL_MAX = "vel_max_trigg_left"
    TRIGG_ANG_MAX = "ang_max_trigg_left"
    TRIGG_VEL_MIN = "vel_min_trigg_left"
    TRIGG_ANG_MIN = "ang_min_trigg_left"

    VALID_TRIGG_VEL_MAX = "valid_vel_max_left"
    VALID_TRIGG_ANG_MAX = "valid_ang_max_left"
    VALID_TRIGG_VEL_MIN = "valid_vel_min_left"
    VALID_TRIGG_ANG_MIN = "valid_ang_min_left"

    VALUE_VEL_MAX = "vel_max_left (rad/s)"
    VALUE_VEL_MIN = "vel_min_left (rad/s)"
    VALUE_ANG_MAX = "ang_max_left (rad)"
    VALUE_ANG_MIN = "ang_min_left (rad)"

    CENTER_ANG = "center_ang"
    CENTER_VEL = "center_vel"

    VEL_STEADY_STATE = "normalized_centered_velocity"
    ANG_STEADY_STATE = "normalized_centered_angle"
    RESCALE_FACTOR = "gamma"

    # Stride event detector
    NOT_INITIALIZED = "before_initialization"
    ENABLE_TRIGGER = "enable_trigger"
    ENABLE_DETECTOR = "enable_detector"
    VALID_STRIDE = "valid_stride"
    STRIDE_EVENT = "stride_event"

    # Amplitude Modulation
    RADIUS = "portrait_radius"
    SCALED_RADIUS = "scaled_portrait_radius"
    SIGMOID_RADIUS = "after_sigmoid"
    AMPLITUDE_LEFT = "amplitude"

    # Reference Motion
    GAIT_PHASE_LEFT = "gait_phase_left"
    SIN_WAVE_LEFT = "sinusoidal_behavior"
    REF_MOT_LEFT = "reference_motion"

    # Cubic Spline Interpolation
    MAPPING_KEY = "motion_mapping_key"
    MAPPING_VALUE = "motion_mapping_value"

    # Right lower limb tests
    ANG_RIGHT = "angle_right (rad)"
    VEL_RIGHT = "velocity_right (rad/s)"

    GAIT_PHASE_RIGHT = "gait_phase_right"
    SIN_WAVE_RIGHT = "sinusoidal_behavior_right"
    AMPLITUDE_RIGHT = "amplitude_right"

    REF_MOT_RIGHT = "motor_reference_right (rad)"

    # Signal pre-processing with raw left angle
    RAW_ANG_LEFT = "angle_left_raw (rad)"

    NO_DRIFT_ANG_LPF = "angle_no_drift_lpf"
    NO_DRIFT_ANG_NOTCH = "angle_no_drift_notch"

    FILTERED_ANG_SOGIFLL = "angle_surrogate_sogifll"
    FILTERED_VEL_SOGIFLL = "vel_quadrature_sogifll"

    FILTERED_VEL_DISCRETE = "vel_discrete_derivative"

    FILTERED_ANG_LPF = "angle_filtered_lpf"
    FILTERED_VEL_LPF = "vel_derivative_lpf"
