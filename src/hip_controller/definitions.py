"""Common definitions for this module."""

import sys
from dataclasses import asdict, dataclass, field

from numpy.typing import NDArray

from hip_controller.utils.state_space import StateSpaceLinear

if sys.version_info >= (3, 11):
    from enum import StrEnum, auto
else:
    from enum import Enum

    class StrEnum(str, Enum):
        """String enum backport for Python <3.11."""


from math import pi
from pathlib import Path

import numpy as np

from hip_controller.utils.state_space import StateSpaceLinear

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


class DriftRemovalMethod(StrEnum):
    """Drift removal strategy options."""

    LOW_PASS = auto()
    NOTCH = auto()


class FilteringMethod(StrEnum):
    """Filtering strategy options."""

    SOGI = auto()
    KALMAN = auto()
    LOW_PASS = auto()


class VelocityEstimationMethod(StrEnum):
    """Velocity estimation strategy options."""

    SOGI = auto()
    DISCRETE_DERIVATIVE = auto()
    LOW_PASS = auto()
    GYROSCOPE = auto()


class SolverType(StrEnum):
    """Selects the numerical integration strategy for the LPF.

    FORWARD_EULER   -- Discrete, output = current state before update.
                       Maps to Simulink discrete integrator (Forward Euler method).
    BACKWARD_EULER  -- Discrete, output = state + dt*u (input feedthrough).
                       Maps to Simulink discrete integrator (Backward Euler method).
    TRAPEZOIDAL     -- Discrete, output = state + dt/2*u (average of current/next).
                       Maps to Simulink discrete integrator (Trapezoidal method).
    RK4             -- Continuous, four-stage Runge-Kutta.
                       Maps to Simulink continuous integrator + ode4 solver.
    """

    FORWARD_EULER = auto()
    BACKWARD_EULER = auto()
    TRAPEZOIDAL = auto()
    RUNGE_KUTTA = auto()


@dataclass(frozen=True)
class BasicConfig:
    """Basic configurations for the hip controller."""

    # general frequency
    frequency: int = 100

    # if data is pre filtered - skip the pre processing
    filtered: bool = False

    # if the graph is displayed or not
    left_limb_plot: bool = False
    right_limb_plot: bool = False

    # if the wiring settings are reversed or not
    left_limb_reverse: bool = True
    right_limb_reverse: bool = False

    # either read data from imu or read data from csv file using csv player
    read_from_imu: bool = False

    # the path where data is read from
    read_data_from_path: Path = (
        DATA_DIR / "sensor_data" / "data_input_filtered_2026_01_09.csv"
    )

    # select which DriftRemovalMethod, VelocityEstimationMethod
    drift_removal_method: DriftRemovalMethod = DriftRemovalMethod.LOW_PASS

    filtering_method: FilteringMethod = FilteringMethod.SOGI
    velocity_estimation_method: VelocityEstimationMethod = (
        VelocityEstimationMethod.DISCRETE_DERIVATIVE
    )
    # cut-off frequency for the 2ndOrderLP filter
    cut_off_freq_low_pass_rad_per_sec: float = 80.0


@dataclass
class LowPassFilterConfig:
    """Settings for the second-order low-pass filter containing cut_off_frequency, damping_ratio, initial_condition, solver_type."""

    cut_off_frequency_rad_per_sec: float = (
        BasicConfig.cut_off_freq_low_pass_rad_per_sec
    )  # in rad/s
    damping_ratio: float = 1.0  # 1.0 = critically damped
    initial_condition: float = 0.0
    solver_type: SolverType = (
        SolverType.RUNGE_KUTTA
    )  # SolverType enum of numerical integration strategy

@dataclass(frozen=True)
class KalmanFilterConfig:
    """Settings for the Kalman filter."""

    process_noise: NDArray = field(default_factory=lambda: 2e-2 * np.eye(2))
    measurement_noise: NDArray = field(default_factory=lambda: 0.75 * np.eye(1))
    state_space: StateSpaceLinear = field(
        default_factory=lambda: StateSpaceLinear(
            A=np.array([[1.0, 0.01], [0.0, 1.0]]), C=np.array([[1.0, 0.0]])
        )
    )
    initial_state: NDArray = field(default_factory=lambda: np.array([0.0, 0.0]))
    initial_covariance: NDArray = field(default_factory=lambda: 10 * np.eye(2))
    
# Pre processing


@dataclass(frozen=True)
class NotchConfig:
    """Configurations for the notch function."""

    center_freq_hz: float = 0.0
    bandwidth_3db_hz: float = 0.1
    sample_rate_hz: float = BasicConfig.frequency


@dataclass(frozen=True)
class SogiFllConfig:
    """SOGI-FLL parameter set.

    :param float lower_cadence_bound: f_min Minimum expected cadence (Hz): lower if very slow walking is possible
    :param float upper_cadence_bound: f_max Maximum expected cadence (Hz): increase if including fast walking/running
    :param float sogi_adaptation_gain: k_sogi Tune k_sogi only if the portrait is ringy or too sluggish. Increase to 1.2-1.4 if theta/theta_quad look underdamped / not tracking
    :param float fll_adaptation_gain: k_fll Frequency adaptation speed: increase to track speed changes faster, decrease if noisy/jittery
    :param float lower_energy_threshold: E_lo Energy threshold to block adaptation during noise/standing: increase if false locking occurs
    :param float upper_energy_threshold: E_hi  Energy threshold for full adaptation: decrease if it never locks during slow gait
    :param float frequency_estimate_smoother_bandwidth: fc_f_smooth
    :param float lock_state_smoother_bandwidth: fc_lock
    :param float initial_frequency_guess: f_init
    :param float decay_not_walking: decay_notwalking
    :param float numerical_safety_floor: epsSmall
    """

    # cadence bounds (walking/running range)
    lower_cadence_bound: float = 0.3  # 0.2 -> extremely slow walking
    upper_cadence_bound: float = 1.8  # 4.0 -> very fast running

    # Tune only if the portrait is ringy or too sluggish:s
    #   - increase to 1.2-1.4 if theta/theta_quad look underdamped / not tracking well
    #   - decrease to 0.8-0.9 if very noisy and jitter is observed
    sogi_adaptation_gain: float = 1.0  # 0.7 #1.0
    # Frequency adaptation speed:
    #   - increase to track speed changes faster
    #   - decrease if noisy/jittery (sensor/noise dependent)
    fll_adaptation_gain: float = 5.0  # 15.0#1.0

    # lock thresholds (amplitude/noise dependent)
    lower_energy_threshold: float = 1e-4  # 5e-4 #1e-4
    upper_energy_threshold: float = 1e-1  # 5e-2 #1e-2

    # Tune only if internal frequency becomes jittery or too laggy:
    # - decrease to 0.2 for smoother (more lag)
    # - increase to 0.5 for faster (more jitter)
    frequency_estimate_smoother_bandwidth: float = 0.8  # 2.0  # 0.20 #0.30

    # Tune only if lock flickers or reacts too slowly:
    # - decrease (0.3) to reduce flicker
    # - increase (0.8-1.0) for faster start/stop response
    lock_state_smoother_bandwidth: float = 1.50  # 0.30  #0.50

    # [Hz] initial guess (walking/running general default)
    # Tune only if you want faster lock at startup:
    # - set near typical cadence in your trials (walk ~1-2 Hz, run ~2-3 Hz)
    initial_frequency_guess: float = 0.7  # 1.0

    # % state decay when standing
    # Tune only if oscillator rings too long after stopping:
    # - faster decay: 0.995
    # - slower decay: 0.9995
    decay_not_walking: float = 0.999

    # numerical safety
    # Increase only if you see NaN/Inf in extreme low-motion segments (e.g., 1e-8)
    numerical_safety_floor: float = 1e-9


class VelocityInputAngle(StrEnum):
    """Which angle is fed to the velocity-estimation stage.

    RAW             -- ``raw_signal.angle_rad`` straight from the sensor.
    DRIFT_REMOVED   -- output of the drift-removal stage (LPF or notch).
    FILTERED        -- output of the SOGI-FLL stage (current default).
    """

    RAW = auto()
    DRIFT_REMOVED = auto()
    FILTERED = auto()


class PreprocessorConfig:
    """Configurations for the sensor preprocessor."""

    # Selects which angle is fed into the velocity-estimation stage.
    # See VelocityInputAngle for the options. Default keeps the historical
    # behavior (use the SOGI-FLL filtered angle). Ignored when
    # velocity_estimation_method == SOGI (the quadrature is taken directly
    # from the angle-stage SOGI-FLL filter).
    velocity_input_angle: VelocityInputAngle = VelocityInputAngle.FILTERED

    # Configurations for the filters
    drift_removal_second_order_lpf_config: LowPassFilterConfig = LowPassFilterConfig(
        cut_off_frequency_rad_per_sec=1.25, damping_ratio=1.0, initial_condition=0.0
    )
    drift_removal_notch_config: NotchConfig = NotchConfig(
        center_freq_hz=0.0, bandwidth_3db_hz=0.1, sample_rate_hz=BasicConfig.frequency
    )

    filtering_kalman_config: KalmanFilterConfig = KalmanFilterConfig()

    # SOGI-FLL config used at construction time -- the SogiFllFilter is
    # initialized with this config. The active config can then be swapped
    # at runtime via ``SensorPreprocessor.set_locomotion_mode(class_id)``,
    # which selects from the per-mode configs below (level/ascend/descend).
    # State (in-phase, quadrature, omega_est, frequency_estimate,
    # confidence_state) is preserved across swaps; only the parameter
    # values change. The FLL re-adapts to the new mode's cadence over
    # 1-2 strides.
    filtering_sogifll_config: SogiFllConfig = SogiFllConfig()

    # Per-locomotion-mode SOGI configs. Selected by class_id:
    #   0 -> LEVEL   (default level-walking tuning -- matches the global
    #                 ``filtering_sogifll_config`` so cold-start = level)
    #   1 -> ASCEND  (slower cadence bounds; k_sogi bumped slightly because
    #                 stair-ascend has sharper angle transitions; gentler
    #                 fll adaptation because stair gait has more harmonics
    #                 that perturb the FLL gradient)
    #   2 -> DESCEND (same slower cadence; k_sogi at level value)
    # Adjust empirically. Starting values are intentionally conservative.
    filtering_sogifll_config_level: SogiFllConfig = SogiFllConfig(
        lower_cadence_bound=0.3,
        upper_cadence_bound=1.8,
        sogi_adaptation_gain=1.0,
        fll_adaptation_gain=5.0,
        frequency_estimate_smoother_bandwidth=0.8,
        lock_state_smoother_bandwidth=1.50,
        initial_frequency_guess=0.7,
    )
    filtering_sogifll_config_ascend: SogiFllConfig = SogiFllConfig(
        lower_cadence_bound=0.25,
        upper_cadence_bound=1.2,
        sogi_adaptation_gain=1.2,
        fll_adaptation_gain=4.5,
        frequency_estimate_smoother_bandwidth=0.6,
        lock_state_smoother_bandwidth=1.50,
        initial_frequency_guess=0.55,
    )
    filtering_sogifll_config_descend: SogiFllConfig = SogiFllConfig(
        lower_cadence_bound=0.25,
        upper_cadence_bound=1.2,
        sogi_adaptation_gain=1.0,
        fll_adaptation_gain=4.5,
        frequency_estimate_smoother_bandwidth=0.6,
        lock_state_smoother_bandwidth=1.50,
        initial_frequency_guess=0.55,
    )
    # Demo (classification-free assist). Wider SOGI bandwidth + faster FLL
    # for lower phase lag between IMU angle and the filtered signal the
    # demo LUT consumes. Same cadence bounds as level. Trade-off: more
    # sensor noise reaches the motor -- if the motor feels jittery on the
    # demo, dial the gains back toward the level config.
    filtering_sogifll_config_demo: SogiFllConfig = SogiFllConfig(
        lower_cadence_bound=0.3,
        upper_cadence_bound=3.5,
        sogi_adaptation_gain=1.0,
        fll_adaptation_gain=1.0,
        frequency_estimate_smoother_bandwidth=0.3,
        lock_state_smoother_bandwidth=0.5,
        initial_frequency_guess=1.4,
    )

    filtering_second_order_lpf_config: LowPassFilterConfig = LowPassFilterConfig(
        cut_off_frequency_rad_per_sec=90.0, damping_ratio=1.0, initial_condition=0.0
    )

    velocity_estimation_low_pass_config: LowPassFilterConfig = LowPassFilterConfig(
        cut_off_frequency_rad_per_sec=20.0, damping_ratio=1.0, initial_condition=0.0
    )

    # Toggle the DC-notch drift removal applied to the estimated velocity,
    # independent of which velocity_estimation_method is selected.
    # True  -> notch is applied (default).
    # False -> velocity is passed through unfiltered.
    apply_velocity_drift_removal: bool = True

    # Notch-at-DC applied to the estimated velocity (SOGI quadrature, discrete
    # derivative, LPF derivative, or gyroscope) when apply_velocity_drift_removal
    # is True. Surgical DC-bias removal on the velocity signal.
    velocity_drift_removal_notch_config: NotchConfig = NotchConfig(
        center_freq_hz=0.0, bandwidth_3db_hz=0.1, sample_rate_hz=BasicConfig.frequency
    )


# baseline removal using first N samples
BASELINE_REMOVAL_SAMPLE_NUM = 10

# centering & normalization
VALUE_NEAR_ZERO = 1e-6

# stride event detector
STRIDE_EVENT_COUNTER_TIME = 0.3099  # in s
STRIDE_EVENT_HIT_CROSSING_OFFSET = -0.1

# ---mid level---
LAG_COMPENSATION = 0  # Lag correction

# Amplitude modulation
SCALE_LEVEL_MODE = 2  # Needed if the SOGI-FLL quadrature is taken as velocity output
SIGMOID_POWER = 30  # 50
AMPLITUDE_GAIN = -6  # 6.5  # Motor position desidered amplitude (rad)
# Per-component weight applied to velocity inside the portrait radius:
#   r = sqrt(angle^2 + (VELOCITY_WEIGHT_LEVEL_MODE * velocity)^2).
# 1.0 -> identical to the historical symmetric radius.
# >1  -> velocity contributes more (the sigmoid trips earlier on fast motion).
# <1  -> velocity contributes less (more sensitive to angle).
VELOCITY_WEIGHT_LEVEL_MODE = 2.0  # 1.0

# Kalman filter definitions
PROCESS_NOISE = 2e-2
MEASUREMENT_NOISE = 0.75


# Cubic Spline Interpolation
@dataclass(frozen=True)
class LookUpTable:
    """Stores breakpoint and table data of the motion mapping.

    Held for backward compatibility / external callers. The active
    per-mode tabledata is selected at runtime by ``MotionMapping`` via
    the ``LOOKUP_TABLEDATA_*`` constants below.
    """

    breakpoints = np.array(
        [-1, -0.8, -0.6, -0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.6, 0.8, 1],
        dtype=np.float64,
    )

    tabledata = np.array(
        [-1, -0.8, -0.6, -0.4, -0.2, -0.1, 0, 0, 0.005, 0.01, 0.015, 0.02, 0.025],
        dtype=np.float64,
    )


# Per-locomotion-mode motion-mapping table data. Same breakpoints as
# ``LookUpTable.breakpoints``; only the y-values differ per mode.
#
# The FLEXION half (indices 0..6, breakpoints from -1 to 0) is identical
# across modes -- the flexion-assist shape stays the same; per-mode
# `ModeParameters.gain` scales its strength.
#
# The EXTENSION half (indices 7..12, breakpoints from 0.1 to 1) is
# zeroed for ASC and DSC. Empirical drift tracking (see
# ``add_drift_tracking.py`` in the application repo) showed the motor's
# REST position drifts ~0.20 rad negative (left) per ASC/DSC bout,
# meaning the motor was structurally paying out ~0.16-0.21 rad of cable
# per stride between flexions on stair modes (= lookup value 0.025 *
# gain). That pay-out accumulates as cable slack that produces "late
# assist" right after coming off stairs. With the extension half zeroed
# on ASC/DSC, the motor holds its rest position between strides on stairs
# instead of unwinding -- the per-stride slack contribution is removed
# at the source.
#
# Level Ground keeps the original asymmetric table because (a) the drift
# diagnostic shows MP rest median stays approximately flat during steady
# LG walking, so LG is not the culprit, and (b) the small extension-side
# counter-pull on LG was tuned for natural-feeling swing-through.
LOOKUP_TABLEDATA_LEVEL: np.ndarray = np.array(
    [-1, -0.8, -0.6, -0.4, -0.2, -0.1, 0, 0, 0.005, 0.01, 0.015, 0.02, 0.025],
    dtype=np.float64,
)
LOOKUP_TABLEDATA_ASCEND: np.ndarray = np.array(
    [-1, -0.8, -0.6, -0.4, -0.2, -0.1, 0, 0, 0, 0, 0, 0, 0],
    dtype=np.float64,
)
LOOKUP_TABLEDATA_DESCEND: np.ndarray = np.array(
    [-1, -0.8, -0.6, -0.4, -0.2, -0.1, 0, 0, 0, 0, 0, 0, 0],
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

    :timestamp: current timestamp.
    :angle_rad: hip angle of the lower limb in radians.
    :velocity_rad_per_sec: hip angle velocity of the lower limb in radians per second.
    """

    timestamp: float | None
    angle_rad: float = 0.0
    velocity_rad_per_sec: float = 0.0


@dataclass
class ExosuitData:
    """Container for the measurements from the sensor of both lower limbs.

    :left: Signal state of the left lower limb.
    :right: Signal state of the right lower limb.
    """

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
    main_switch: str = "main_switch"

    fake_frequency_hz: int = BasicConfig.frequency


@dataclass(frozen=True)
class PIDConfig:
    """Configurations for PID controller."""

    proportional_gain: float = 8.0
    integral_gain: float = 0.0
    derivative_gain: float = 0.02
    output_limits: tuple[float, float] | None = None


@dataclass(frozen=True)
class PlotConfig:
    """Plot Configurations for the hip controller."""

    graph_width = 1000
    graph_height = 500

    draw_sample_frequency = 10

    # left time-series graph config
    time_plot_title: str = "Motor Command Time Series"
    time_plot_x_axis_label = "Time"
    time_plot_y_axis_label = "Motor Velocity Command"
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
    phase_plot_title: str = "Phase Portrait"
    phase_plot_size: int = 150
    phase_plot_window_margin = 1.1

    phase_plot_axis_angle = "Centered and Scaled Angle"
    phase_plot_axis_velocity = "Centered Angular Velocity"
    phase_plot_scatter_size = 5

    # Since the spots are fading transparently, RGB values are be given separately
    phase_plot_scatter_color_r = 56
    phase_plot_scatter_color_g = 136
    phase_plot_scatter_color_b = 56
    phase_plot_line_width = 2
    phase_plot_line_color = "#36BB63"


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


DEFAULT_LOG_LEVEL: str = LogLevel.info
DEFAULT_LOG_FILENAME = "log_file"
