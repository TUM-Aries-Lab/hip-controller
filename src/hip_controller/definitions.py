"""Common definitions for this module."""

import sys
from dataclasses import asdict, dataclass, field
from enum import IntEnum, auto

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        """String enum backport for Python <3.11."""


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


class AssistMode(IntEnum):
    """Which tuned parameter set the controller runs.

    One axis, because the contexts are mutually exclusive: nobody descends
    stairs while on a 5% treadmill ramp. Values 0-2 are locomotion modes,
    normally produced by the TCN classifier (and selected by hand on the
    exosuit's mode switches); 3-4 are treadmill ramp inclinations, for the
    ramp project.

    The value is passed to :meth:`WalkOnController.set_locomotion_mode` as a
    plain ``int`` -- the classifier emits integers, so this enum documents and
    names them without constraining callers. Every consumer falls back to
    :attr:`LEVEL` for a value it does not recognise, so a project may use only
    the subset it cares about and older code stays safe against newer ids.
    """

    LEVEL = 0
    ASCEND_STAIRS = 1
    DESCEND_STAIRS = 2
    # Uphill treadmill ramps, named for their inclination in percent.
    RAMP_2_5 = 3
    RAMP_5 = 4


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

    FORWARD_EULER = "forward_euler"
    BACKWARD_EULER = "backward_euler"
    TRAPEZOIDAL = "trapezoidal"
    RUNGE_KUTTA = "rk4"


@dataclass
class LowPassFilterConfig:
    """Settings for the second-order low-pass filter containing cut_off_frequency, damping_ratio, initial_condition, solver_type."""

    cut_off_frequency_rad_per_sec: float = 60.0  # in rad/s
    damping_ratio: float = 1.0  # 1.0 = critically damped
    initial_condition: float = 0.0
    solver_type: SolverType = (
        SolverType.RUNGE_KUTTA
    )  # SolverType enum of numerical integration strategy


# Pre processing


@dataclass(frozen=True)
class NotchConfig:
    """Configurations for the notch function."""

    sample_rate_hz: float
    center_freq_hz: float = 0.0
    bandwidth_3db_hz: float = 0.1


# Baseline removal (hip angle offset).
#
# The IMU angle carries a subject- and mounting-dependent DC offset. It is
# removed by averaging the angle over a short window and subtracting the mean
# from every later sample, exactly as the deployed Simulink implementation does
# (``getHipKinematics_IMU.m``: ``hip_angle = hip_angle_cont - angle_offset``).
#
# The window is anchored to an explicit trigger -- on the exosuit, the main
# switch (motor enable) -- not to process start-up, so the offset is captured
# at a moment the operator controls rather than at whatever the leg happened to
# be doing when the controller booted.
BASELINE_REMOVAL_WINDOW_S: float = 0.2
# A window is only accepted while the limb is still: any sample whose raw
# velocity magnitude exceeds this restarts the window. Matches the stand-still
# notion already used by the pause detector in ``control/app.py``.
BASELINE_REMOVAL_MAX_VELOCITY_RAD_PER_SEC: float = 0.2


@dataclass(frozen=True)
class BaselineRemovalConfig:
    """Settings for baseline removal (hip angle offset).

    :param int window_sample_count: Number of still samples averaged into one
        offset. Derived from :data:`BASELINE_REMOVAL_WINDOW_S` and the
        controller loop frequency, so the averaged *duration* is rate-independent.
    :param float max_velocity_rad_per_sec: Raw velocity magnitude above which the
        limb counts as moving and the collection window restarts [rad/s].

    There is deliberately no "take a baseline at start-up" option. The main
    switch is recorded alongside the sensor data, so the offline paths replay the
    real trigger rather than inventing a moment the operator never chose. A
    caller that genuinely wants an offset from the first samples asks for it
    explicitly via :meth:`BaselineRemoval.set_trigger`.
    """

    window_sample_count: int
    max_velocity_rad_per_sec: float = BASELINE_REMOVAL_MAX_VELOCITY_RAD_PER_SEC


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


class DriftRemovalMethod(StrEnum):
    """Drift removal strategy options."""

    LOW_PASS = auto()
    NOTCH = auto()


class FilteringMethod(StrEnum):
    """Filtering strategy options for the angle stage."""

    SOGI = auto()
    LOW_PASS = auto()


class VelocityInputAngle(StrEnum):
    """Which angle is fed to the velocity-estimation stage.

    RAW             -- ``raw_signal.angle_rad`` straight from the sensor.
    DRIFT_REMOVED   -- output of the drift-removal stage (LPF or notch).
    FILTERED        -- output of the SOGI-FLL stage (current default).
    """

    RAW = auto()
    DRIFT_REMOVED = auto()
    FILTERED = auto()


class VelocityEstimationMethod(StrEnum):
    """Velocity estimation strategy options."""

    SOGI = auto()
    DISCRETE_DERIVATIVE = auto()
    LOW_PASS = auto()
    GYROSCOPE = auto()


class PreprocessorConfig:
    """Configurations for the sensor preprocessor.

    Built per :class:`BasicConfig` so every sample-rate-dependent filter is
    derived from the controller's actual loop frequency. These used to be class
    attributes evaluated at import time, which meant the notch filters always
    carried the default 100 Hz even when the controller ran at another rate --
    silently mistuning them.

    Strategy *selection* lives on :class:`BasicConfig`; this class only holds
    the parameter sets each strategy is constructed from.
    """

    def __init__(self, sample_rate_hz: int) -> None:
        """Build the per-stage filter configurations for one sample rate.

        :param int sample_rate_hz: Controller loop frequency [Hz].
        :return: None
        """
        self.sample_rate_hz: int = sample_rate_hz

        # Selects which angle is fed into the velocity-estimation stage.
        # See VelocityInputAngle for the options. Default keeps the historical
        # behavior (use the SOGI-FLL filtered angle). Ignored when
        # velocity_estimation_method == SOGI (the quadrature is taken directly
        # from the angle-stage SOGI-FLL filter).
        self.velocity_input_angle: VelocityInputAngle = VelocityInputAngle.FILTERED

        # Configurations for the filters
        self.drift_removal_second_order_lpf_config: LowPassFilterConfig = (
            LowPassFilterConfig(
                cut_off_frequency_rad_per_sec=1.25,
                damping_ratio=1.0,
                initial_condition=0.0,
            )
        )
        self.drift_removal_notch_config: NotchConfig = NotchConfig(
            sample_rate_hz=sample_rate_hz, center_freq_hz=0.0, bandwidth_3db_hz=0.1
        )
        # SOGI-FLL config used at construction time -- the SogiFllFilter is
        # initialized with this config. The active config can then be swapped
        # at runtime via ``SensorPreprocessor.set_locomotion_mode(class_id)``,
        # which selects from the per-mode configs below (level/ascend/descend).
        # State (in-phase, quadrature, omega_est, frequency_estimate,
        # confidence_state) is preserved across swaps; only the parameter
        # values change. The FLL re-adapts to the new mode's cadence over
        # 1-2 strides.
        self.filtering_sogifll_config: SogiFllConfig = SogiFllConfig()

        # Per-locomotion-mode SOGI configs. Selected by class_id:
        #   0 -> LEVEL   (default level-walking tuning -- matches the global
        #                 ``filtering_sogifll_config`` so cold-start = level)
        #   1 -> ASCEND  (slower cadence bounds; k_sogi bumped slightly because
        #                 stair-ascend has sharper angle transitions; gentler
        #                 fll adaptation because stair gait has more harmonics
        #                 that perturb the FLL gradient)
        #   2 -> DESCEND (same slower cadence; k_sogi at level value)
        # Adjust empirically. Starting values are intentionally conservative.
        self.filtering_sogifll_config_level: SogiFllConfig = SogiFllConfig(
            lower_cadence_bound=0.3,
            upper_cadence_bound=1.8,
            sogi_adaptation_gain=1.0,
            fll_adaptation_gain=5.0,
            frequency_estimate_smoother_bandwidth=0.8,
            lock_state_smoother_bandwidth=1.50,
            initial_frequency_guess=0.7,
        )
        self.filtering_sogifll_config_ascend: SogiFllConfig = SogiFllConfig(
            lower_cadence_bound=0.25,
            upper_cadence_bound=1.2,
            sogi_adaptation_gain=1.2,
            fll_adaptation_gain=4.5,
            frequency_estimate_smoother_bandwidth=0.6,
            lock_state_smoother_bandwidth=1.50,
            initial_frequency_guess=0.55,
        )
        self.filtering_sogifll_config_descend: SogiFllConfig = SogiFllConfig(
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
        self.filtering_sogifll_config_demo: SogiFllConfig = SogiFllConfig(
            lower_cadence_bound=0.3,
            upper_cadence_bound=3.5,
            sogi_adaptation_gain=1.0,
            fll_adaptation_gain=1.0,
            frequency_estimate_smoother_bandwidth=0.3,
            lock_state_smoother_bandwidth=0.5,
            initial_frequency_guess=1.4,
        )

        # Treadmill ramps. No ramp-specific SOGI tuning has been measured yet,
        # so both reuse the level parameters -- ramp gait is continuous like
        # level walking, unlike the stair modes. Named fields rather than a
        # fallback so the tuned values have an obvious home once they exist.
        self.filtering_sogifll_config_ramp_2_5: SogiFllConfig = (
            self.filtering_sogifll_config_level
        )
        self.filtering_sogifll_config_ramp_5: SogiFllConfig = (
            self.filtering_sogifll_config_level
        )

        self.filtering_second_order_lpf_config: LowPassFilterConfig = (
            LowPassFilterConfig(
                cut_off_frequency_rad_per_sec=90.0,
                damping_ratio=1.0,
                initial_condition=0.0,
            )
        )

        # Toggle the DC-notch drift removal applied to the estimated velocity,
        # independent of which velocity_estimation_method is selected.
        # True  -> notch is applied (default).
        # False -> velocity is passed through unfiltered.
        self.apply_velocity_drift_removal: bool = True

        # Notch-at-DC applied to the estimated velocity (SOGI quadrature, discrete
        # derivative, LPF derivative, or gyroscope) when apply_velocity_drift_removal
        # is True. Surgical DC-bias removal on the velocity signal.
        self.velocity_drift_removal_notch_config: NotchConfig = NotchConfig(
            sample_rate_hz=sample_rate_hz, center_freq_hz=0.0, bandwidth_3db_hz=0.1
        )

        # Baseline removal. The averaged window is specified in seconds and
        # converted here, so the same configuration means the same duration at
        # any controller rate.
        self.baseline_removal_config: BaselineRemovalConfig = BaselineRemovalConfig(
            window_sample_count=max(
                1, round(BASELINE_REMOVAL_WINDOW_S * sample_rate_hz)
            )
        )


@dataclass(frozen=True)
class BasicConfig:
    """Basic configurations for the hip controller.

    Carries both the run-level switches (plotting, wiring, data source) and the
    strategy selection for the preprocessing pipeline. ``preprocessor_config``
    is derived from ``frequency`` in ``__post_init__``, so a controller built at
    a non-default rate gets correctly tuned sample-rate-dependent filters.

    ``preprocessor_config`` is an instance field rather than a class attribute,
    so reach it through an instance: ``BasicConfig().preprocessor_config``.
    """

    # general loop frequency
    frequency: int = 100

    # if the incoming signal is already preprocessed, skip the preprocessor
    filtered: bool = False

    # if the graph is displayed or not. Off by default: the controller runs
    # headless on the exosuit, and the standalone runner opts in explicitly.
    left_limb_plot: bool = False
    right_limb_plot: bool = False

    # if the wiring settings are reversed or not
    left_limb_reverse: bool = False
    right_limb_reverse: bool = True

    # either read data from imu or read data from csv file using csv player
    read_from_imu: bool = False

    # the path where data is read from
    read_data_from_path: Path = (
        DATA_DIR / "sensor_data" / "data_input_filtered_2026_01_09.csv"
    )

    # Strategy selection for the three preprocessing stages. The parameter sets
    # each strategy is built from live in PreprocessorConfig.
    drift_removal_method: DriftRemovalMethod = DriftRemovalMethod.LOW_PASS
    filtering_method: FilteringMethod = FilteringMethod.SOGI
    velocity_estimation_method: VelocityEstimationMethod = VelocityEstimationMethod.SOGI

    preprocessor_config: PreprocessorConfig = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Derive the preprocessor configuration from the configured frequency.

        :return: None
        """
        object.__setattr__(
            self,
            "preprocessor_config",
            PreprocessorConfig(sample_rate_hz=self.frequency),
        )


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
AMPLITUDE_GAIN = -6.5  # Motor position desired amplitude (rad)
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
    # Doubles as the baseline-removal trigger: it is the switch the operator
    # flips when the subject is standing ready, which is exactly the moment the
    # angle offset should be taken.
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
