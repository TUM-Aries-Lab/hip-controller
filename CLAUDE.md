# hip-controller

Hip flexion exosuit controller Python package for the ARIES Lab (IBRS, TUM). Translates real-time IMU sensor data (hip angle + velocity) into motor commands for a tendon-driven soft exosuit that assists hip flexion during walking.

**PyPI:** hip-controller | **Current version:** 0.2.0 | **Python:** 3.10+

## Project Structure

```
src/hip_controller/
├── __init__.py
├── __main__.py                      # Standalone runner: CSV playback -> controller -> output CSV
├── definitions.py                   # ALL constants, configs, enums and dataclasses
├── control/
│   ├── app.py                       # WalkOnController — one instance per limb
│   ├── gait_phase_control/
│   │   ├── gait_controller.py            # GaitController — gait phase via atan2
│   │   ├── motion_state_machine.py       # MotionStateMachine + ExtremaTrigger
│   │   ├── steady_state_tracker.py       # SteadyStateTracker — centering, normalization
│   │   └── stride_event_detector.py      # StrideEventDetector — velocity crossing
│   ├── signal_processing/
│   │   ├── sensor_preprocessor.py        # SensorPreprocessor — owns the stages below
│   │   ├── baseline_removal.py           # BaselineRemoval — trigger-driven angle offset
│   │   ├── drift_removal.py              # LowPassDriftRemoval, NotchDriftRemoval
│   │   ├── filtering.py                  # SogiFllFiltering, LowPassFiltering, KalmanFiltering
│   │   └── velocity_estimation.py        # DiscreteDerivative, LowPass, Gyroscope
│   └── motor_reference_control/
│       ├── motor_reference_controller.py # MotionReferenceController + MotionMapping
│       ├── amplitude_modulation.py       # AmplitudeModulation + per-mode ModeStrategy
│       └── pid_controller.py             # PIDController (implemented, not in the signal path)
├── filters/                         # Reusable filter primitives: SOGI-FLL, notch, 2nd-order
│                                    # LPF (4 solver strategies), discrete derivative, Kalman
├── plotter/                         # Live phase portrait, CSV player, CSV inspector (Qt)
└── utils/                           # Math helpers, state-space model, logging, CSV utils
```

Uses src-layout. Package metadata and dependencies in `pyproject.toml`. Tests in `tests/`, recorded data in `data/`.

## Signal Flow

One call to `WalkOnController.step(curr_signal)` triggers:

```
SensorSignal (timestamp + angle_rad + velocity_rad_per_sec)
    │
    ├─► stand-still detection (EMA of |raw velocity|, hysteresis)
    │       └─► freezes the SOGI/FLL frequency tracking while paused
    │
    ├─► SensorPreprocessor.filter()        [skipped when config.filtered=True]
    │       ├─► BaselineRemoval    — subtract the triggered angle offset
    │       ├─► drift removal      — LowPass (default) or Notch
    │       ├─► filtering          — SOGI-FLL (default), LowPass or Kalman
    │       └─► velocity estimation— SOGI quadrature (default), LPF, derivative or gyro
    │                                 + optional DC notch on the velocity
    │
    ├─► GaitController.update_and_compute()
    │       ├─► MotionStateMachine   — validates extrema order
    │       ├─► SteadyStateTracker   — center + scale factor once per stride
    │       ├─► StrideEventDetector  — velocity crossing with refractory period
    │       └─► atan2(vel, ang) → gait_phase (rad)
    │
    ├─► AmplitudeModulation.compute_amplitude()
    │       └─► portrait radius → sigmoid → × gain × reverse → amplitude
    │
    └─► MotionReferenceController.compute_motor_command()
            └─► lag compensation → -sin(phase) → CubicSpline lookup →
                × amplitude → LPF → saturation at ±600π/180 → command (rad)
```

Returns a single float.

## Key Design Patterns

- **Strategy pattern** throughout: `DriftRemovalStrategy`, `FilteringStrategy`, `VelocityEstimationStrategy`, `ModeStrategy`. Which one is built is selected on `BasicConfig`; the parameters they are built from live in `PreprocessorConfig`.
- **Composition**: `WalkOnController` holds `SensorPreprocessor`, `GaitController`, `AmplitudeModulation`, `MotionReferenceController`. One controller per limb — there is no bilateral wrapper class; integrations construct two.
- **Frozen dataclasses** for configuration objects.
- **Per-mode tables, not branch chains**: `AMPLITUDE_MODES` (app.py), `_sogi_config_by_mode` (sensor_preprocessor.py) and `MotionMapping._splines` are all keyed by `int` class_id, each with a level-ground fallback. Adding a mode is one entry per table.
- All tunable parameters live in `definitions.py` — never hardcode values in algorithm code.

## Assist modes

`AssistMode` (definitions.py) is one axis, because its contexts are mutually exclusive:

| value | meaning | source |
|---|---|---|
| 0 `LEVEL` | level ground | default |
| 1 `ASCEND_STAIRS` | stair ascent | TCN classifier, or the exosuit mode switches |
| 2 `DESCEND_STAIRS` | stair descent | TCN classifier, or the exosuit mode switches |
| 3 `RAMP_2_5` | 2.5% treadmill ramp | operator, for the ramp project |
| 4 `RAMP_5` | 5% treadmill ramp | operator, for the ramp project |

`set_locomotion_mode(class_id: int)` fans one integer out to amplitude, SOGI tuning and lookup table. It takes a plain `int` because the classifier emits integers; every consumer falls back to `LEVEL` for an unrecognised value, so a caller may use only the subset it needs.

## Baseline removal

Removes the subject- and mounting-dependent DC offset on the hip angle, the way the deployed Simulink implementation does. Anchored to an explicit trigger — the main / operation switch — never to process start-up:

- `set_baseline_removal_trigger(active)` on `WalkOnController`; edges detected internally, safe to call every loop iteration.
- Stand-still gated: motion during the window restarts it.
- The offset is 0 and the angle passes through untouched until a window completes. Nothing downstream ever sees a fabricated value.
- `filter()` never mutates the caller's `SensorSignal`, which is what makes CSV replay exact — **the logged raw angle must stay pre-baseline, or playback subtracts the offset twice.**
- Playback (`__main__.py`) drives the trigger from the recorded `main_switch` column; `--no-baseline-removal` suppresses it for recordings that predate the feature, and a file without that column never removes a baseline.

## Important constants (definitions.py)

- `AMPLITUDE_GAIN = -6.5` — motor position amplitude (rad)
- `SIGMOID_POWER = 30`, `SCALE_LEVEL_MODE = 2`, `VELOCITY_WEIGHT_LEVEL_MODE = 2.0`
- `LAG_COMPENSATION = 0` — phase offset for lag correction (rad)
- `BASELINE_REMOVAL_WINDOW_S = 0.2`, `BASELINE_REMOVAL_MAX_VELOCITY_RAD_PER_SEC = 0.2`
- `STRIDE_EVENT_HIT_CROSSING_OFFSET = -0.1`, `STRIDE_EVENT_COUNTER_TIME = 0.3099` (s)
- `PositionLimitation.upper/lower = ±600π/180` — motor command saturation (rad)

Sample-rate dependent values (notch filters, the baseline window) are derived from `BasicConfig.frequency` in `PreprocessorConfig.__init__`, so a controller built at a non-default rate is tuned correctly. Never reintroduce import-time constants for these.

## Known gaps

1. **Two parameter sets are unmeasured placeholders.** `RAMP_2_5` and `RAMP_5` currently return the level-ground values in all three consumers, and `test_the_ramps_are_uphill_inclinations_not_stair_modes` fails deliberately when that changes. Do not invent tuning values.
2. **Kalman filtering is selectable but mistuned.** `FilteringMethod.KALMAN` works and is not the default. With the shipped `MEASUREMENT_NOISE = 0.75` rad² (a measurement std of ~0.87 rad, larger than the hip angle) it lags walking data by ~30 ms and is worse than the raw signal; `test_filtering_reduces_noise_while_walking` is xfail with the numbers.
3. **`WalkOnController.reset()` does not reset the gait controller** — `GaitController` exposes no `reset()`, so extrema and stride state survive. `test_reset_restores_initial_response` is xfail.
4. **PID controller is implemented but not in the signal path.**
5. **No locomotion-mode classifier in this package** — `set_locomotion_mode` is called by the integrations.
6. `tests/utils_test/utils_test.py::test_logger_init` fails on Windows only (loguru holds the log file across `TemporaryDirectory` cleanup). It passes on the Linux CI runners.

## How to Run

```bash
pip install -e .                  # install from source
python -m hip_controller          # CSV playback; -p PATH, -f fast, -n no baseline removal
pytest                            # tests
ruff check .                      # linter — NOTE: pyproject sets fix = true, so this
                                  # rewrites files, including ones you did not touch
```

## This Package in Context

`hip-controller` is one module in a larger exosuit system. Sibling packages:
- `imu-python` — BNO055 IMU reading, drift removal, SOGI+FLL filtering
- `motor-python` — CubeMars AK60-6 motor communication via CAN/serial
- `exosuit-python` — top-level integration: GPIO switches, threading, recording
- `LocomotionMode_IMUbased` (not a package) — `receiver.py` is what actually runs on the rig today, with the TCN locomotion classifier. Its Arduino firmware already does its own baseline zeroing at enable, so the Python baseline removal must **not** also be driven there.

## Conventions

See `.claude/skills/code-review-nathalie.md` for the full checklist. Highlights:
- Physical quantities carry units in the name: `angle_rad`, `velocity_rad_per_sec`
- Type hints on all signatures, modern syntax (`float | None`, not `Optional[float]`)
- Sphinx/reST docstrings on all public classes and functions
- `loguru` for logging, not `logging`; `pathlib.Path`, not `os.path`
- `@dataclass(frozen=True)` for configuration objects
- No bare `except:` — always catch specific exceptions

## Restrictions

- **Never modify motor safety limits** (`PositionLimitation`) without explicit discussion
- **Never hardcode hardware-specific paths** (serial ports, I2C addresses)
- **Never push directly to main** — always use pull requests
- **Never install new dependencies** without approval
- **Never invent tuning values** for the unmeasured parameter sets above — they are placeholders pending measurement on the rig
- Always run `ruff check .` and `pytest` before considering a task complete
