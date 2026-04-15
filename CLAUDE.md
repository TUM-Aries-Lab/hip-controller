# hip-controller

Hip flexion exosuit controller Python package for the ARIES Lab (IBRS, TUM). Translates real-time IMU sensor data (hip angle + velocity) into motor position commands for a tendon-driven soft exosuit that assists hip flexion during walking.

**PyPI:** hip-controller | **Current version:** 0.0.5 | **Python:** 3.11+

## Project Structure

```
src/hip_controller/
├── __init__.py
├── __main__.py              # Standalone testing entry point
├── definitions.py           # ALL constants, configs, and dataclasses
├── control/
│   ├── app.py               # WalkOnController (single leg), ExoController (bilateral wrapper)
│   ├── high_level_controller/
│   │   ├── high_level.py          # HighLevelController — gait phase estimation via atan2
│   │   ├── motion_state_machine.py # MotionStateMachine + ExtremaTrigger — extrema detection FSM
│   │   ├── steady_state_tracker.py # SteadyStateTracker — centering, normalization, rescaling
│   │   └── stride_event_detector.py # StrideEventDetector — stride detection via velocity crossing
│   ├── mid_level_controller/
│   │   ├── mid_level.py           # MidLevelController + MotionMapping — motor command generation
│   │   ├── amplitude_modulation.py # AmplitudeModulation + mode strategies — sigmoid scaling
│   │   └── kalman.py              # KalmanFilter (present but NOT in active signal path)
│   └── low_level_controller/
│       ├── low_level.py           # Stop condition helpers only (incomplete)
│       └── pid_controller.py      # PID controller (NOT YET IMPLEMENTED — file is empty)
├── plotter/                 # Visualization: phase portrait, time plots, CSV playback, simulator
└── utils/                   # Math helpers (zero-crossing, sigmoid, atan2), state-space model, logging
```

Uses src-layout. Package metadata and dependencies in `pyproject.toml`.
Tests in `tests/` at root level. Recorded data in `data/`.

## Signal Flow

One call to `WalkOnController.step(signal, timestamp)` triggers:

```
SensorSignal (angle_rad + velocity_rad_per_sec + timestamp)
    │
    ├─► HighLevelController.update_and_compute()
    │       ├─► MotionStateMachine — validates extrema order (Angle min → Vel max → Angle max → Vel min → ...)
    │       ├─► SteadyStateTracker — records extrema, computes center + scale factor once per stride
    │       ├─► StrideEventDetector — velocity crossing at -0.1 offset with refractory period
    │       └─► atan2(vel_steady_state, ang_steady_state) → gait_phase (rad)
    │
    ├─► AmplitudeModulation.compute_amplitude()
    │       ├─► portrait radius = sqrt(angle² + velocity²)
    │       ├─► sigmoid scaling: r^p / (r^p + 1)
    │       └─► multiply by gain × reverse → amplitude
    │
    └─► MidLevelController.compute_motor_command()
            ├─► lag compensation (phase + phi_lag)
            ├─► -sin(phase) → cyclic motion
            ├─► CubicSpline motion mapping (asymmetric lookup table)
            ├─► multiply by amplitude
            └─► saturation at ±600π/180 → motor_command (rad)
```

Returns a single float: the motor position reference in rad.

## Key Design Patterns

- **Strategy pattern** for locomotion modes: `ModeStrategy` ABC with `LevelGroundMode`, `AscendStairsMode`, `DescendStairsMode` — each returns different gain/sigmoid_power/scale parameters
- **Composition over inheritance**: `WalkOnController` contains `HighLevelController`, `AmplitudeModulation`, `MidLevelController` as attributes
- **Factory pattern** used in sibling packages (IMU, motor) for hardware abstraction
- **Frozen dataclasses** for immutable config: `ConfigTension`, `MotorDefaults`, `LookUpTable`, `PositionLimitation`
- All tunable parameters live in `definitions.py` — never hardcode values in algorithm code

## Important Constants (definitions.py)

- `AMPLITUDE_GAIN = -6.5` — motor position desired amplitude (rad)
- `SIGMOID_POWER = 50` — sigmoid scaling exponent
- `LAG_COMPENSATION = 0` — phase offset for lag correction (rad)
- `STRIDE_EVENT_HIT_CROSSING_OFFSET = -0.1` — velocity threshold for stride detection
- `STRIDE_EVENT_COUNTER_TIME = 0.3099` — refractory period (s)
- `PositionLimitation.upper/lower = ±600π/180` — motor command saturation (rad)

## Known Incomplete Components

1. **`pid_controller.py` is empty** — PID feedback loop not yet implemented in Python
2. **No signal filtering in this package** — assumes pre-filtered angle/velocity from IMU package
3. **Locomotion mode classification not wired** — `set_mode()` is never called, defaults to LevelGroundMode
4. **Low-level controller incomplete** — only stop condition helper functions
5. **Threading, not multiprocessing** — current threads are sequential, not truly parallel

## How to Run

```bash
# Install from source
pip install -e .

# Run standalone testing (reads from CSV or live IMU, configurable in definitions.py)
python -m hip_controller

# Run tests
pytest

# Run linter
ruff check .
```

## This Package in Context

`hip-controller` is one module in a larger exosuit system. It is imported by `exosuit-python` (the top-level package that orchestrates IMU reading, motor control, and the controller). Sibling packages:
- `imu-python` — BNO055 IMU reading, drift removal, SOGI+FLL filtering
- `motor-python` — CubeMars AK60-6 motor communication via CAN/serial
- `exosuit-python` — Top-level integration, threading, CSV recording

## Conventions

See `.claude/skills/code-review.md` for the full coding conventions checklist. Key highlights:
- Physical quantities must include units in variable names: `angle_rad`, `velocity_rad_per_sec`
- Type hints on all function signatures, modern syntax (`float | None`, not `Optional[float]`)
- Sphinx/reST docstrings on all public classes and functions
- Use `loguru` for logging, not Python's built-in `logging`
- Use `pathlib.Path`, not `os.path`
- Use `@dataclass(frozen=True)` for configuration objects
- No bare `except:` — always catch specific exceptions

## Restrictions

- **Never modify motor safety limits** (`PositionLimitation`) without explicit discussion
- **Never hardcode hardware-specific paths** (serial ports, I2C addresses)
- **Never push directly to main** — always use pull requests
- **Never install new dependencies** without approval
- Always run `ruff check .` and `pytest` before considering a task complete
