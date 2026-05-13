# hip-controller

[![Coverage Status](https://coveralls.io/repos/github/TUM-Aries-Lab/hip-controller/badge.svg?branch=main)](https://coveralls.io/github/TUM-Aries-Lab/hip-controller?branch=main)
![Docker Image CI](https://github.com/TUM-Aries-Lab/hip-controller/actions/workflows/ci.yml/badge.svg)

Simple README.md for a Python project template.

## Install

To install the library from PyPI:

```bash
uv pip install hip-controller==latest
```
OR
```bash
uv add git+https://github.com/TUM-Aries-Lab/hip-controller.git@<specific-tag>  # needs credentials
```

## Development
0. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) from Astral.
1. `git clone git@github.com:TUM-Aries-Lab/hip-controller.git`
2. `make init` to create the virtual environment and install dependencies
3. `make format` to format the code and check for errors
4. `make test` to run the test suite
5. `make clean` to delete the temporary files and directories


## Publishing
It's super easy to publish your own packages on PyPI. To build and publish this package, run:
1. Update the version number in pyproject.toml and imu_module/__init__.py
2. Commit your changes and add a `git tag <new.version.number>`
3. Push the tag `git push --tag`

```bash
uv build
uv publish  # make sure your version in pyproject.toml is updated

or

Update the version number in pyproject.toml and imu_module/__init__.py
Commit your changes and add a git tag v<new.version.number>
Push the tag git push --tag

The package can then be found at: https://pypi.org/project/hip-controller

## Module Usage
```python
"""Basic docstring for my module."""

from loguru import logger

from hip_controller import definitions

def main() -> None:
    """Run a simple demonstration."""
    logger.info("Hello World!")

if __name__ == "__main__":
    main()
```

## Program Usage
```bash
uv run python -m hip_controller
```
To use plotting via PyQt6:

On linux computer, edit the Jetson entry in `.ssh/config` or `$HOME\.ssh\config` on Windows (to be verified)
```
Host jetson
  HostName 10.183.230.183
  User aries-orin-1
  ForwardX11 yes
  ForwardX11Trusted yes
```
Then (in VSCode) run this in the remote terminal:
```
echo $DISPLAY
```
Which should show: (if empty, then forwarding is not working properly)
```
localhost:10.0
```
To run PyQT6 on the Jetson, it also needs to install `libxcb-cursor-dev` with:
```bash
sudo apt-get install -y libxcb-cursor-dev
```

## Introduction

The hip-controller is a Python-based control system designed for lower limb exosuits, specifically targeting hip assistance during gait. It integrates sensor data processing, real-time gait phase detection, and adaptive assistance control to enhance mobility and reduce energy expenditure for users.

Key components include:
- **Signal Processing**: Filters and preprocessors for IMU and other sensor data.
- **Gait Phase Control**: Detection of stride events, motion state machines, and steady-state tracking.
- **Assistance Control**: PID controllers, amplitude modulation, and mid-level control strategies.
- **Plotting and Simulation**: Tools for visualization, live plotting, and offline simulation.

This project is developed as part of research at TUM-Aries-Lab, focusing on exoskeleton technology for rehabilitation and assistance.

## Features

- Real-time sensor data preprocessing with filters like Kalman, notch, and low-pass filters.
- Gait phase estimation using motion state machines and stride event detection.
- Adaptive assistance control with PID and amplitude modulation.
- Visualization tools for phase portraits, comparison plots, and CSV playback.
- Comprehensive test suite for validation.
- Docker support for containerized deployment.

## Requirements

- Python 3.11 to 3.13
- Dependencies managed via `uv` (see `pyproject.toml` for details)
- Optional: MATLAB for simulation files (not required for Python usage)

## Installation

In addition to the PyPI installation mentioned above, for development:

1. Ensure `uv` is installed.
2. Clone the repository.
3. Run `make init` to set up the virtual environment and install dependencies.

For Docker usage:
```bash
docker build -t hip-controller .
docker run hip-controller
```

## How to Import

Import the main module:
```python
from hip_controller import definitions
```

Import specific components:
```python
from hip_controller.control.app import AppController
from hip_controller.control.signal_processing.kalman_filter import KalmanFilter
from hip_controller.plotter.simulator import Simulator
```

## Architecture and Flow

The hip-controller follows a modular architecture:

1. **Data Input**: Sensor data (e.g., IMU angles, velocities) from CSV files or live sources.
2. **Preprocessing**: Apply filters (second-order low pass, notch, sogi-fll) to clean the data.
3. **Gait Phase Detection**: Use state machines and event detectors to identify phases (stance, swing).
4. **Control Calculation**: Compute assistance motor command using PID controllers and modulation.
5. **Output**: Generate control signals for the exosuit actuators.
6. **Visualization**: Plot results in real-time or from recorded data.

The main entry point is `app.py` in the control module, which orchestrates the flow.

## Usage

### Basic Program Usage
Run the main application:
```python
from hip_controller.control.app import WalkOnController
from hip_controller.definitions import SensorSignal
logger.info("Initializing the lower limb controller.")


self.controller_left = WalkOnController(reverse=False, plot=False)
self.controller_right = WalkOnController(reverse=True, plot=False)

while True:
    signal_left = SensorSignal(timestamp=timestamp_left, angle_rad=data_left.quat.to_euler(seq="xyz").z, velocity_rad_per_sec=data_left.device_data.gyro.z)
    signal_right = SensorSignal(timestamp=timestamp_right, angle_rad=data_right.quat.to_euler(seq="xyz").z, velocity_rad_per_sec=data_right.device_data.gyro.z)

    command_left = self.controller_left.step(curr_signal=signal_left)
    command_right = self.controller_right.step(curr_signal=signal_left)
               

    motor_left.set_velocity(command_left)
    motor_right.set_velocity(command_right)

    time.sleep(0.01)
```

### Module Usage Examples

#### Preprocessing Raw Sensor Data
```python
from hip_controller.control.signal_processing.sensor_preprocessor import SensorPreprocessor
from hip_controller.definitions import PreprocessorConfig, SensorSignal

# Initialize preprocessor with default config
preprocessor = SensorPreprocessor(config=PreprocessorConfig())

# Raw sensor signal
raw_signal = SensorSignal(timestamp=0.01, angle_rad=0.5, velocity_rad_per_sec=1.2)

# Get filtered angle and estimated velocity
processed_signal = preprocessor.filter(raw_signal)
```

#### Computing Gait Phase
```python
from hip_controller.control.gait_phase_control.gait_controller import GaitController
from hip_controller.definitions import SensorSignal

# Initialize gait controller
gait_controller = GaitController()

# Sensor signals with angle and velocity
signal1 = SensorSignal(timestamp=0.01, angle_rad=0.1, velocity_rad_per_sec=0.5)
signal2 = SensorSignal(timestamp=0.02, angle_rad=0.15, velocity_rad_per_sec=0.8)

# Update and compute gait phase
gait_phase1 = gait_controller.update_and_compute(signal1)
gait_phase2 = gait_controller.update_and_compute(signal2)
```

#### Computing Motor Command Velocity with Assistance Controller
```python
from hip_controller.control.assistance_control.assistance_controller import MotionReferenceController
from hip_controller.control.assistance_control.amplitude_modulation import LevelGroundMode

# Initialize controllers
motion_controller = MotionReferenceController()
amplitude_mode = LevelGroundMode()

# Compute motor command (position reference)
motor_command = motion_controller.compute_motor_command(gait_phase, amplitude)

```

#### Using PID Controller for Velocity Control
```python
from hip_controller.control.assistance_control.pid_controller import PIDController
from hip_controller.definitions import PIDConfig, LowPassFilterConfig, SolverType

# Initialize PID config
pid_config = PIDConfig(
    proportional_gain=1.0,
    integral_gain=0.1,
    derivative_gain=0.05,
    output_limits=(-10.0, 10.0)
)
filter_config = LowPassFilterConfig(
    wn=20.0, zt=1.0, x0=0.0, solver_type=SolverType.RK4
)

pid_controller = PIDController(pid_config, filter_config)

# Example: motor reference position and current position
timestamp = 0.01
motor_reference = 1.0
motor_position = 0.8

# Compute velocity command
velocity_command = pid_controller.pid_tuning(timestamp, motor_reference, motor_position)

```

#### Plotting with Simulator
```python
from hip_controller.plotter.simulator import simulate_comparison_dynamic
from pathlib import Path
from hip_controller.definitions import DATA_DIR

# Example: simulate motion mapping comparison
def motion_mapping_func(value):
    # Simplified example
    return value * 0.5, None

csv_path = DATA_DIR / "sensor_data" / "look_up_table_2026_02_25.csv"
simulate_comparison_dynamic(
    input_name="motion_mapping_key",
    expected_output_name="motion_mapping_value",
    func=motion_mapping_func,
    path=csv_path
)
```

#### Using the Main Exosuit Controller
```python
from hip_controller.control.app import ExoController
from hip_controller.definitions import ExosuitData, SensorSignal

# Initialize the main controller
controller = ExoController()

# Example sensor data for left and right limbs
left_signal = SensorSignal(timestamp=0.01, angle_rad=0.2, velocity_rad_per_sec=0.5)
right_signal = SensorSignal(timestamp=0.01, angle_rad=0.1, velocity_rad_per_sec=0.3)
sensor_data = ExosuitData(timestamp=0.01, left=left_signal, right=right_signal)

# Step the controller to compute motor commands
controller.step(sensor_data)
# Motor commands are computed internally; access via controller.left_controller.step() or similar for individual limbs
```

#### Using a Single Limb Controller
```python
from hip_controller.control.app import WalkOnController
from hip_controller.definitions import SensorSignal

# Initialize for left limb (reverse=False for left, True for right if needed)
limb_controller = WalkOnController(reverse=False, plot=False)

# Sensor signal
signal = SensorSignal(timestamp=0.01, angle_rad=0.2, velocity_rad_per_sec=0.5)

# Compute motor command
motor_command = limb_controller.step(signal)
```
## Structure
The following tree shows the important permanent files. Run `make tree` to update.
<!-- TREE-START -->
```
├── data
│   ├── evaluation_raw_data
│   │   ├── incline_walk
│   │   │   ├── incline_walk_10
│   │   │   │   ├── AB01_incline_walk_2_down10_angle.csv
│   │   │   │   ├── AB01_incline_walk_2_up10_angle.csv
│   │   │   │   ├── AB02_incline_walk_2_down10_angle.csv
│   │   │   │   ├── AB02_incline_walk_2_up10_angle.csv
│   │   │   │   ├── AB03_incline_walk_2_down10_angle.csv
│   │   │   │   ├── AB03_incline_walk_2_up10_angle.csv
│   │   │   │   ├── AB05_incline_walk_2_down10_angle.csv
│   │   │   │   ├── AB05_incline_walk_2_up10_angle.csv
│   │   │   │   ├── AB06_incline_walk_2_down10_angle.csv
│   │   │   │   ├── AB06_incline_walk_2_up10_angle.csv
│   │   │   │   ├── AB07_incline_walk_2_angle.csv
│   │   │   │   ├── AB08_incline_walk_2_down10_angle.csv
│   │   │   │   ├── AB08_incline_walk_2_up10_angle.csv
│   │   │   │   ├── AB09_incline_walk_2_down10_angle.csv
│   │   │   │   ├── AB09_incline_walk_2_up10_angle.csv
│   │   │   │   ├── AB10_incline_walk_2_down10_angle.csv
│   │   │   │   ├── AB10_incline_walk_2_up10_angle.csv
│   │   │   │   ├── AB11_incline_walk_2_down10_angle.csv
│   │   │   │   ├── AB11_incline_walk_2_up10_angle.csv
│   │   │   │   ├── AB12_incline_walk_2_down10_angle.csv
│   │   │   │   ├── AB12_incline_walk_2_up10_angle.csv
│   │   │   │   ├── AB13_incline_walk_2_down10_angle.csv
│   │   │   │   └── AB13_incline_walk_2_up10_angle.csv
│   │   │   └── incline_walk_5
│   │   │       ├── AB01_incline_walk_1_down5_angle.csv
│   │   │       ├── AB01_incline_walk_1_up5_angle.csv
│   │   │       ├── AB02_incline_walk_1_down5_angle.csv
│   │   │       ├── AB02_incline_walk_1_up5_angle.csv
│   │   │       ├── AB03_incline_walk_1_down5_angle.csv
│   │   │       ├── AB03_incline_walk_1_up5_angle.csv
│   │   │       ├── AB05_incline_walk_1_down5_angle.csv
│   │   │       ├── AB05_incline_walk_1_up5_angle.csv
│   │   │       ├── AB06_incline_walk_1_down5_angle.csv
│   │   │       ├── AB06_incline_walk_1_up5_angle.csv
│   │   │       ├── AB07_incline_walk_1_angle.csv
│   │   │       ├── AB08_incline_walk_1_down5_angle.csv
│   │   │       ├── AB08_incline_walk_1_up5_angle.csv
│   │   │       ├── AB09_incline_walk_1_down5_angle.csv
│   │   │       ├── AB09_incline_walk_1_up5_angle.csv
│   │   │       ├── AB10_incline_walk_1_down5_angle.csv
│   │   │       ├── AB10_incline_walk_1_up5_angle.csv
│   │   │       ├── AB11_incline_walk_1_down5_angle.csv
│   │   │       ├── AB11_incline_walk_1_up5_angle.csv
│   │   │       ├── AB12_incline_walk_1_down5_angle.csv
│   │   │       ├── AB12_incline_walk_1_up5_angle.csv
│   │   │       ├── AB13_incline_walk_1_down5_angle.csv
│   │   │       └── AB13_incline_walk_1_up5_angle.csv
│   │   ├── incline_walk_combined
│   │   │   ├── AB01_incline_walk_10_combined.csv
│   │   │   ├── AB01_incline_walk_5_combined.csv
│   │   │   ├── AB02_incline_walk_10_combined.csv
│   │   │   ├── AB02_incline_walk_5_combined.csv
│   │   │   ├── AB03_incline_walk_10_combined.csv
│   │   │   ├── AB03_incline_walk_5_combined.csv
│   │   │   ├── AB05_incline_walk_10_combined.csv
│   │   │   ├── AB05_incline_walk_5_combined.csv
│   │   │   ├── AB06_incline_walk_10_combined.csv
│   │   │   ├── AB06_incline_walk_5_combined.csv
│   │   │   ├── AB07_incline_walk_10_combined.csv
│   │   │   ├── AB07_incline_walk_5_combined.csv
│   │   │   ├── AB08_incline_walk_10_combined.csv
│   │   │   ├── AB08_incline_walk_5_combined.csv
│   │   │   ├── AB09_incline_walk_10_combined.csv
│   │   │   ├── AB09_incline_walk_5_combined.csv
│   │   │   ├── AB10_incline_walk_10_combined.csv
│   │   │   ├── AB10_incline_walk_5_combined.csv
│   │   │   ├── AB11_incline_walk_10_combined.csv
│   │   │   ├── AB11_incline_walk_5_combined.csv
│   │   │   ├── AB12_incline_walk_10_combined.csv
│   │   │   ├── AB12_incline_walk_5_combined.csv
│   │   │   ├── AB13_incline_walk_10_combined.csv
│   │   │   └── AB13_incline_walk_5_combined.csv
│   │   ├── normal_walk
│   │   │   ├── normal_walk_0_6
│   │   │   │   ├── AB01_normal_walk_1_0-6_angle.csv
│   │   │   │   ├── AB02_normal_walk_1_0-6_angle.csv
│   │   │   │   ├── AB03_normal_walk_1_0-6_angle.csv
│   │   │   │   ├── AB05_normal_walk_1_0-6_angle.csv
│   │   │   │   ├── AB06_normal_walk_1_0-6_angle.csv
│   │   │   │   ├── AB08_normal_walk_1_0-6_angle.csv
│   │   │   │   ├── AB09_normal_walk_1_0-6_angle.csv
│   │   │   │   ├── AB10_normal_walk_1_0-6_angle.csv
│   │   │   │   ├── AB11_normal_walk_1_0-6_angle.csv
│   │   │   │   ├── AB12_normal_walk_1_0-6_angle.csv
│   │   │   │   └── AB13_normal_walk_1_0-6_angle.csv
│   │   │   ├── normal_walk_1_2
│   │   │   │   ├── AB01_normal_walk_1_1-2_angle.csv
│   │   │   │   ├── AB02_normal_walk_1_1-2_angle.csv
│   │   │   │   ├── AB03_normal_walk_1_1-2_angle.csv
│   │   │   │   ├── AB05_normal_walk_1_1-2_angle.csv
│   │   │   │   ├── AB06_normal_walk_1_1-2_angle.csv
│   │   │   │   ├── AB08_normal_walk_1_1-2_angle.csv
│   │   │   │   ├── AB09_normal_walk_1_1-2_angle.csv
│   │   │   │   ├── AB10_normal_walk_1_1-2_angle.csv
│   │   │   │   ├── AB11_normal_walk_1_1-2_angle.csv
│   │   │   │   ├── AB12_normal_walk_1_1-2_angle.csv
│   │   │   │   └── AB13_normal_walk_1_1-2_angle.csv
│   │   │   └── normal_walk_1_8
│   │   │       ├── AB01_normal_walk_1_1-8_angle.csv
│   │   │       ├── AB02_normal_walk_1_1-8_angle.csv
│   │   │       ├── AB03_normal_walk_1_1-8_angle.csv
│   │   │       ├── AB05_normal_walk_1_1-8_angle.csv
│   │   │       ├── AB06_normal_walk_1_1-8_angle.csv
│   │   │       ├── AB08_normal_walk_1_1-8_angle.csv
│   │   │       ├── AB09_normal_walk_1_1-8_angle.csv
│   │   │       ├── AB10_normal_walk_1_1-8_angle.csv
│   │   │       ├── AB11_normal_walk_1_1-8_angle.csv
│   │   │       ├── AB12_normal_walk_1_1-8_angle.csv
│   │   │       └── AB13_normal_walk_1_1-8_angle.csv
│   │   ├── parsing_mat
│   │   │   ├── incline_walk_10
│   │   │   │   ├── 01
│   │   │   │   │   ├── incline_walk_2_down10_parsing.mat
│   │   │   │   │   └── incline_walk_2_up10_parsing.mat
│   │   │   │   ├── 02
│   │   │   │   │   ├── incline_walk_2_down10_parsing.mat
│   │   │   │   │   └── incline_walk_2_up10_parsing.mat
│   │   │   │   ├── 03
│   │   │   │   │   ├── incline_walk_2_down10_parsing.mat
│   │   │   │   │   └── incline_walk_2_up10_parsing.mat
│   │   │   │   ├── 05
│   │   │   │   │   ├── incline_walk_2_down10_parsing.mat
│   │   │   │   │   └── incline_walk_2_up10_parsing.mat
│   │   │   │   ├── 06
│   │   │   │   │   ├── incline_walk_2_down10_parsing.mat
│   │   │   │   │   └── incline_walk_2_up10_parsing.mat
│   │   │   │   ├── 08
│   │   │   │   │   ├── incline_walk_2_down10_parsing.mat
│   │   │   │   │   └── incline_walk_2_up10_parsing.mat
│   │   │   │   ├── 09
│   │   │   │   │   ├── incline_walk_2_down10_parsing.mat
│   │   │   │   │   └── incline_walk_2_up10_parsing.mat
│   │   │   │   ├── 10
│   │   │   │   │   ├── incline_walk_2_down10_parsing.mat
│   │   │   │   │   └── incline_walk_2_up10_parsing.mat
│   │   │   │   ├── 11
│   │   │   │   │   ├── incline_walk_2_down10_parsing.mat
│   │   │   │   │   └── incline_walk_2_up10_parsing.mat
│   │   │   │   ├── 12
│   │   │   │   │   ├── incline_walk_2_down10_parsing.mat
│   │   │   │   │   └── incline_walk_2_up10_parsing.mat
│   │   │   │   └── 13
│   │   │   │       ├── incline_walk_2_down10_parsing.mat
│   │   │   │       └── incline_walk_2_up10_parsing.mat
│   │   │   ├── incline_walk_5
│   │   │   │   ├── 01
│   │   │   │   │   ├── incline_walk_1_down5_parsing.mat
│   │   │   │   │   └── incline_walk_1_up5_parsing.mat
│   │   │   │   ├── 02
│   │   │   │   │   ├── incline_walk_1_down5_parsing.mat
│   │   │   │   │   └── incline_walk_1_up5_parsing.mat
│   │   │   │   ├── 03
│   │   │   │   │   ├── incline_walk_1_down5_parsing.mat
│   │   │   │   │   └── incline_walk_1_up5_parsing.mat
│   │   │   │   ├── 05
│   │   │   │   │   ├── incline_walk_1_down5_parsing.mat
│   │   │   │   │   └── incline_walk_1_up5_parsing.mat
│   │   │   │   ├── 06
│   │   │   │   │   ├── incline_walk_1_down5_parsing.mat
│   │   │   │   │   └── incline_walk_1_up5_parsing.mat
│   │   │   │   ├── 08
│   │   │   │   │   ├── incline_walk_1_down5_parsing.mat
│   │   │   │   │   └── incline_walk_1_up5_parsing.mat
│   │   │   │   ├── 09
│   │   │   │   │   ├── incline_walk_1_down5_parsing.mat
│   │   │   │   │   └── incline_walk_1_up5_parsing.mat
│   │   │   │   ├── 10
│   │   │   │   │   ├── incline_walk_1_down5_parsing.mat
│   │   │   │   │   └── incline_walk_1_up5_parsing.mat
│   │   │   │   ├── 11
│   │   │   │   │   ├── incline_walk_1_down5_parsing.mat
│   │   │   │   │   └── incline_walk_1_up5_parsing.mat
│   │   │   │   ├── 12
│   │   │   │   │   ├── incline_walk_1_down5_parsing.mat
│   │   │   │   │   └── incline_walk_1_up5_parsing.mat
│   │   │   │   └── 13
│   │   │   │       ├── incline_walk_1_down5_parsing.mat
│   │   │   │       └── incline_walk_1_up5_parsing.mat
│   │   │   ├── normal_walk_0_6
│   │   │   │   ├── 01
│   │   │   │   │   └── normal_walk_1_0-6_parsing.mat
│   │   │   │   ├── 02
│   │   │   │   │   └── normal_walk_1_0-6_parsing.mat
│   │   │   │   ├── 03
│   │   │   │   │   └── normal_walk_1_0-6_parsing.mat
│   │   │   │   ├── 05
│   │   │   │   │   └── normal_walk_1_0-6_parsing.mat
│   │   │   │   ├── 06
│   │   │   │   │   └── normal_walk_1_0-6_parsing.mat
│   │   │   │   ├── 08
│   │   │   │   │   └── normal_walk_1_0-6_parsing.mat
│   │   │   │   ├── 09
│   │   │   │   │   └── normal_walk_1_0-6_parsing.mat
│   │   │   │   ├── 10
│   │   │   │   │   └── normal_walk_1_0-6_parsing.mat
│   │   │   │   ├── 11
│   │   │   │   │   └── normal_walk_1_0-6_parsing.mat
│   │   │   │   ├── 12
│   │   │   │   │   └── normal_walk_1_0-6_parsing.mat
│   │   │   │   └── 13
│   │   │   │       └── normal_walk_1_0-6_parsing.mat
│   │   │   ├── normal_walk_1_2
│   │   │   │   ├── 01
│   │   │   │   │   └── normal_walk_1_1-2_parsing.mat
│   │   │   │   ├── 02
│   │   │   │   │   └── normal_walk_1_1-2_parsing.mat
│   │   │   │   ├── 03
│   │   │   │   │   └── normal_walk_1_1-2_parsing.mat
│   │   │   │   ├── 05
│   │   │   │   │   └── normal_walk_1_1-2_parsing.mat
│   │   │   │   ├── 06
│   │   │   │   │   └── normal_walk_1_1-2_parsing.mat
│   │   │   │   ├── 08
│   │   │   │   │   └── normal_walk_1_1-2_parsing.mat
│   │   │   │   ├── 09
│   │   │   │   │   └── normal_walk_1_1-2_parsing.mat
│   │   │   │   ├── 10
│   │   │   │   │   └── normal_walk_1_1-2_parsing.mat
│   │   │   │   ├── 11
│   │   │   │   │   └── normal_walk_1_1-2_parsing.mat
│   │   │   │   ├── 12
│   │   │   │   │   └── normal_walk_1_1-2_parsing.mat
│   │   │   │   └── 13
│   │   │   │       └── normal_walk_1_1-2_parsing.mat
│   │   │   ├── normal_walk_1_8
│   │   │   │   ├── 01
│   │   │   │   │   └── normal_walk_1_1-8_parsing.mat
│   │   │   │   ├── 02
│   │   │   │   │   └── normal_walk_1_1-8_parsing.mat
│   │   │   │   ├── 03
│   │   │   │   │   └── normal_walk_1_1-8_parsing.mat
│   │   │   │   ├── 05
│   │   │   │   │   └── normal_walk_1_1-8_parsing.mat
│   │   │   │   ├── 06
│   │   │   │   │   └── normal_walk_1_1-8_parsing.mat
│   │   │   │   ├── 08
│   │   │   │   │   └── normal_walk_1_1-8_parsing.mat
│   │   │   │   ├── 09
│   │   │   │   │   └── normal_walk_1_1-8_parsing.mat
│   │   │   │   ├── 10
│   │   │   │   │   └── normal_walk_1_1-8_parsing.mat
│   │   │   │   ├── 11
│   │   │   │   │   └── normal_walk_1_1-8_parsing.mat
│   │   │   │   ├── 12
│   │   │   │   │   └── normal_walk_1_1-8_parsing.mat
│   │   │   │   └── 13
│   │   │   │       └── normal_walk_1_1-8_parsing.mat
│   │   │   ├── stairs
│   │   │   │   ├── 01
│   │   │   │   │   ├── stairs_1_10_down_parsing.mat
│   │   │   │   │   ├── stairs_1_1_up_parsing.mat
│   │   │   │   │   ├── stairs_1_2_down_parsing.mat
│   │   │   │   │   ├── stairs_1_3_up_parsing.mat
│   │   │   │   │   ├── stairs_1_4_down_parsing.mat
│   │   │   │   │   ├── stairs_1_5_up_parsing.mat
│   │   │   │   │   ├── stairs_1_6_down_parsing.mat
│   │   │   │   │   ├── stairs_1_7_up_parsing.mat
│   │   │   │   │   ├── stairs_1_8_down_parsing.mat
│   │   │   │   │   └── stairs_1_9_up_parsing.mat
│   │   │   │   ├── 02
│   │   │   │   │   ├── stairs_1_10_down_parsing.mat
│   │   │   │   │   ├── stairs_1_11_up_parsing.mat
│   │   │   │   │   ├── stairs_1_12_down_parsing.mat
│   │   │   │   │   ├── stairs_1_1_up_parsing.mat
│   │   │   │   │   ├── stairs_1_2_down_parsing.mat
│   │   │   │   │   ├── stairs_1_3_up_parsing.mat
│   │   │   │   │   ├── stairs_1_4_down_parsing.mat
│   │   │   │   │   ├── stairs_1_5_up_parsing.mat
│   │   │   │   │   ├── stairs_1_5_up_segmented.mat
│   │   │   │   │   ├── stairs_1_6_down_parsing.mat
│   │   │   │   │   ├── stairs_1_7_up_parsing.mat
│   │   │   │   │   ├── stairs_1_8_down_parsing.mat
│   │   │   │   │   └── stairs_1_9_up_parsing.mat
│   │   │   │   ├── 03
│   │   │   │   │   ├── stairs_1_10_down_parsing.mat
│   │   │   │   │   ├── stairs_1_1_up_parsing.mat
│   │   │   │   │   ├── stairs_1_2_down_parsing.mat
│   │   │   │   │   ├── stairs_1_3_up_parsing.mat
│   │   │   │   │   ├── stairs_1_4_down_parsing.mat
│   │   │   │   │   ├── stairs_1_5_up_parsing.mat
│   │   │   │   │   ├── stairs_1_6_down_parsing.mat
│   │   │   │   │   ├── stairs_1_7_up_parsing.mat
│   │   │   │   │   ├── stairs_1_8_down_parsing.mat
│   │   │   │   │   └── stairs_1_9_up_parsing.mat
│   │   │   │   ├── 05
│   │   │   │   │   ├── stairs_1_10_down_parsing.mat
│   │   │   │   │   ├── stairs_1_1_up_parsing.mat
│   │   │   │   │   ├── stairs_1_2_down_parsing.mat
│   │   │   │   │   ├── stairs_1_3_up_parsing.mat
│   │   │   │   │   ├── stairs_1_4_down_parsing.mat
│   │   │   │   │   ├── stairs_1_5_up_parsing.mat
│   │   │   │   │   ├── stairs_1_6_down_parsing.mat
│   │   │   │   │   ├── stairs_1_7_up_parsing.mat
│   │   │   │   │   ├── stairs_1_8_down_parsing.mat
│   │   │   │   │   └── stairs_1_9_up_parsing.mat
│   │   │   │   ├── 06
│   │   │   │   │   ├── stairs_1_10_down_parsing.mat
│   │   │   │   │   ├── stairs_1_1_up_parsing.mat
│   │   │   │   │   ├── stairs_1_2_down_parsing.mat
│   │   │   │   │   ├── stairs_1_3_up_parsing.mat
│   │   │   │   │   ├── stairs_1_4_down_parsing.mat
│   │   │   │   │   ├── stairs_1_5_up_parsing.mat
│   │   │   │   │   ├── stairs_1_6_down_parsing.mat
│   │   │   │   │   ├── stairs_1_7_up_parsing.mat
│   │   │   │   │   ├── stairs_1_8_down_parsing.mat
│   │   │   │   │   └── stairs_1_9_up_parsing.mat
│   │   │   │   ├── 07
│   │   │   │   │   ├── stairs_1_10_down_parsing.mat
│   │   │   │   │   ├── stairs_1_1_up_parsing.mat
│   │   │   │   │   ├── stairs_1_2_down_parsing.mat
│   │   │   │   │   ├── stairs_1_3_up_parsing.mat
│   │   │   │   │   ├── stairs_1_4_down_parsing.mat
│   │   │   │   │   ├── stairs_1_5_up_parsing.mat
│   │   │   │   │   ├── stairs_1_6_down_parsing.mat
│   │   │   │   │   ├── stairs_1_7_up_parsing.mat
│   │   │   │   │   ├── stairs_1_8_down_parsing.mat
│   │   │   │   │   └── stairs_1_9_up_parsing.mat
│   │   │   │   ├── 08
│   │   │   │   │   ├── stairs_1_10_down_parsing.mat
│   │   │   │   │   ├── stairs_1_1_up_parsing.mat
│   │   │   │   │   ├── stairs_1_2_down_parsing.mat
│   │   │   │   │   ├── stairs_1_3_up_parsing.mat
│   │   │   │   │   ├── stairs_1_4_down_parsing.mat
│   │   │   │   │   ├── stairs_1_5_up_parsing.mat
│   │   │   │   │   ├── stairs_1_6_down_parsing.mat
│   │   │   │   │   ├── stairs_1_7_up_parsing.mat
│   │   │   │   │   ├── stairs_1_8_down_parsing.mat
│   │   │   │   │   └── stairs_1_9_up_parsing.mat
│   │   │   │   ├── 09
│   │   │   │   │   ├── stairs_1_10_down_parsing.mat
│   │   │   │   │   ├── stairs_1_1_up_parsing.mat
│   │   │   │   │   ├── stairs_1_2_down_parsing.mat
│   │   │   │   │   ├── stairs_1_3_up_parsing.mat
│   │   │   │   │   ├── stairs_1_4_down_parsing.mat
│   │   │   │   │   ├── stairs_1_5_up_parsing.mat
│   │   │   │   │   ├── stairs_1_6_down_parsing.mat
│   │   │   │   │   ├── stairs_1_7_up_parsing.mat
│   │   │   │   │   ├── stairs_1_8_down_parsing.mat
│   │   │   │   │   └── stairs_1_9_up_parsing.mat
│   │   │   │   ├── 10
│   │   │   │   │   ├── stairs_1_10_down_parsing.mat
│   │   │   │   │   ├── stairs_1_1_up_parsing.mat
│   │   │   │   │   ├── stairs_1_2_down_parsing.mat
│   │   │   │   │   ├── stairs_1_3_up_parsing.mat
│   │   │   │   │   ├── stairs_1_4_down_parsing.mat
│   │   │   │   │   ├── stairs_1_5_up_parsing.mat
│   │   │   │   │   ├── stairs_1_6_down_parsing.mat
│   │   │   │   │   ├── stairs_1_7_up_parsing.mat
│   │   │   │   │   ├── stairs_1_8_down_parsing.mat
│   │   │   │   │   └── stairs_1_9_up_parsing.mat
│   │   │   │   ├── 11
│   │   │   │   │   ├── stairs_1_10_down_parsing.mat
│   │   │   │   │   ├── stairs_1_1_up_parsing.mat
│   │   │   │   │   ├── stairs_1_2_down_parsing.mat
│   │   │   │   │   ├── stairs_1_3_up_parsing.mat
│   │   │   │   │   ├── stairs_1_4_down_parsing.mat
│   │   │   │   │   ├── stairs_1_5_up_parsing.mat
│   │   │   │   │   ├── stairs_1_6_down_parsing.mat
│   │   │   │   │   ├── stairs_1_7_up_parsing.mat
│   │   │   │   │   ├── stairs_1_8_down_parsing.mat
│   │   │   │   │   └── stairs_1_9_up_parsing.mat
│   │   │   │   ├── 12
│   │   │   │   │   ├── stairs_1_10_down_parsing.mat
│   │   │   │   │   ├── stairs_1_11_up_parsing.mat
│   │   │   │   │   ├── stairs_1_12_down_parsing.mat
│   │   │   │   │   ├── stairs_1_1_up_parsing.mat
│   │   │   │   │   ├── stairs_1_2_down_parsing.mat
│   │   │   │   │   ├── stairs_1_3_up_parsing.mat
│   │   │   │   │   ├── stairs_1_4_down_parsing.mat
│   │   │   │   │   ├── stairs_1_5_up_parsing.mat
│   │   │   │   │   ├── stairs_1_6_down_parsing.mat
│   │   │   │   │   ├── stairs_1_7_up_parsing.mat
│   │   │   │   │   ├── stairs_1_8_down_parsing.mat
│   │   │   │   │   └── stairs_1_9_up_parsing.mat
│   │   │   │   └── 13
│   │   │   │       ├── stairs_1_10_down_parsing.mat
│   │   │   │       ├── stairs_1_1_up_parsing.mat
│   │   │   │       ├── stairs_1_2_down_parsing.mat
│   │   │   │       ├── stairs_1_3_up_parsing.mat
│   │   │   │       ├── stairs_1_4_down_parsing.mat
│   │   │   │       ├── stairs_1_5_up_parsing.mat
│   │   │   │       ├── stairs_1_6_down_parsing.mat
│   │   │   │       ├── stairs_1_7_up_parsing.mat
│   │   │   │       ├── stairs_1_8_down_parsing.mat
│   │   │   │       └── stairs_1_9_up_parsing.mat
│   │   │   ├── turn_and_step_left
│   │   │   │   ├── 01
│   │   │   │   │   └── turn_and_step_1_left-turn_parsing.mat
│   │   │   │   ├── 02
│   │   │   │   │   └── turn_and_step_1_left-turn_parsing.mat
│   │   │   │   ├── 03
│   │   │   │   │   └── turn_and_step_1_left-turn_parsing.mat
│   │   │   │   ├── 05
│   │   │   │   │   └── turn_and_step_1_left-turn_parsing.mat
│   │   │   │   ├── 06
│   │   │   │   │   └── turn_and_step_1_left-turn_parsing.mat
│   │   │   │   ├── 07
│   │   │   │   │   └── turn_and_step_1_left-turn_parsing.mat
│   │   │   │   ├── 08
│   │   │   │   │   └── turn_and_step_1_left-turn_parsing.mat
│   │   │   │   ├── 09
│   │   │   │   │   └── turn_and_step_1_left-turn_parsing.mat
│   │   │   │   ├── 10
│   │   │   │   │   └── turn_and_step_1_left-turn_parsing.mat
│   │   │   │   ├── 11
│   │   │   │   │   └── turn_and_step_1_left-turn_parsing.mat
│   │   │   │   ├── 12
│   │   │   │   │   └── turn_and_step_1_left-turn_parsing.mat
│   │   │   │   └── 13
│   │   │   │       └── turn_and_step_1_left-turn_parsing.mat
│   │   │   └── turn_and_step_right
│   │   │       ├── 01
│   │   │       │   └── turn_and_step_1_right-turn_parsing.mat
│   │   │       ├── 02
│   │   │       │   └── turn_and_step_1_right-turn_parsing.mat
│   │   │       ├── 03
│   │   │       │   └── turn_and_step_1_right-turn_parsing.mat
│   │   │       ├── 05
│   │   │       │   └── turn_and_step_1_right-turn_parsing.mat
│   │   │       ├── 06
│   │   │       │   └── turn_and_step_1_right-turn_parsing.mat
│   │   │       ├── 07
│   │   │       │   └── turn_and_step_1_right-turn_parsing.mat
│   │   │       ├── 08
│   │   │       │   └── turn_and_step_1_right-turn_parsing.mat
│   │   │       ├── 09
│   │   │       │   └── turn_and_step_1_right-turn_parsing.mat
│   │   │       ├── 10
│   │   │       │   └── turn_and_step_1_right-turn_parsing.mat
│   │   │       ├── 11
│   │   │       │   └── turn_and_step_1_right-turn_parsing.mat
│   │   │       ├── 12
│   │   │       │   └── turn_and_step_1_right-turn_parsing.mat
│   │   │       └── 13
│   │   │           └── turn_and_step_1_right-turn_parsing.mat
│   │   ├── stairs_combined
│   │   │   ├── AB01_stairs_combined.csv
│   │   │   ├── AB02_stairs_combined.csv
│   │   │   ├── AB03_stairs_combined.csv
│   │   │   ├── AB05_stairs_combined.csv
│   │   │   ├── AB06_stairs_combined.csv
│   │   │   ├── AB07_stairs_combined.csv
│   │   │   ├── AB08_stairs_combined.csv
│   │   │   ├── AB09_stairs_combined.csv
│   │   │   ├── AB10_stairs_combined.csv
│   │   │   ├── AB11_stairs_combined.csv
│   │   │   ├── AB12_stairs_combined.csv
│   │   │   └── AB13_stairs_combined.csv
│   │   └── turn_and_step
│   │       ├── turn_left
│   │       │   ├── AB01_turn_and_step_1_left-turn_angle.csv
│   │       │   ├── AB02_turn_and_step_1_left-turn_angle.csv
│   │       │   ├── AB03_turn_and_step_1_left-turn_angle.csv
│   │       │   ├── AB05_turn_and_step_1_left-turn_angle.csv
│   │       │   ├── AB06_turn_and_step_1_left-turn_angle.csv
│   │       │   ├── AB07_turn_and_step_1_left-turn_angle.csv
│   │       │   ├── AB08_turn_and_step_1_left-turn_angle.csv
│   │       │   ├── AB09_turn_and_step_1_left-turn_angle.csv
│   │       │   ├── AB10_turn_and_step_1_left-turn_angle.csv
│   │       │   ├── AB11_turn_and_step_1_left-turn_angle.csv
│   │       │   ├── AB12_turn_and_step_1_left-turn_angle.csv
│   │       │   └── AB13_turn_and_step_1_left-turn_angle.csv
│   │       └── turn_right
│   │           ├── AB01_turn_and_step_1_right-turn_angle.csv
│   │           ├── AB02_turn_and_step_1_right-turn_angle.csv
│   │           ├── AB03_turn_and_step_1_right-turn_angle.csv
│   │           ├── AB05_turn_and_step_1_right-turn_angle.csv
│   │           ├── AB06_turn_and_step_1_right-turn_angle.csv
│   │           ├── AB07_turn_and_step_1_right-turn_angle.csv
│   │           ├── AB08_turn_and_step_1_right-turn_angle.csv
│   │           ├── AB09_turn_and_step_1_right-turn_angle.csv
│   │           ├── AB10_turn_and_step_1_right-turn_angle.csv
│   │           ├── AB11_turn_and_step_1_right-turn_angle.csv
│   │           ├── AB12_turn_and_step_1_right-turn_angle.csv
│   │           └── AB13_turn_and_step_1_right-turn_angle.csv
│   ├── logs
│   └── sensor_data
│       ├── arduino2_2026_03_23.csv
│       ├── arduino_2026_03_23.csv
│       ├── data_input_2025_12_17.csv
│       ├── data_input_filtered_2026_01_09.csv
│       ├── data_kinematics_2026_02_16.csv
│       ├── data_raw_2025_12_17.xlsx
│       └── look_up_table_2026_02_25.csv
├── docs
│   ├── UML_hip_controller_2026_03_10.json
│   └── paper.pdf
├── scripts
│   ├── compare_all.py
│   ├── compare_matlab.py
│   ├── compare_matlab_module.py
│   ├── csv_converter.py
│   ├── csv_utils.py
│   ├── evaluation_matplotlib.py
│   ├── evaluation_record.py
│   ├── live_comparison_plot.py
│   ├── main.py
│   ├── mat_to_csv.py
│   ├── normalize_output_time.py
│   ├── reference_versus_calculated.py
│   ├── script.py
│   └── simulator.py
├── src
│   └── hip_controller
│       ├── control
│       │   ├── gait_phase_control
│       │   │   ├── gait_controller.py
│       │   │   ├── motion_state_machine.py
│       │   │   ├── steady_state_tracker.py
│       │   │   └── stride_event_detector.py
│       │   ├── motor_reference_control
│       │   │   ├── amplitude_modulation.py
│       │   │   ├── motor_reference_controller.py
│       │   │   └── pid_controller.py
│       │   ├── signal_processing
│       │   │   ├── drift_removal.py
│       │   │   ├── sensor_preprocessor.py
│       │   │   └── velocity_estimation.py
│       │   ├── __init__.py
│       │   └── app.py
│       ├── filters
│       │   ├── discrete_derivative_filter.py
│       │   ├── kalman_filter.py
│       │   ├── notch_filter.py
│       │   ├── second_order_low_pass_filter.py
│       │   └── sogi_fll_filter.py
│       ├── plotter
│       │   ├── csv_player.py
│       │   └── live_phase_portrait.py
│       ├── utils
│       │   ├── math_utils.py
│       │   ├── state_space.py
│       │   └── utils.py
│       ├── __init__.py
│       ├── __main__.py
│       └── definitions.py
├── tests
│   ├── controller_test
│   │   ├── assistance_testing
│   │   │   ├── amplitude_test.py
│   │   │   └── mid_level_test.py
│   │   ├── gait_phase_testing
│   │   │   ├── high_level_test.py
│   │   │   ├── motion_state_machine_test.py
│   │   │   └── stride_event_detector_test.py
│   │   ├── pre_process_testing
│   │   │   ├── drift_removal_test.py
│   │   │   ├── filtering_test.py
│   │   │   ├── kalman_test.py
│   │   │   └── second_order_lpf_test.py
│   │   ├── testing_data
│   │   │   ├── amplitude_modulation_2026_03_03.csv
│   │   │   ├── extrema_2026_01_26.csv
│   │   │   ├── filtering_2026_03_19.csv
│   │   │   ├── gait_phase_left_2026_03_03.csv
│   │   │   ├── look_up_table_2026_02_25.csv
│   │   │   ├── reference_motion_2026_03_05.csv
│   │   │   ├── reference_motion_2026_03_06.csv
│   │   │   ├── second_order_lpf_2026_03_06.csv
│   │   │   ├── stride_event_detector_2026_02_26.csv
│   │   │   ├── valid_trigger_left_2026_01_15.csv
│   │   │   └── zero_crossing_left_2026_01_09.csv
│   │   └── app_test.py
│   ├── utils_test
│   │   ├── csv_player_test.py
│   │   ├── math_utils_test.py
│   │   └── utils_test.py
│   ├── __init__.py
│   └── conftest.py
├── .darglint
├── .dockerignore
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── CONTRIBUTING.md
├── Dockerfile
├── LICENSE
├── Makefile
├── README.md
├── pyproject.toml
├── repo_tree.py
└── uv.lock
```
<!-- TREE-END -->

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project.

## License

This project is licensed under the terms specified in [LICENSE](LICENSE).

## Authors

- Tsmorz (tony.smoragiewicz@tum.de)
- Zhongge Lin (zg.lin@outlook.com)

## Acknowledgments

Developed at TUM-Aries-Lab as part of research on exoskeleton control systems.
