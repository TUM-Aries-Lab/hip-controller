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
It's super easy to publish your own packages on PyPI. To build and publish this package run:

```bash
uv build
uv publish  # make sure your version in pyproject.toml is updated
```
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

## Structure
The following tree shows the important permanent files. Run `make tree` to update.
<!-- TREE-START -->
```
├── data
│   ├── logs
│   └── sensor_data
│       ├── data_input_2025_12_17.csv
│       ├── data_input_filtered_2026_01_09.csv
│       ├── data_kinematics_2026_02_16.csv
│       ├── data_raw_2025_12_17.xlsx
│       └── look_up_table_2026_02_25.csv
├── docs
│   └── paper.pdf
├── src
│   ├── data
│   │   └── logs
│   └── hip_controller
│       ├── control
│       │   ├── high_level_controller
│       │   │   ├── high_level.py
│       │   │   ├── motion_state_machine.py
│       │   │   ├── steady_state_tracker.py
│       │   │   └── stride_event_detector.py
│       │   ├── low_level_controller
│       │   │   └── low_level.py
│       │   ├── mid_level_controller
│       │   │   ├── amplitude_modulation.py
│       │   │   ├── cubic_spline_interpolation.py
│       │   │   ├── kalman.py
│       │   │   └── mid_level.py
│       │   ├── __init__.py
│       │   └── app.py
│       ├── plotter
│       │   ├── csv_player.py
│       │   ├── phase_portrait.py
│       │   ├── simulator.py
│       │   └── time_plot.py
│       ├── utils
│       │   ├── math_utils.py
│       │   ├── state_space.py
│       │   └── utils.py
│       ├── __init__.py
│       ├── __main__.py
│       └── definitions.py
├── tests
│   ├── controller_test
│   │   ├── high_level_testing
│   │   │   ├── high_level_testing_data
│   │   │   │   ├── ang_ss_2026_01_26.csv
│   │   │   │   ├── extrema_2026_01_26.csv
│   │   │   │   ├── gait_phase_left_2026_01_21.csv
│   │   │   │   ├── gait_phase_left_2026_03_03.csv
│   │   │   │   ├── stride_event_detector_2026_02_26.csv
│   │   │   │   ├── valid_trigger_left_2026_01_15.csv
│   │   │   │   ├── vel_ss_2026_01_26.csv
│   │   │   │   └── zero_crossing_left_2026_01_09.csv
│   │   │   ├── high_level_test.py
│   │   │   ├── motion_state_machine_test.py
│   │   │   └── stride_event_detector_test.py
│   │   ├── mid_level_testing
│   │   │   ├── mid_level_testing_data
│   │   │   │   ├── amplitude_modulation_2026_03_03.csv
│   │   │   │   ├── look_up_table_2026_02_25.csv
│   │   │   │   └── sinusoidal_behavior_left_2026_01_29.csv
│   │   │   ├── amplitude_test.py
│   │   │   ├── cubic_spline_interpolation_test.py
│   │   │   ├── kalman_test.py
│   │   │   └── mid_level_test.py
│   │   └── app_test.py
│   ├── utilstest
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
