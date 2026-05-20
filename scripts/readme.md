# Scripts Folder

This document describes the purpose of each script in `scripts/` and how they are intended to be used.


## Script descriptions

### `compare_all.py`
A comparison utility for MATLAB reference data and module output data. It parses matched MATLAB/output CSV pairs, computes RMSE/MAPE metrics, and generates comparison plots.

### `compare_matlab.py`
Same purpose as `compare_all.py`: compare Simulink/MATLAB CSV data against module output data. Use it to inspect phase and motor command agreement between MATLAB and Python results.

### `compare_matlab_module.py`
A second comparison helper with the same MATLAB vs module output workflow. It is also intended for RMSE/MAPE calculation and plotting of matched CSV pairs.

### `controller_simulator.py`
A GUI-focused simulation script. It can replay CSV data through the controller and display live comparison plots. Key functions:
- `simulate_controller_with_data(...)` for full controller playback
- `simulate_comparison_dynamic(...)` for comparing actual output with expected output

### `csv_converter.py`
CSV processing utilities for walking data. Primary helpers:
- `concatenate_stairs(...)` to merge stair trial CSVs per participant
- `combine_two_files(...)` to join two CSVs while keeping the selected columns

### `csv_player.py`
A stateful CSV player for real-time-style playback. It loads a CSV once and returns one row at a time so GUI/plotting components can simulate sensor streaming.

### `evaluation_matplotlib.py`
Plotting utilities for evaluation output. It contains functions to draw gait phase, filtered signals, amplitude, and motor command figures from evaluation CSV files.

### `evaluation_record.py`
The evaluation preprocessing pipeline. It reads raw sensor CSVs from `data/evaluation_raw_data/`, downsamples and converts them, and writes the resulting evaluation CSV files under `scripts/evaluation_output/`.

### `live_comparison_plot.py`
A PyQt6 plot window for live signal comparison. Used by simulator components to show input, actual output, and expected output in real time.

### `mat_to_csv.py`
MAT file conversion helper. It extracts `left` and `right` arrays from `.mat` files and writes them as CSV, preserving folder structure.

### `normalize_output_time.py`
Normalizes output CSV time columns so `time (s)` starts at `0.00` and increments by `0.01`. Useful for preparing evaluation/output files for plotting or comparison.

### `plot_scenarios.py`
A comparison plot module for preprocessing validation. It replays CSV rows through a callable, then plots actual vs expected output and residuals in a 3-panel figure.

### `reference_versus_calculated.py`
A gait-phase comparison tool. It builds a reference phase signal from left/right cycle boundaries and compares it against calculated gait phase output, including RMSE reporting and plots.

## Usage notes

- Many scripts are library-style and are best used by importing their functions from Python.
- `evaluation_record.py`, `mat_to_csv.py`, and `normalize_output_time.py` include `__main__` runners for direct execution.
- Visualization scripts generally require `matplotlib`, and GUI scripts require `PyQt6`/`pyqtgraph`.

For exact call patterns, open the corresponding script and inspect the top-level functions or the `if __name__ == '__main__'` section.

## Examples

### Run directly from the shell

```bash
python scripts/evaluation_record.py
python scripts/mat_to_csv.py
python scripts/normalize_output_time.py
```

### Compare MATLAB vs module CSV outputs

```python
from scripts.compare_matlab import parse_matlab_module, compute_metrics
from pathlib import Path

matlab_csv = Path('data/evaluation_raw_data/matlab_ref.csv')
output_csv = Path('data/evaluation_output/module_out.csv')
comparison = parse_matlab_module(matlab_csv, output_csv)
metrics = compute_metrics(comparison)
```

### Use the CSV player for real-time-style replay

```python
from scripts.csv_player import ScriptPlayer

player = ScriptPlayer(Path('data/evaluation_raw_data/some_walk.csv'))
while player.has_next_line():
    row = player.get_data_from_csv('angle_right (rad)', 'gait_phase_right (rad)')
```

### Build a preprocessing comparison plot

```python
from scripts.plot_scenarios import plot_preprocessor_comparison
from hip_controller.definitions import SensorSignal
from hip_controller.control.signal_processing.sensor_preprocessor import SensorPreprocessor, PreprocessorConfig

preprocessor = SensorPreprocessor(PreprocessorConfig())
plot_preprocessor_comparison(
    csv_path='scripts/evaluation_output/normal_walk/AB01_normal_walk.csv',
    time_col='time (s)',
    input_col='angle_right (rad)',
    expected_output_col='filtered_velocity_right (rad/s)',
    build_signal=lambda t, a: SensorSignal(timestamp=t, angle_rad=a, velocity_rad_per_sec=0.0),
    run_callable=preprocessor.filter,
    extract_output=lambda sig: sig.velocity_rad_per_sec,
)
```

### Simulate controller playback

```python
from scripts.controller_simulator import simulate_controller_with_data
from pathlib import Path

simulate_controller_with_data(csv_path=Path('data/evaluation_raw_data/normal_walk/AB01_normal_walk.csv'))
```

### Normalize evaluation output time values

```python
from scripts.normalize_output_time import normalize_output_folder
normalize_output_folder('scripts/evaluation_output', 'scripts/normalized_output')
```
