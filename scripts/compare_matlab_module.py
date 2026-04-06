"""Compare the output data of MATLAB Simulink with the module's output data. """

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

class DataFileKeys(StrEnum):
    """Column names in matlab and output csv files."""
    output_timekey = "time (s)"
    matlab_timekey = "Time"
    output_gait_phase_left = "gait_phase_left (rad)"
    output_gait_phase_right = "gait_phase_right (rad)"
    matlab_gait_phase_left = "gait_phase_left"
    matlab_gait_phase_right = "gait_phase_right"
    output_motor_command_left = "motor_command_left (rad/s)"
    output_motor_command_right = "motor_command_right (rad/s)"
    matlab_motor_command_left = "motor_command_left"
    matlab_motor_command_right = "motor_command_right"


@dataclass
class ComparisonData:
    """Data pairs of gait phases and motor commands."""
    matlab_gait_phase_left: np.ndarray
    matlab_gait_phase_right: np.ndarray
    output_gait_phase_left: np.ndarray
    output_gait_phase_right: np.ndarray
    matlab_motor_command_left: np.ndarray
    matlab_motor_command_right: np.ndarray
    output_motor_command_left: np.ndarray
    output_motor_command_right: np.ndarray

@dataclass
class ComparisonMetrics:
    gait_phase_left_rmse: float
    gait_phase_left_mape: float
    gait_phase_right_rmse: float
    gait_phase_right_mape: float
    motor_command_left_rmse: float
    motor_command_left_mape: float
    motor_command_right_rmse: float
    motor_command_right_mape: float

def parse_matlab_module(matlab_file: str | Path, output_file: str | Path) -> ComparisonData:
    """Parse the file pairs and extract the relevant data for comparison. """
    matlab_df = pd.read_csv(matlab_file)
    output_df = pd.read_csv(output_file)

    if len(matlab_df[DataFileKeys.matlab_timekey]) != len(output_df[DataFileKeys.output_timekey]):
        raise RuntimeError("The lengths of two datafiles must match.")
    comparison_data = ComparisonData(
        matlab_gait_phase_left=matlab_df[DataFileKeys.matlab_gait_phase_left].to_numpy(),
        matlab_gait_phase_right=matlab_df[DataFileKeys.matlab_gait_phase_right].to_numpy(),
        output_gait_phase_left=output_df[DataFileKeys.output_gait_phase_left].to_numpy(),
        output_gait_phase_right=output_df[DataFileKeys.output_gait_phase_right].to_numpy(),
        matlab_motor_command_left=matlab_df[DataFileKeys.matlab_motor_command_left].to_numpy(),
        matlab_motor_command_right=matlab_df[DataFileKeys.matlab_motor_command_right].to_numpy(),
        output_motor_command_left=output_df[DataFileKeys.output_motor_command_left].to_numpy(),
        output_motor_command_right=output_df[DataFileKeys.output_motor_command_right].to_numpy()
    )
    return comparison_data

def iter_matlab_module_file_pair(matlab_folder: str | Path, output_folder: str | Path):
    """Extract the MATLAB / module datafile pairs.

    The two folders' structures must match.
    """
    matlab_path = Path(matlab_folder)
    output_path = Path(output_folder)
    for matlab_file in matlab_path.rglob("*.csv"):
        output_file = output_path / matlab_file.relative_to(matlab_path)
        if not output_file.exists():
            raise FileNotFoundError(f"No matching file for {matlab_file}")

        yield matlab_file, output_file


def compute_metrics(data: ComparisonData) -> ComparisonMetrics:
    """Calculate RMSE and MAPE for each pair, using matlab arrays as reference."""

    def rmse(ref: np.ndarray, out: np.ndarray) -> float:
        return np.sqrt(np.mean((ref - out) ** 2))

    def mape(ref: np.ndarray, out: np.ndarray) -> float:
        mask = ref != 0  # avoid division by zero
        return np.mean(np.abs((ref[mask] - out[mask]) / ref[mask])) * 100 # type: ignore

    return ComparisonMetrics(
        gait_phase_left_rmse=rmse(data.matlab_gait_phase_left, data.output_gait_phase_left),
        gait_phase_left_mape=mape(data.matlab_gait_phase_left, data.output_gait_phase_left),
        gait_phase_right_rmse=rmse(data.matlab_gait_phase_right, data.output_gait_phase_right),
        gait_phase_right_mape=mape(data.matlab_gait_phase_right, data.output_gait_phase_right),
        motor_command_left_rmse=rmse(data.matlab_motor_command_left, data.output_motor_command_left),
        motor_command_left_mape=mape(data.matlab_motor_command_left, data.output_motor_command_left),
        motor_command_right_rmse=rmse(data.matlab_motor_command_right, data.output_motor_command_right),
        motor_command_right_mape=mape(data.matlab_motor_command_right, data.output_motor_command_right),
    )

def plot_comparison(data: ComparisonData, metrics: ComparisonMetrics) -> None:
    """Plot matlab vs output for all 4 pairs with RMSE and MAPE annotations."""

    pairs = [
        ("Gait Phase — Left",    data.matlab_gait_phase_left,     data.output_gait_phase_left,
         metrics.gait_phase_left_rmse,     metrics.gait_phase_left_mape),
        ("Gait Phase — Right",   data.matlab_gait_phase_right,    data.output_gait_phase_right,
         metrics.gait_phase_right_rmse,    metrics.gait_phase_right_mape),
        ("Motor Command — Left", data.matlab_motor_command_left,  data.output_motor_command_left,
         metrics.motor_command_left_rmse,  metrics.motor_command_left_mape),
        ("Motor Command — Right",data.matlab_motor_command_right, data.output_motor_command_right,
         metrics.motor_command_right_rmse, metrics.motor_command_right_mape),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    fig.suptitle("MATLAB vs Output Comparison", fontsize=14, fontweight="bold")

    for ax, (title, matlab, output, rmse_val, mape_val) in zip(axes.flat, pairs):
        ax.plot(matlab, label="MATLAB", linewidth=1.5)
        ax.plot(output, label="Output", linewidth=1.5, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Sample")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.text(
            0.98, 0.98,
            f"RMSE: {rmse_val:.4f}\nMAPE: {mape_val:.2f}%",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    mat_folder = root / "scripts/matlab_output_data"
    output_folder = root / "scripts/normalized_output"
    for matlab_file, output_file in iter_matlab_module_file_pair(matlab_folder=mat_folder, output_folder=output_folder):
        if matlab_file.stem == "AB01_normal_walk_1_0-6_angle":
            print(matlab_file)
            print(output_file)
            comparison_data = parse_matlab_module(matlab_file=matlab_file, output_file=output_file)
            metrics = compute_metrics(comparison_data)
            plot_comparison(comparison_data, metrics)
            break # TODO
