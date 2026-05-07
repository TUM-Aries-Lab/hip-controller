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

def plot_comparison(data: ComparisonData, metrics: ComparisonMetrics, filepath: Path) -> None:
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
    fig.savefig(filepath, dpi=150)


def plot_aggregate_comparison(data_list: list[ComparisonData], filepath: Path) -> None:
    """
    Plot all matlab and output signals per pair, with mean and std envelope.
    Each pair gets its own subplot (4 rows).
    """

    pairs = [
        ("Gait Phase — Left",
         [d.matlab_gait_phase_left for d in data_list],
         [d.output_gait_phase_left for d in data_list]),
        ("Gait Phase — Right",
         [d.matlab_gait_phase_right for d in data_list],
         [d.output_gait_phase_right for d in data_list]),
        ("Motor Command — Left",
         [d.matlab_motor_command_left for d in data_list],
         [d.output_motor_command_left for d in data_list]),
        ("Motor Command — Right",
         [d.matlab_motor_command_right for d in data_list],
         [d.output_motor_command_right for d in data_list]),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(10, 16))
    fig.suptitle("Aggregate MATLAB vs Output Comparison", fontsize=14, fontweight="bold")

    for ax, (title, matlab_signals, output_signals) in zip(axes, pairs):
        matlab_arr = np.stack(matlab_signals)   # (n_trials, n_samples)
        output_arr = np.stack(output_signals)

        matlab_mean, matlab_std = matlab_arr.mean(axis=0), matlab_arr.std(axis=0)
        output_mean, output_std = output_arr.mean(axis=0), output_arr.std(axis=0)
        xs = np.arange(matlab_arr.shape[1])

        # Individual trials
        for trial in matlab_arr:
            ax.plot(trial, color="tab:blue", alpha=0.15, linewidth=0.8)
        for trial in output_arr:
            ax.plot(trial, color="tab:orange", alpha=0.15, linewidth=0.8)

        # Std envelope
        ax.fill_between(xs, matlab_mean - matlab_std, matlab_mean + matlab_std,
                        color="tab:blue", alpha=0.2)
        ax.fill_between(xs, output_mean - output_std, output_mean + output_std,
                        color="tab:orange", alpha=0.2)

        # Mean trend
        ax.plot(matlab_mean, color="tab:blue", linewidth=2, label="MATLAB mean")
        ax.plot(output_mean, color="tab:orange", linewidth=2, linestyle="--", label="Output mean")

        ax.set_title(title)
        ax.set_xlabel("Sample")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(filepath, dpi=150)

def plot_rmse_errorbar(trial_groups: dict[str, list[ComparisonData]]) -> None:
    """
    For each group, compute per-trial average RMSE (mean of left+right legs,
    for both gait phase and motor command), then plot an error bar chart
    sorted by ascending mean RMSE per signal type.
    """

    def per_trial_avg_rmse(data_list: list[ComparisonData]) -> dict[str, np.ndarray]:
        """Returns arrays of shape (n_trials,) for each signal type."""
        gait, motor = [], []
        for d in data_list:
            m = compute_metrics(d)
            gait.append(np.mean([m.gait_phase_left_rmse, m.gait_phase_right_rmse]))
            motor.append(np.mean([m.motor_command_left_rmse, m.motor_command_right_rmse]))
        return {
            "Gait Phase":    np.array(gait),
            "Motor Command": np.array(motor),
        }

    # Compute stats per group
    group_stats: dict[str, dict[str, tuple[float, float]]] = {}
    for group_name, data_list in trial_groups.items():
        trial_rmses = per_trial_avg_rmse(data_list)
        group_stats[group_name] = {
            signal: (values.mean(), values.std())
            for signal, values in trial_rmses.items()
        }

    signal_types = ["Gait Phase", "Motor Command"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("RMSE by Trial Group", fontsize=14, fontweight="bold")

    for ax, signal in zip(axes, signal_types):
        # Sort groups by ascending mean RMSE for this signal
        sorted_groups = sorted(group_stats.items(), key=lambda kv: kv[1][signal][0])
        labels = [g for g, _ in sorted_groups]
        means  = np.array([stats[signal][0] for _, stats in sorted_groups])
        stds   = np.array([stats[signal][1] for _, stats in sorted_groups])

        xs = np.arange(len(labels))
        ax.errorbar(xs, means, yerr=stds, fmt="o", capsize=5,
                    linewidth=1.5, markersize=6, color="tab:blue", ecolor="tab:blue")

        ax.set_title(signal)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("RMSE")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()

# --------------------------------------------------------------------------------------------------------------

def plot_all_pipeline():
    """
    Plot comparisons for all MATLAB and output file pairs.
    """
    root = Path(__file__).resolve().parents[1]
    mat_folder = root / "scripts/matlab_output_data"
    output_folder = root / "scripts/normalized_output"
    plot_folder = root / "scripts/matlab_output_plots"
    plot_folder.mkdir(parents=True, exist_ok=True)
    for matlab_file, output_file in iter_matlab_module_file_pair(matlab_folder=mat_folder, output_folder=output_folder):
        print(matlab_file)
        print(output_file)
        comparison_data = parse_matlab_module(matlab_file=matlab_file, output_file=output_file)
        metrics = compute_metrics(comparison_data)
        plot_comparison(data=comparison_data, metrics=metrics, filepath=Path(plot_folder)/matlab_file.stem)

def plot_combined_pipeline():
    """
    Plot aggregate comparisons for selected MATLAB and output file pairs.
    """
    root = Path(__file__).resolve().parents[1]
    mat_folder = root / "scripts/matlab_output_data"
    output_folder = root / "scripts/normalized_output"
    plot_folder = root / "scripts/matlab_output_plots"
    plot_folder.mkdir(parents=True, exist_ok=True)
    mat_file_list: list[Path] = []
    output_file_list: list[Path] = []
    comparison_data_combined: list[ComparisonData] = []
    for matlab_file, output_file in iter_matlab_module_file_pair(matlab_folder=mat_folder, output_folder=output_folder):
        if "normal_walk_1_1-2" in matlab_file.stem:
            print(matlab_file)
            mat_file_list.append(matlab_file)
            print(output_file)
            output_file_list.append(output_file)
            comparison_data = parse_matlab_module(matlab_file=matlab_file, output_file=output_file)
            comparison_data_combined.append(comparison_data)

    plot_aggregate_comparison(data_list=comparison_data_combined, filepath=Path(plot_folder) / "normal_walk_1_1-2")

def plot_error_bar_pipeline():
    """
    Plot RMSE error bars for different trial groups.
    """
    root = Path(__file__).resolve().parents[1]
    mat_folder = root / "scripts/matlab_output_data"
    output_folder = root / "scripts/normalized_output"
    plot_folder = root / "scripts/matlab_output_plots"
    plot_folder.mkdir(parents=True, exist_ok=True)
    trial_group_names: list[str] = [
        "incline_walk_5",
        "incline_walk_10",
        "normal_walk_1_0-6",
        "normal_walk_1_1-2",
        "normal_walk_1_1-8",
        #"stairs_combined",
        "turn_and_step_1_left",
        "turn_and_step_1_right"
    ]
    trial_group_rename: list[str] = [
        "incline_walk_5",
        "incline_walk_10",
        "normal_walk_0-6",
        "normal_walk_1-2",
        "normal_walk_1-8",
        #"stairs_combined",
        "turn_and_step_left",
        "turn_and_step_right"
    ]
    trial_groups: dict[str, list[ComparisonData]] = {name: [] for name in trial_group_rename}
    for matlab_file, output_file in iter_matlab_module_file_pair(matlab_folder=mat_folder, output_folder=output_folder):
        index = 0
        for trial in trial_group_names:
            if trial in matlab_file.stem:
                trial_groups[trial_group_rename[index]].append(parse_matlab_module(matlab_file=matlab_file, output_file=output_file))
            index += 1
    plot_rmse_errorbar(trial_groups=trial_groups)


if __name__ == "__main__":
    plot_error_bar_pipeline()
