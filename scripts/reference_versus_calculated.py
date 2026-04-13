from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from loguru import logger


class Scenarios(StrEnum):
        incline_walk_down5 = "incline_walk_1_down5_parsing"
        incline_walk_up5 = "incline_walk_1_up5_parsing"
        incline_walk_up10 = "incline_walk_2_up10_parsing"
        incline_walk_down10 = "incline_walk_2_down10_parsing"
        normal_walk_0_6 = "normal_walk_1_0-6_parsing"
        incline_walk_1_2 = "normal_walk_1_1-2_parsing"
        incline_walk_1_8 = "normal_walk_1_1-8_parsing"

@dataclass
class EvaluationData:
    subject: int
    scenario: str
    ref_offset: float
    num_cycles: int
    RMSE: float
    RMSE_spike_rm: float

def load_and_validate(reference_path: str | Path, calculated_path: str | Path):
    """Load and validate both CSV files."""
    ref_df = pd.read_csv(reference_path)
    calc_df = pd.read_csv(calculated_path)

    required_ref_left = {"left_col1", "left_col2"}
    required_ref_right = {"right_col1", "right_col2"}
    required_calc = {"time (s)", "gait_phase_left (rad)"}

    if not required_ref_left.issubset(ref_df.columns):
        if not required_ref_right.issubset(ref_df.columns):
            raise ValueError(f"Reference CSV must contain columns: {required_ref_left}. Found: {set(ref_df.columns)}")
        is_left = False
    else:
        is_left = True
    if not required_calc.issubset(calc_df.columns):
        raise ValueError(f"Calculated CSV must contain columns: {required_calc}. Found: {set(calc_df.columns)}")

    return ref_df, calc_df, is_left


def build_reference_gait_phase(ref_df: pd.DataFrame, time_axis: np.ndarray, is_left: bool, offset: float = 0.0) -> np.ndarray:
    """
    For every gait cycle defined by (left_col1, left_col2), linearly interpolate
    gait phase from -pi to pi over that interval.  Time points outside any cycle
    are set to NaN.
    """
    ref_phase = np.full_like(time_axis, np.nan, dtype=float)

    for _, row in ref_df.iterrows():
        if is_left:
            t_start, t_end = float(row["left_col1"]) + offset, float(row["left_col2"]) + offset
        else:
            t_start, t_end = float(row["right_col1"]) + offset, float(row["right_col2"]) + offset
        if t_end <= t_start:
            logger.warning(f"  [warning] skipping degenerate cycle: start={t_start}, end={t_end}")
            continue

        mask = (time_axis >= t_start) & (time_axis <= t_end)
        if not np.any(mask):
            continue

        t_cycle = time_axis[mask]

        # Linear interpolation: maps [t_start, t_end] → [-π, π]
        ref_phase[mask] = np.interp(t_cycle, [t_start, t_end], [-np.pi, np.pi])

    return ref_phase


def compute_metrics(ref: np.ndarray, calc: np.ndarray):
    """Compute RMSE."""
    valid = ~(np.isnan(ref) | np.isnan(calc))
    r, c = ref[valid], calc[valid]

    rmse = np.sqrt(np.mean((r - c) ** 2))

    # Filter out the spikes
    error = r - c
    mask = np.abs(error) < 5.0  # Threshold for outlier removal (adjust as needed)
    rmse_spike_removed = np.sqrt(np.mean((error[mask]) ** 2))

    return rmse, rmse_spike_removed, valid


def plot_comparison(
    time_axis: np.ndarray,
    ref_df: pd.DataFrame,
    calc_phase: np.ndarray,
    is_left: bool,
    scenario: str,
    save_path: str | None = None,
    offset: float = 0.0
):
    """Draw two-panel comparison figure."""

    ref_phase = build_reference_gait_phase(ref_df=ref_df, time_axis=time_axis, is_left=is_left)
    rmse, rmse_spike_removed, valid_mask = compute_metrics(ref_phase, calc_phase)
    leg_title = "Left Leg" if is_left else "Right Leg"

    fig = plt.figure(figsize=(13, 9))
    fig.suptitle(f"Gait Phase Comparison – {scenario} – {leg_title}", fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45)

    # ── Panel 1: Phase signals ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time_axis, ref_phase,  label="Reference (interpolated)", color="#2563EB", linewidth=1.4, zorder=3)
    ax1.plot(time_axis, calc_phase, label="Calculated",               color="#DC2626", linewidth=1.0,
             linestyle="--", alpha=0.85, zorder=2)
    ax1.set_ylabel("Gait Phase (rad)", fontsize=10)
    ax1.set_xlabel("Time (s)", fontsize=10)
    ax1.set_title("Reference vs. Calculated Gait Phase", fontsize=11)
    ax1.set_ylim(-np.pi - 0.3, np.pi + 0.3)
    ax1.axhline(-np.pi, color="gray", linewidth=0.5, linestyle=":")
    ax1.axhline( np.pi, color="gray", linewidth=0.5, linestyle=":")
    ax1.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax1.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    ref_phase = build_reference_gait_phase(ref_df=ref_df, time_axis=time_axis, offset=offset, is_left=is_left)
    rmse, rmse_spike_removed, valid_mask = compute_metrics(ref_phase, calc_phase)

    # ── Panel 2: Phase signals ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1])
    ax3.plot(time_axis, ref_phase,  label="Reference (offset)", color="#2563EB", linewidth=1.4, zorder=3)
    ax3.plot(time_axis, calc_phase, label="Calculated",               color="#DC2626", linewidth=1.0,
             linestyle="--", alpha=0.85, zorder=2)
    offset_string = f"Ref Offset = {offset:.2f}\nNumber of cycles = {len(ref_df)}"
    ax3.text(
        1.01, 0.5, offset_string,
        transform=ax3.transAxes,
        fontsize=10, verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F3F4F6", edgecolor="#9CA3AF", linewidth=1.2),
    )
    ax3.set_ylabel("Gait Phase (rad)", fontsize=10)
    ax3.set_xlabel("Time (s)", fontsize=10)
    ax3.set_title("Reference (offset) vs. Calculated Gait Phase", fontsize=11)
    ax3.set_ylim(-np.pi - 0.3, np.pi + 0.3)
    ax3.axhline(-np.pi, color="gray", linewidth=0.5, linestyle=":")
    ax3.axhline( np.pi, color="gray", linewidth=0.5, linestyle=":")
    ax3.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax3.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ── Panel 3: Error ──────────────────────────────────────────────────────
    error = ref_phase - calc_phase
    ax2 = fig.add_subplot(gs[2], sharex=ax1)
    ax2.fill_between(time_axis, error, 0,
                     where=valid_mask, color="#7C3AED", alpha=0.35, label="Error (ref − calc)")  # type: ignore
    ax2.plot(time_axis[valid_mask], error[valid_mask],
             color="#7C3AED", linewidth=0.9, alpha=0.8)
    ax2.axhline(0, color="black", linewidth=0.8)

    # Metrics text box
    metrics_text = (
        f"RMSE = {rmse:.4f} rad"
        f"\nRMSE (spike removed) = {rmse_spike_removed:.4f} rad"
    )
    ax2.text(
        1.01, 0.5, metrics_text,
        transform=ax2.transAxes,
        fontsize=10, verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F3F4F6", edgecolor="#9CA3AF", linewidth=1.2),
    )

    ax2.set_ylabel("Error (rad)", fontsize=10)
    ax2.set_xlabel("Time (s)", fontsize=10)
    ax2.set_title("Phase Error", fontsize=11)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.subplots_adjust(right=0.82)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved → {save_path}")
    else:
        plt.show()

    plt.close(fig)


def compare_gait_phases(scenario: str, reference_path: Path, calculated_path: Path, save_path: str | None = None, plot_flag: bool = True):
    logger.info(f"\nLoading reference CSV  : {reference_path}")
    logger.info(f"Loading calculated CSV : {calculated_path}")

    ref_df, calc_df, is_left = load_and_validate(reference_path, calculated_path)

    time_axis  = calc_df["time (s)"].to_numpy(dtype=float)
    calc_phase = calc_df["gait_phase_left (rad)"].to_numpy(dtype=float)

    logger.info(f"  Calculated CSV rows  : {len(time_axis)}")
    logger.info(f"  Reference gait cycles: {len(ref_df)}")

    offset_tmp = -0.5
    offset = 0.0
    least_rmse = np.inf
    rmse = np.inf
    valid_mask: np.ndarray = np.array([])

    while offset_tmp <= 0.5 and offset_tmp >= -0.5:
        offset_tmp += 0.01

        ref_phase = build_reference_gait_phase(ref_df=ref_df, time_axis=time_axis, offset=offset_tmp, is_left=is_left)

        rmse, rmse_spike_removed, valid_mask_tmp = compute_metrics(ref_phase, calc_phase)
        if rmse < least_rmse:
            offset = offset_tmp
            least_rmse = rmse
            valid_mask = valid_mask_tmp

    covered = valid_mask.sum()
    logger.info(f"\n  Time points inside a reference cycle: {covered} / {len(time_axis)}")
    logger.info(f"  RMSE : {least_rmse:.6f} rad")

    if plot_flag:
        plot_comparison(time_axis=time_axis,
                        ref_df=ref_df,
                        is_left=is_left,
                        scenario=scenario,
                        calc_phase=calc_phase,
                        save_path=save_path,
                        offset=offset)

    ref_phase = build_reference_gait_phase(ref_df=ref_df, time_axis=time_axis, offset=offset, is_left=is_left)
    rmse, rmse_spike_removed, valid_mask = compute_metrics(ref_phase, calc_phase)
    return EvaluationData(
        subject=int(reference_path.parent.stem),
        scenario=reference_path.stem,
        ref_offset=offset,
        num_cycles=len(ref_df),
        RMSE=least_rmse,
        RMSE_spike_rm=rmse_spike_removed
    )


# ── CLI entry point ────────────────────────────────────────────────────────────

def iter_normal_walk(reference_dir: Path, calculated_dir: Path):
    for ref in reference_dir.rglob("*.csv"):
        trial_num = str(ref.parent.stem)
        output_dir = calculated_dir / ref.resolve().parents[1].stem
        for cal in output_dir.rglob("*.csv"):
            if "AB" + trial_num in cal.stem:
                yield ref, cal, trial_num

def iter_incline_walk(reference_dir: Path, calculated_dir: Path):
    for ref in reference_dir.rglob("*.csv"):
        trial_num = str(ref.parent.stem)
        output_dir = calculated_dir / ref.resolve().parents[1].stem
        direction: str
        if "up" in ref.stem:
            direction = "up"
        elif "down" in ref.stem:
            direction = "down"
        else:
            raise ValueError
        for cal in output_dir.rglob("*.csv"):
            if "AB" + trial_num in cal.stem and direction in cal.stem:
                yield ref, cal, trial_num


def normal_walk_pipeline():
    root = Path(__file__).resolve().parents[1]
    reference_dir = root / "scripts/ground_truth_gait_parsing/normal_walk"
    calculated_dir = root / "scripts/evaluation_output/normal_walk"
    save_path = "scripts/normal_walk/"
    scenario = "Normal Walk"
    plot_folder = Path(save_path)
    plot_folder.mkdir(parents=True, exist_ok=True)
    for ref, cal, num in iter_normal_walk(reference_dir, calculated_dir):
        compare_gait_phases(scenario=scenario, reference_path=ref, calculated_path=cal, save_path=save_path+ref.stem+num)

def incline_walk_pipeline():
    root = Path(__file__).resolve().parents[1]
    reference_dir = root / "scripts/ground_truth_gait_parsing/incline_walk"
    calculated_dir = root / "scripts/evaluation_output/incline_walk"
    save_path = "scripts/incline_walk/"
    scenario = "Incline Walk"
    plot_folder = Path(save_path)
    plot_folder.mkdir(parents=True, exist_ok=True)
    for ref, cal, num in iter_incline_walk(reference_dir, calculated_dir):
        compare_gait_phases(scenario=scenario, reference_path=ref, calculated_path=cal, save_path=save_path+ref.stem+num)



def calculate_mean_rmse():
    trials: dict[str, list[EvaluationData]] = {scenario: [] for scenario in Scenarios}
    root = Path(__file__).resolve().parents[1]
    reference_dir = root / "scripts/ground_truth_gait_parsing/normal_walk"
    calculated_dir = root / "scripts/evaluation_output/normal_walk"
    save_path = "scripts/normal_walk/"
    scenario = "Normal Walk"
    plot_folder = Path(save_path)
    plot_folder.mkdir(parents=True, exist_ok=True)
    for ref, cal, num in iter_normal_walk(reference_dir, calculated_dir):
        evaluation_data = compare_gait_phases(scenario=scenario, reference_path=ref, calculated_path=cal, save_path=save_path+ref.stem+num, plot_flag=False)
        logger.info(evaluation_data)
        appended = False
        for s in Scenarios:
            if evaluation_data.scenario in s:
                trials[s].append(evaluation_data)
                appended = True
                break
        if not appended:
            logger.warning(f"Trial not appended: {ref}")
    root = Path(__file__).resolve().parents[1]
    reference_dir = root / "scripts/ground_truth_gait_parsing/incline_walk"
    calculated_dir = root / "scripts/evaluation_output/incline_walk"
    save_path = "scripts/incline_walk/"
    scenario = "Incline Walk"
    plot_folder = Path(save_path)
    plot_folder.mkdir(parents=True, exist_ok=True)
    for ref, cal, num in iter_incline_walk(reference_dir, calculated_dir):
        evaluation_data = compare_gait_phases(scenario=scenario, reference_path=ref, calculated_path=cal, save_path=save_path+ref.stem+num, plot_flag=False)
        logger.info(evaluation_data)
        appended = False
        for s in Scenarios:
            if evaluation_data.scenario in s:
                trials[s].append(evaluation_data)
                appended = True
                break
        if not appended:
            logger.warning(f"Trial not appended: {ref}")

    for scenario in trials.keys():
        data_list = trials[scenario]
        RMSE_mean = np.mean([data.RMSE for data in data_list])
        RMSE_spike_rm_mean = np.mean([data.RMSE_spike_rm for data in data_list])
        RMSE_variance = np.var([data.RMSE for data in data_list])
        RMSE_spike_rm_variance = np.var([data.RMSE_spike_rm for data in data_list])
        logger.info(f"Scenario: {scenario}")
        logger.info(f"  Mean RMSE: {RMSE_mean:.6f} rad")
        logger.info(f"  Mean RMSE (spike removed): {RMSE_spike_rm_mean:.6f} rad")
        logger.info(f"  RMSE Variance: {RMSE_variance:.6f} rad^2")
        logger.info(f"  RMSE (spike removed) Variance: {RMSE_spike_rm_variance:.6f} rad^2")


if __name__ == "__main__":
    calculate_mean_rmse()
