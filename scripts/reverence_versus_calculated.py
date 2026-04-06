import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import sys
from loguru import logger

def load_and_validate(reference_path: str, calculated_path: str):
    """Load and validate both CSV files."""
    ref_df = pd.read_csv(reference_path)
    calc_df = pd.read_csv(calculated_path)

    required_ref = {"left_col1", "left_col2"}
    required_calc = {"time (s)", "gait_phase_left (rad)"}

    if not required_ref.issubset(ref_df.columns):
        raise ValueError(f"Reference CSV must contain columns: {required_ref}. Found: {set(ref_df.columns)}")
    if not required_calc.issubset(calc_df.columns):
        raise ValueError(f"Calculated CSV must contain columns: {required_calc}. Found: {set(calc_df.columns)}")

    return ref_df, calc_df


def build_reference_gait_phase(ref_df: pd.DataFrame, time_axis: np.ndarray) -> np.ndarray:
    """
    For every gait cycle defined by (left_col1, left_col2), linearly interpolate
    gait phase from -pi to pi over that interval.  Time points outside any cycle
    are set to NaN.
    """
    ref_phase = np.full_like(time_axis, np.nan, dtype=float)

    for _, row in ref_df.iterrows():
        t_start, t_end = float(row["left_col1"]), float(row["left_col2"])
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
    """Compute RMSE and MAPE ignoring NaN positions."""
    valid = ~(np.isnan(ref) | np.isnan(calc))
    r, c = ref[valid], calc[valid]

    rmse = np.sqrt(np.mean((r - c) ** 2))

    # MAPE – guard against zero reference values
    nonzero = np.abs(r) > 1e-9
    mape = np.mean(np.abs((r[nonzero] - c[nonzero]) / r[nonzero])) * 100 if nonzero.any() else np.nan

    return rmse, mape, valid


def plot_comparison(
    time_axis: np.ndarray,
    ref_phase: np.ndarray,
    calc_phase: np.ndarray,
    rmse: float,
    mape: float,
    valid_mask: np.ndarray,
    save_path: str | None = None,
):
    """Draw two-panel comparison figure."""
    error = ref_phase - calc_phase

    fig = plt.figure(figsize=(13, 7))
    fig.suptitle("Gait Phase Comparison – Left Leg", fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.45)

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

    # ── Panel 2: Error ──────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(time_axis, error, 0,
                     where=valid_mask, color="#7C3AED", alpha=0.35, label="Error (ref − calc)")
    ax2.plot(time_axis[valid_mask], error[valid_mask],
             color="#7C3AED", linewidth=0.9, alpha=0.8)
    ax2.axhline(0, color="black", linewidth=0.8)

    # Metrics text box
    metrics_text = (
        f"RMSE = {rmse:.4f} rad\n"
        f"MAPE = {mape:.2f} %"
        if not np.isnan(mape)
        else f"RMSE = {rmse:.4f} rad\nMAPE = N/A (ref≈0)"
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


def compare_gait_phases(reference_path: str, calculated_path: str, save_path: str | None = None):
    logger.info(f"\nLoading reference CSV  : {reference_path}")
    logger.info(f"Loading calculated CSV : {calculated_path}")

    ref_df, calc_df = load_and_validate(reference_path, calculated_path)

    time_axis  = calc_df["time (s)"].to_numpy(dtype=float)
    calc_phase = calc_df["gait_phase_left (rad)"].to_numpy(dtype=float)

    logger.info(f"  Calculated CSV rows  : {len(time_axis)}")
    logger.info(f"  Reference gait cycles: {len(ref_df)}")

    ref_phase = build_reference_gait_phase(ref_df, time_axis)

    rmse, mape, valid_mask = compute_metrics(ref_phase, calc_phase)

    covered = valid_mask.sum()
    logger.info(f"\n  Time points inside a reference cycle: {covered} / {len(time_axis)}")
    logger.info(f"  RMSE : {rmse:.6f} rad")
    if not np.isnan(mape):
        logger.info(f"  MAPE : {mape:.4f} %")
    else:
        logger.info("  MAPE : N/A (reference values too close to zero)")

    plot_comparison(time_axis, ref_phase, calc_phase, rmse, mape, valid_mask, save_path)


# ── CLI entry point ────────────────────────────────────────────────────────────


reference_csv = "scripts/ground_truth_gait_parsing/normal_walk_1_2/01/normal_walk_1_1-2_parsing.csv"
calculated_csv = "scripts/evaluation_output/normal_walk/normal_walk_1_2/AB01_normal_walk_1_1-2_angle.csv"
save_path = "scripts/test"

if __name__ == "__main__":
    compare_gait_phases(reference_path=reference_csv, calculated_path=calculated_csv, save_path=save_path)
