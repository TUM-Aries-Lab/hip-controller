"""
Plotting utilities for WalkOnController evaluation output.

Each function accepts a DataFrame (one loaded CSV) and an optional Axes or
Figure object so plots can be embedded in larger layouts or saved standalone.

Columns expected (as written by run_evaluation.py):
    time (s)
    angle_right (deg), angle_left (deg)
    angle_right (rad), angle_left (rad)
    filtered_angle_right (rad), filtered_angle_left (rad)
    filtered_velocity_right (rad/s), filtered_velocity_left (rad/s)
    gait_phase_right (rad), gait_phase_left (rad)
    motor_command_right (rad/s), motor_command_left (rad/s)
"""

import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from pathlib import Path
from loguru import logger

# ── colour palette ──────────────────────────────────────────────────────────
C_RIGHT  = "#E05C5C"   # red family  – right side
C_LEFT   = "#4A90D9"   # blue family – left side
C_CMD_R  = "#C0392B"   # motor command right (darker red)
C_CMD_L  = "#1A5276"   # motor command left  (darker blue)
C_FILT   = "#7DCEA0"   # filtered signal accent

CONTROLLER_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = CONTROLLER_ROOT / Path("scripts/evaluation_output")
PLOT_ROOT = CONTROLLER_ROOT / Path("scripts/evaluation_plots")
# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _save_or_show(fig: Figure, save_path: str | Path | None) -> None:
    """Save or show the figure and close it.

    :param Figure fig: Figure object to save or display.
    :param str | Path | None save_path: Output file path. If None, use plt.show().
    :return: None
    """
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _title(df: pd.DataFrame, suffix: str) -> str:
    """Build a figure title from source_file if available, including participant ID.

    :param pandas.DataFrame df: DataFrame with optional source_file column.
    :param str suffix: Supplemental text for the title.
    :return: The computed chart title.
    """
    if "source_file" in df.columns:
        name = df["source_file"].iloc[0]
        import re

        participant = None
        m = re.search(r"AB\d{2}", str(name))
        if m:
            participant = m.group(0)

        if participant:
            return f"Participant {participant} — {name}  |  {suffix}"
        return f"{name}  |  {suffix}"
    return suffix


# ---------------------------------------------------------------------------
# 1.  Gait phase over time  (left + right)
# ---------------------------------------------------------------------------
def plot_gait_phase(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot left and right gait phase over time.

    :param pandas.DataFrame df: Evaluation data with gait phase columns.
    :param str | Path | None save_path: Optional path to save the figure PNG.
    :returns: The created Figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    t = df["time (s)"]

    ax.plot(t, df["gait_phase_right (rad)"], color=C_RIGHT, lw=1.5, label="Right")
    ax.plot(t, df["gait_phase_left (rad)"],  color=C_LEFT,  lw=1.5, label="Left",  alpha=0.85)
    ax.axhline(math.pi, color="grey", lw=0.8, ls="--", label="π (mid-cycle)")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Gait phase (rad)")
    ax.set_title(_title(df, "Gait Phase over Time"))
    ax.legend(loc="upper right")
    ax.set_ylim(-math.pi, math.pi + 0.2)
    ax.set_yticks([-math.pi, -math.pi / 2, 0, math.pi / 2, math.pi],
                  ["-π", "-π/2", "0", "π/2", "π"])

    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 2.  Gait phase + motor command overlay
# ---------------------------------------------------------------------------
def plot_phase_and_command(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot gait phase with motor command over time on twin y-axes for each side.

    :param pandas.DataFrame df: Evaluation data with gait phase and motor command columns.
    :param str | Path | None save_path: Optional path to save the figure PNG.
    :return: The created Figure object.
    """
    fig, (ax_r, ax_l) = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)
    t = df["time (s)"]

    for ax, side, phase_col, cmd_col, c_phase, c_cmd in [
        (ax_r, "Right", "gait_phase_right (rad)", "motor_command_right (rad/s)", C_RIGHT, C_CMD_R),
        (ax_l, "Left",  "gait_phase_left (rad)",  "motor_command_left (rad/s)",  C_LEFT,  C_CMD_L),
    ]:
        ax2 = ax.twinx()

        ax.plot(t, df[phase_col], color=c_phase, lw=1.5, label="Gait phase")

        ax.set_ylabel("Gait phase (rad)", color=c_phase)
        ax.set_ylim(-math.pi, math.pi + 0.2)
        ax.set_yticks([-math.pi, 0, math.pi], ["-π", "0", "π"])
        ax.tick_params(axis="y", labelcolor=c_phase)

        ax2.plot(t, df[cmd_col], color="green", lw=1.6, alpha=0.85, label="Motor command")
        ax2.set_ylabel("Motor command (rad/s)", color="green")
        ax2.tick_params(axis="y", labelcolor="green")

        # combined legend
        lines  = ax.get_lines() + ax2.get_lines()
        labels = [str(l.get_label()) for l in lines]
        ax.legend(lines, labels, loc="upper right", fontsize=8)
        ax.set_title(f"{side} — Gait Phase vs Motor Command")

    ax_l.set_xlabel("Time (s)")
    fig.suptitle(_title(df, "Phase & Motor Command Timing"), fontsize=12, y=1.01)
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 3.  Raw vs filtered hip angle
# ---------------------------------------------------------------------------
def plot_angle_filtering(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot raw vs filtered hip angle for left and right sides.

    :param pandas.DataFrame df: Evaluation data with raw and filtered angle columns.
    :param str | Path | None save_path: Optional path to save the figure PNG.
    :returns: The created Figure object.
    """
    fig, (ax_r, ax_l) = plt.subplots(2, 1, figsize=(13, 6), sharex=True, constrained_layout=True)
    t = df["time (s)"]

    for ax, side, raw_col, filt_col, c in [
        (ax_r, "Right", "angle_right (rad)", "filtered_angle_right (rad)", C_RIGHT),
        (ax_l, "Left",  "angle_left (rad)",  "filtered_angle_left (rad)",  C_LEFT),
    ]:
        ax.plot(t, df[raw_col],  color=c,      lw=1.0, alpha=0.45, label="Raw")
        ax.plot(t, df[filt_col], color=C_FILT, lw=1.5, label="Filtered")
        ax.set_ylabel("Hip angle (rad)")
        ax.set_title(f"{side} Hip Angle — Raw vs Filtered")
        ax.legend(loc="upper right", fontsize=8)

    ax_l.set_xlabel("Time (s)")
    fig.suptitle(_title(df, "Raw vs Filtered Hip Angle"), fontsize=12, y=1.01)
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 4.  Full overview dashboard  (all signals, one figure)
# ---------------------------------------------------------------------------
def plot_overview(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> Figure:
    """Create a multi-panel overview for all key signals.

    :param pandas.DataFrame df: Evaluation data including angles, phase, velocity, and commands.
    :param str | Path | None save_path: Optional path to save the figure PNG.
    :return: The created Figure object.
    """
    t = df["time (s)"]
    fig = plt.figure(figsize=(14, 14), constrained_layout=True)
    gs  = gridspec.GridSpec(4, 1, hspace=0.45, figure=fig)

    # ── Row 1: raw angle (degrees) ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, df["angle_right (deg)"], color=C_RIGHT, lw=1.2, label="Right")
    ax1.plot(t, df["angle_left (deg)"],  color=C_LEFT,  lw=1.2, label="Left", alpha=0.8)
    ax1.set_ylabel("Hip angle (°)")
    ax1.set_title("Hip Flexion Angle (raw)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.axhline(0, color="lightgrey", lw=0.6)

    # ── Row 2: filtered angle + velocity ───────────────────────────────────
    ax2  = fig.add_subplot(gs[1])
    ax2v = ax2.twinx()
    ax2.plot(t, df["filtered_angle_right (rad)"], color=C_RIGHT, lw=1.2, label="Filt angle R")
    ax2.plot(t, df["filtered_angle_left (rad)"],  color=C_LEFT,  lw=1.2, label="Filt angle L", alpha=0.8)
    ax2v.plot(t, df["filtered_velocity_right (rad/s)"], color=C_RIGHT, lw=0.9, ls=":", alpha=0.6, label="Velocity R")
    ax2v.plot(t, df["filtered_velocity_left (rad/s)"],  color=C_LEFT,  lw=0.9, ls=":", alpha=0.6, label="Velocity L")
    ax2.set_ylabel("Filtered angle (rad)")
    ax2v.set_ylabel("Velocity (rad/s)", alpha=0.7)
    ax2.set_title("Filtered Angle & Angular Velocity")
    lines  = ax2.get_lines() + ax2v.get_lines()
    ax2.legend(lines, [str(l.get_label()) for l in lines], loc="upper right", fontsize=8)

    # ── Row 3: gait phase ──────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(t, df["gait_phase_right (rad)"], color=C_RIGHT, lw=1.4, label="Right")
    ax3.plot(t, df["gait_phase_left (rad)"],  color=C_LEFT,  lw=1.4, label="Left", alpha=0.85)
    ax3.axhline(0, color="grey", lw=0.8, ls="--", label="0")
    ax3.set_ylim(-math.pi, math.pi + 0.2)
    ax3.set_yticks([-math.pi, 0, math.pi], ["-π", "0", "π"])
    ax3.set_ylabel("Gait phase (rad)")
    ax3.set_title("Gait Phase")
    ax3.legend(loc="upper right", fontsize=8)

    # ── Row 4: motor command ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3], sharex=ax3)
    ax4.plot(t, df["motor_command_right (rad/s)"], color=C_CMD_R, lw=1.4, label="Right")
    ax4.plot(t, df["motor_command_left (rad/s)"],  color=C_CMD_L, lw=1.4, label="Left", alpha=0.85)
    ax4.axhline(0, color="lightgrey", lw=0.6)
    ax4.set_ylabel("Motor command (rad/s)")
    ax4.set_xlabel("Time (s)")
    ax4.set_title("Motor Command")
    ax4.legend(loc="upper right", fontsize=8)

    fig.suptitle(_title(df, "Full Evaluation Overview"), fontsize=13, y=1.005)
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 5.  Batch: process entire evaluation_output folder
# ---------------------------------------------------------------------------
def plot_all(
    eval_root: str | Path,
    plot_root: str | Path,
    plots: tuple[str, ...] = ("overview", "phase", "phase_command", "filtering"),
) -> None:
    """Generate selected plots for every CSV in a folder tree.

    :param str | Path eval_root: Input folder containing evaluation CSV files.
    :param str | Path plot_root: Output folder for generated plot images.
    :param tuple[str, ...] plots: Which plot types to generate.
    :return: None
    """
    eval_root = Path(eval_root)
    plot_root = Path(plot_root)

    dispatch = {
        "overview":      plot_overview,
        "phase":         plot_gait_phase,
        "phase_command": plot_phase_and_command,
        "filtering":     plot_angle_filtering,
    }

    csv_files = sorted(eval_root.rglob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files — generating plots…\n")

    for csv_path in csv_files:
        df  = pd.read_csv(csv_path)
        rel = csv_path.relative_to(eval_root).with_suffix("")
        out_dir = plot_root / rel.parent / rel.name
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"  {csv_path.relative_to(eval_root)}")
        for name in plots:
            fn = dispatch.get(name)
            if fn is None:
                logger.warning(f"    Unknown plot type '{name}' — skipping.")
                continue
            fn(df, save_path=out_dir / f"{name}.png")


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Generate all 4 plot types for every CSV
    plot_all(
        eval_root=OUTPUT_ROOT,
        plot_root=PLOT_ROOT,
        plots=("overview", "phase", "phase_command", "filtering"),
    )

    # Or load one file and call individually
    # df = pd.read_csv("evaluation_output/stairs_combined/AB01_stairs_combined.csv")
    # plot_overview(df)
    # plot_gait_phase(df, save_path="AB01_phase.png")
    # plot_phase_and_command(df)
