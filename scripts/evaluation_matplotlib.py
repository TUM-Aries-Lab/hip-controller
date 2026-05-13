"""
Plotting utilities for WalkOnController evaluation output.

Columns expected (as written by run_evaluation.py):
    time (s)
    angle_right (deg), angle_left (deg)
    angle_right (rad), angle_left (rad)
    filtered_angle_right (rad), filtered_angle_left (rad)
    filtered_velocity_right (rad/s), filtered_velocity_left (rad/s)
    gait_phase_right (rad), gait_phase_left (rad)
    amplitude_right, amplitude_left
    motor_command_right (rad/s), motor_command_left (rad/s)
"""

import math
import re
from enum import StrEnum
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from loguru import logger
from typing import Callable

CONTROLLER_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = CONTROLLER_ROOT / Path("scripts/evaluation_output")
PLOT_ROOT   = CONTROLLER_ROOT / Path("scripts/evaluation_plots")


# ============================================================================
# StrEnum – single source of truth for all column names, labels, and titles
# ============================================================================

class Col(StrEnum):
    """CSV column names. Change here if the pipeline output columns are renamed."""
    TIME            = "time (s)"
    ANGLE_R_RAD     = "angle_right (rad)"
    ANGLE_L_RAD     = "angle_left (rad)"
    FILT_ANGLE_R    = "filtered_angle_right (rad)"
    FILT_ANGLE_L    = "filtered_angle_left (rad)"
    FILT_VEL_R      = "filtered_velocity_right (rad/s)"
    FILT_VEL_L      = "filtered_velocity_left (rad/s)"
    PHASE_R         = "gait_phase_right (rad)"
    PHASE_L         = "gait_phase_left (rad)"
    AMPLITUDE_R     = "amplitude_right"
    AMPLITUDE_L     = "amplitude_left"
    CMD_R           = "motor_command_right (rad/s)"
    CMD_L           = "motor_command_left (rad/s)"
    SOURCE_FILE     = "source_file"


class Label(StrEnum):
    """Axis labels and legend entries."""
    TIME            = "Time (s)"
    ANGLE_RAD       = "Hip angle (rad)"
    FILT_ANGLE      = "Filtered angle (rad)"
    VELOCITY        = "Filtered velocity (rad/s)"
    PHASE           = "Gait phase (rad)"
    AMPLITUDE       = "Amplitude"
    MOTOR_CMD       = "Motor command (rad/s)"
    RIGHT           = "Right"
    LEFT            = "Left"
    RAW             = "Raw"
    FILTERED        = "Filtered"


class Title(StrEnum):
    """Figure supertitles and panel titles."""
    GAIT_PHASE          = "Gait Phase over Time"
    PHASE_CMD           = "Phase & Motor Command Timing"
    FILTERING           = "Raw vs Filtered Hip Angle"
    OVERVIEW            = "Full Evaluation Overview"
    PHASE_AMP_CMD       = "Gait Phase, Amplitude & Motor Command"
    PANEL_PHASE_R       = "Right — Gait Phase"
    PANEL_PHASE_L       = "Left — Gait Phase"
    PANEL_AMP_R         = "Right — Amplitude"
    PANEL_AMP_L         = "Left — Amplitude"
    PANEL_CMD_R         = "Right — Motor Command"
    PANEL_CMD_L         = "Left — Motor Command"
    PANEL_RAW_R         = "Right — Raw Angle"
    PANEL_RAW_L         = "Left — Raw Angle"
    PANEL_FILT_ANG_R        = "Right — Filtered Angle"
    PANEL_FILT_ANG_L        = "Left — Filtered Angle"
    PANEL_FILT_VEL_R        = "Right — Filtered Velocity"
    PANEL_FILT_VEL_L        = "Left — Filtered Velocity"
    PANEL_PHASE_CMD_R   = "Right — Gait Phase vs Motor Command"
    PANEL_PHASE_CMD_L   = "Left — Gait Phase vs Motor Command"


class Marker(StrEnum):
    """Reference line tick labels."""
    ZERO        = "0"
    PI          = "π"
    NEG_PI      = "−π"
    HALF_PI     = "π/2"
    NEG_HALF_PI = "−π/2"


# ── Colour palette ───────────────────────────────────────────────────────────
C_RIGHT     = "#E05C5C"
C_LEFT      = "#4A90D9"
C_CMD_R     = "#C0392B"
C_CMD_L     = "#1A5276"
C_FILT      = "#7DCEA0"
C_AMP_R     = "#E67E22"
C_AMP_L     = "#8E44AD"
C_REF       = "grey"
C_REF_LIGHT = "lightgrey"

PI_TICKS  = [-math.pi, -math.pi / 2, 0, math.pi / 2, math.pi]
PI_LABELS = [Marker.NEG_PI, Marker.NEG_HALF_PI, Marker.ZERO, Marker.HALF_PI, Marker.PI]


# ============================================================================
# Helpers
# ============================================================================

def _save_or_show(fig: Figure, save_path: str | Path | None) -> None:
    """
    Save the figure to a file or display it interactively.

    :param Figure fig: The matplotlib figure to save or show.
    :param str | Path | None save_path: Path to save the figure; if None, display interactively.
    """
    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info(f"Saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _build_title(df: pd.DataFrame, suffix: str) -> str:
    """
    Build a title string for the plot including participant info if available.

    :param pd.DataFrame df: The dataframe containing source file info.
    :param str suffix: The suffix to append to the title.
    :return: The constructed title string.
    :rtype: str
    """
    if Col.SOURCE_FILE in df.columns:
        name = df[Col.SOURCE_FILE].iloc[0]
        m = re.search(r"AB\d{2}", str(name))
        participant = f"Participant {m.group(0)} — " if m else ""
        return f"{participant}{name}  |  {suffix}"
    return suffix


def _phase_yticks(ax) -> None:
    """
    Set y-axis ticks and limits for gait phase plots.

    :param ax: The matplotlib axis to configure.
    """
    ax.set_yticks(PI_TICKS, PI_LABELS)
    ax.set_ylim(-math.pi - 0.15, math.pi + 0.15)


def _sci_yaxis(ax) -> None:
    """
    Set y-axis to scientific notation.

    :param ax: The matplotlib axis to configure.
    """
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))


def _load(path: str | Path) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    :param str | Path path: Path to the CSV file.
    :return: The loaded DataFrame.
    :rtype: pd.DataFrame
    """
    return pd.read_csv(Path(path))


# ============================================================================
# 1. Gait phase over time  (Right | Left panels)
# ============================================================================

def plot_gait_phase(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot left and right gait phase over time on separate panels.

    :param pandas.DataFrame df: Evaluation data with gait phase columns.
    :param str | Path | None save_path: Optional path to save the figure PNG.
    :returns: The created Figure object.
    """
    t = df[Col.TIME]
    fig, (ax_r, ax_l) = plt.subplots(1, 2, figsize=(13, 4), sharey=True, constrained_layout=True)

    for ax, col, color, title in [
        (ax_r, Col.PHASE_R, C_RIGHT, Title.PANEL_PHASE_R),
        (ax_l, Col.PHASE_L, C_LEFT,  Title.PANEL_PHASE_L),
    ]:
        ax.plot(t, df[col], color=color, lw=0.8)
        ax.axhline(0,         color=C_REF,       lw=0.7, ls="--")
        ax.axhline( math.pi,  color=C_REF_LIGHT, lw=0.6, ls=":")
        ax.axhline(-math.pi,  color=C_REF_LIGHT, lw=0.6, ls=":")
        _phase_yticks(ax)
        ax.set_xlabel(Label.TIME)
        ax.set_title(title)

    ax_r.set_ylabel(Label.PHASE)
    fig.suptitle(_build_title(df, Title.GAIT_PHASE), fontsize=11)
    _save_or_show(fig, save_path)
    return fig


# ============================================================================
# 2. Gait phase + motor command  (2×2: phase top, command bottom)
# ============================================================================

def plot_phase_and_command(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> Figure:
    """2×2 grid: gait phase (top row) and motor command (bottom row) per side.

    :param pandas.DataFrame df: Evaluation data with gait phase and motor command columns.
    :param str | Path | None save_path: Optional path to save the figure PNG.
    :return: The created Figure object.
    """
    t = df[Col.TIME]
    fig = plt.figure(figsize=(14, 7))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)

    sides = [
        (Title.PANEL_PHASE_CMD_R, Title.PANEL_CMD_R, Col.PHASE_R, Col.CMD_R, C_RIGHT, C_CMD_R, 0),
        (Title.PANEL_PHASE_CMD_L, Title.PANEL_CMD_L, Col.PHASE_L, Col.CMD_L, C_LEFT,  C_CMD_L, 1),
    ]

    for ph_title, cmd_title, phase_col, cmd_col, c_ph, c_cmd, col_idx in sides:
        ax_ph = fig.add_subplot(gs[0, col_idx])
        ax_ph.plot(t, df[phase_col], color=c_ph, lw=0.8)
        ax_ph.axhline(0, color=C_REF, lw=0.7, ls="--")
        _phase_yticks(ax_ph)
        ax_ph.set_title(ph_title)
        ax_ph.set_ylabel(Label.PHASE)
        ax_ph.set_xlabel(Label.TIME)

        ax_cmd = fig.add_subplot(gs[1, col_idx], sharex=ax_ph)
        ax_cmd.plot(t, df[cmd_col], color=c_cmd, lw=0.9)
        ax_cmd.axhline(0, color=C_REF_LIGHT, lw=0.7)
        _sci_yaxis(ax_cmd)
        ax_cmd.set_title(cmd_title)
        ax_cmd.set_ylabel(Label.MOTOR_CMD)
        ax_cmd.set_xlabel(Label.TIME)

    fig.suptitle(_build_title(df, Title.PHASE_CMD), fontsize=11)
    _save_or_show(fig, save_path)
    return fig


# ============================================================================
# 3. Raw vs filtered angle  (2×2: raw top, filtered bottom)
# ============================================================================

def plot_angle_filtering(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> Figure:
    """2×2 grid: raw angle (top row) and filtered angle (bottom row) per side.

    :param pandas.DataFrame df: Evaluation data with raw and filtered angle columns.
    :param str | Path | None save_path: Optional path to save the figure PNG.
    :returns: The created Figure object.
    """
    t = df[Col.TIME]
    fig = plt.figure(figsize=(14, 7))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)

    sides = [
        (Title.PANEL_RAW_R, Title.PANEL_FILT_ANG_R, Col.ANGLE_R_RAD, Col.FILT_ANGLE_R, C_RIGHT, 0),
        (Title.PANEL_RAW_L, Title.PANEL_FILT_ANG_L, Col.ANGLE_L_RAD, Col.FILT_ANGLE_L, C_LEFT,  1),
    ]

    for raw_title, filt_title, raw_col, filt_col, c, col_idx in sides:
        ax_raw = fig.add_subplot(gs[0, col_idx])
        ax_raw.plot(t, df[raw_col], color=c, lw=0.8)
        ax_raw.set_title(raw_title)
        ax_raw.set_ylabel(Label.ANGLE_RAD)
        ax_raw.set_xlabel(Label.TIME)

        ax_filt = fig.add_subplot(gs[1, col_idx], sharex=ax_raw)
        ax_filt.plot(t, df[filt_col], color=C_FILT, lw=0.8)
        ax_filt.set_title(filt_title)
        ax_filt.set_ylabel(Label.FILT_ANGLE)
        ax_filt.set_xlabel(Label.TIME)

    fig.suptitle(_build_title(df, Title.FILTERING), fontsize=11)
    _save_or_show(fig, save_path)
    return fig


# ============================================================================
# 4. Gait phase + amplitude + motor command  (3×2 grid)
# ============================================================================

def plot_phase_amplitude_command(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> Figure:
    """3×2 grid: gait phase / amplitude / motor command for Right and Left.

    Rows (top → bottom): Gait Phase | Amplitude | Motor Command
    Columns (left → right): Right | Left
    All panels share the time x-axis so phase, amplitude, and command are
    vertically aligned for easy timing verification.

    :param pandas.DataFrame df: Evaluation data; must contain amplitude columns.
    :param str | Path | None save_path: Optional path to save the figure PNG.
    :return: The created Figure object.
    """
    t = df[Col.TIME]
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.28)

    sides = [
        (Label.RIGHT, Col.PHASE_R, Col.AMPLITUDE_R, Col.CMD_R,
         C_RIGHT, C_AMP_R, C_CMD_R,
         Title.PANEL_PHASE_R, Title.PANEL_AMP_R, Title.PANEL_CMD_R, 0),
        (Label.LEFT,  Col.PHASE_L, Col.AMPLITUDE_L, Col.CMD_L,
         C_LEFT,  C_AMP_L, C_CMD_L,
         Title.PANEL_PHASE_L, Title.PANEL_AMP_L, Title.PANEL_CMD_L, 1),
    ]

    for (side, phase_col, amp_col, cmd_col,
         c_ph, c_amp, c_cmd,
         ph_title, amp_title, cmd_title, col_idx) in sides:

        # ── Row 0: gait phase ──────────────────────────────────────────────
        ax_ph = fig.add_subplot(gs[0, col_idx])
        ax_ph.plot(t, df[phase_col], color=c_ph, lw=0.9)
        ax_ph.axhline(0, color=C_REF, lw=0.7, ls="--")
        _phase_yticks(ax_ph)
        ax_ph.set_title(ph_title)
        ax_ph.set_ylabel(Label.PHASE)

        # ── Row 1: amplitude ───────────────────────────────────────────────
        ax_amp = fig.add_subplot(gs[1, col_idx], sharex=ax_ph)
        if amp_col in df.columns:
            ax_amp.plot(t, df[amp_col], color=c_amp, lw=0.9)
            ax_amp.axhline(0, color=C_REF_LIGHT, lw=0.6)
        else:
            ax_amp.text(0.5, 0.5, f"Column '{amp_col}' not found",
                        ha="center", va="center", transform=ax_amp.transAxes,
                        color=C_REF, fontsize=9)
        ax_amp.set_title(amp_title)
        ax_amp.set_ylabel(Label.AMPLITUDE)

        # ── Row 2: motor command ───────────────────────────────────────────
        ax_cmd = fig.add_subplot(gs[2, col_idx], sharex=ax_ph)
        ax_cmd.plot(t, df[cmd_col], color=c_cmd, lw=0.9)
        ax_cmd.axhline(0, color=C_REF_LIGHT, lw=0.7)
        _sci_yaxis(ax_cmd)
        ax_cmd.set_title(cmd_title)
        ax_cmd.set_ylabel(Label.MOTOR_CMD)
        ax_cmd.set_xlabel(Label.TIME)

    fig.suptitle(_build_title(df, Title.PHASE_AMP_CMD), fontsize=11)
    _save_or_show(fig, save_path)
    return fig


# ============================================================================
# 5. Full overview dashboard  (4×2)
# ============================================================================

def plot_overview(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> Figure:
    """4x2 dashboard: raw angle | filtered angle | gait phase | motor command.

    :param pandas.DataFrame df: Evaluation data including angles, phase, velocity, and commands.
    :param str | Path | None save_path: Optional path to save the figure PNG.
    :return: The created Figure object.
    """
    t = df[Col.TIME]
    fig = plt.figure(figsize=(14, 14))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.28)

    sides = [
        (Label.RIGHT, C_RIGHT, C_FILT, C_CMD_R,
         Col.ANGLE_R_RAD, Col.FILT_VEL_R, Col.PHASE_R, Col.CMD_R,
         Title.PANEL_RAW_R, Title.PANEL_FILT_VEL_R, Title.PANEL_PHASE_R, Title.PANEL_CMD_R, 0),
        (Label.LEFT,  C_LEFT,  C_FILT, C_CMD_L,
         Col.ANGLE_L_RAD, Col.FILT_VEL_L, Col.PHASE_L, Col.CMD_L,
         Title.PANEL_RAW_L, Title.PANEL_FILT_VEL_L, Title.PANEL_PHASE_L, Title.PANEL_CMD_L, 1),
    ]

    for (side, c, c_f, c_cmd,
         raw_col, filt_col, ph_col, cmd_col,
         raw_title, filt_title, ph_title, cmd_title, col_idx) in sides:

        ax0 = fig.add_subplot(gs[0, col_idx])
        ax0.plot(t, df[raw_col], color=c, lw=0.8)
        ax0.set_title(raw_title)
        ax0.set_ylabel(Label.ANGLE_RAD)

        ax1 = fig.add_subplot(gs[1, col_idx], sharex=ax0)
        ax1.plot(t, df[filt_col], color=c_f, lw=0.8)
        ax1.set_title(filt_title)
        ax1.set_ylabel(Label.FILT_ANGLE)

        ax2 = fig.add_subplot(gs[2, col_idx], sharex=ax0)
        ax2.plot(t, df[ph_col], color=c, lw=0.8)
        ax2.axhline(0, color=C_REF, lw=0.7, ls="--")
        _phase_yticks(ax2)
        ax2.set_title(ph_title)
        ax2.set_ylabel(Label.PHASE)

        ax3 = fig.add_subplot(gs[3, col_idx], sharex=ax0)
        ax3.plot(t, df[cmd_col], color=c_cmd, lw=0.9)
        ax3.axhline(0, color=C_REF_LIGHT, lw=0.7)
        _sci_yaxis(ax3)
        ax3.set_title(cmd_title)
        ax3.set_ylabel(Label.MOTOR_CMD)
        ax3.set_xlabel(Label.TIME)

    fig.suptitle(_build_title(df, Title.OVERVIEW), fontsize=12)
    _save_or_show(fig, save_path)
    return fig


# ============================================================================
# Dispatch table
# ============================================================================

PLOT_DISPATCH: dict[str, Callable] = {
    "overview":          plot_overview,
    "phase":             plot_gait_phase,
    "phase_command":     plot_phase_and_command,
    "filtering":         plot_angle_filtering,
    "phase_amp_command": plot_phase_amplitude_command,
}


# ============================================================================
# Single-file entry point
# ============================================================================

def plot_file(
    file_path: str | Path,
    plots: tuple[str, ...] = ("overview", "phase", "phase_command", "filtering", "phase_amp_command"),
    save_dir: str | Path | None = None,
) -> None:
    """Generate selected plots for a single CSV file.

    :param str | Path file_path: Path to the evaluation CSV.
    :param tuple[str, ...] plots: Plot types to generate. Options:
        'overview', 'phase', 'phase_command', 'filtering', 'phase_amp_command'.
    :param str | Path | None save_dir: Folder to save PNGs into.
        If None, each plot is displayed interactively instead of saved.
    :return: None
    """
    file_path = Path(file_path)
    df = _load(file_path)

    out_dir = Path(save_dir) if save_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Plotting {file_path.name}  [{', '.join(plots)}]")
    for name in plots:
        fn = PLOT_DISPATCH.get(name)
        if fn is None:
            logger.warning(f"Unknown plot type '{name}' — skipping.")
            continue
        save_path = (out_dir / f"{file_path.stem}_{name}.png") if out_dir else None
        fn(df, save_path=save_path)


# ============================================================================
# Batch entry point
# ============================================================================

def plot_all(
    eval_root: str | Path = OUTPUT_ROOT,
    plot_root: str | Path = PLOT_ROOT,
    plots: tuple[str, ...] = ("overview", "phase", "phase_command", "filtering", "phase_amp_command"),
) -> None:
    """Generate selected plots for every CSV under eval_root.

    :param str | Path eval_root: Root folder of evaluation CSVs.
    :param str | Path plot_root: Root folder where PNGs will be saved,
        mirroring the subfolder structure of eval_root.
    :param tuple[str, ...] plots: Plot types to generate per file.
    :return: None
    """
    eval_root = Path(eval_root)
    plot_root = Path(plot_root)

    csv_files = sorted(eval_root.rglob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files — generating plots…")

    for csv_path in csv_files:
        rel     = csv_path.relative_to(eval_root).with_suffix("")
        out_dir = plot_root / rel.parent / rel.name
        plot_file(csv_path, plots=plots, save_dir=out_dir)


# ============================================================================
# Example
# ============================================================================

if __name__ == "__main__":
    # ── Single file (interactive, no save_dir) ───────────────────────────────
    # plot_file("scripts/evaluation_output/incline_walk_combined/AB10_incline_walk_5_combined.csv")

    # ── Single file (save to folder) ─────────────────────────────────────────
    # plot_file(
    #    file_path="scripts/evaluation_output/incline_walk_combined/AB10_incline_walk_5_combined.csv",
    #    plots=("phase_amp_command", "phase_command"),
    #    save_dir="scripts/evaluation_plots/single",
    #)

    # ── Batch all files ───────────────────────────────────────────────────────
    plot_all(plots=("phase_amp_command","overview"))
