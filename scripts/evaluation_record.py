"""
Evaluation pipeline for WalkOnController.

Reads all CSVs from evaluation_raw_data/, processes them through
left and right WalkOnControllers, and writes results to evaluation_data/
mirroring the original subfolder structure.

Assumed signal constructor:  Signal(angle_rad=<float>)
Adjust the `_make_signal` helper below if your Signal class differs.
"""

import math
import pandas as pd
from pathlib import Path

# ── ADJUST THESE IMPORTS TO MATCH YOUR PROJECT ─────────────────────────────
from hip_controller.control.app import WalkOnController   # your controller class
from hip_controller.definitions import SensorSignal
from loguru import logger
# ───────────────────────────────────────────────────────────────────────────

INPUT_ROOT  = Path("/home/minz/thesisproject/hip-controller/data/evaluation_raw_data")
OUTPUT_ROOT = Path("/home/minz/thesisproject/hip-controller/scripts/evaluation_data")

KEEP_COLS   = ["time", "hip_flexion_r", "hip_flexion_l"]  # degrees
RAW_HZ      = 200
TARGET_HZ   = 100
DOWNSAMPLE  = RAW_HZ // TARGET_HZ   # keep every 2nd row



# Step 1 – load & filter columns
def load_and_filter(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = (["source_file"] if "source_file" in df.columns else []) + KEEP_COLS
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing required columns {missing}")
    return df[cols].reset_index(drop=True)


# Step 2 – unit conversion + downsampling
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hip_flexion_r_rad"] = df["hip_flexion_r"].apply(math.radians)
    df["hip_flexion_l_rad"] = df["hip_flexion_l"].apply(math.radians)
    df = df.iloc[::DOWNSAMPLE].reset_index(drop=True)   # 200 Hz → 100 Hz
    return df



# Step 3 – run controllers row-by-row

def run_controllers(df: pd.DataFrame,
                    ctrl_left: WalkOnController,
                    ctrl_right: WalkOnController) -> pd.DataFrame:
    records = []

    for i in range(0, len(df)):
        row = df.iloc[i]

        timestamp = float(row["time"])
        signal_r = SensorSignal(timestamp=timestamp, angle_rad=float(row["hip_flexion_r_rad"]),velocity_rad_per_sec=0.0)
        signal_l = SensorSignal(timestamp=timestamp, angle_rad=float(row["hip_flexion_l_rad"]),velocity_rad_per_sec=0.0)


        # ── right side ──
        filt_r   = ctrl_right.pre_processor.filter(raw_signal=signal_r)
        phase_r  = ctrl_right.gait_controller.update_and_compute(curr_signal=filt_r)
        amplitude_r    = ctrl_right.amplitude_modulation.compute_amplitude(signal=signal_r)
        command_r    = ctrl_right.motor_reference_controller.compute_motor_command(
                       gait_phase=phase_r, amplitude=amplitude_r)

        # ── left side ──
        filt_l   = ctrl_left.pre_processor.filter(raw_signal=signal_l)
        phase_l  = ctrl_left.gait_controller.update_and_compute(curr_signal=filt_l)
        amplitude_l    = ctrl_left.amplitude_modulation.compute_amplitude(signal=signal_l)
        command_l    = ctrl_left.motor_reference_controller.compute_motor_command(
                       gait_phase=phase_l, amplitude=amplitude_l)

        records.append({
            "time (s)":                     row["time"],
            "angle_right (deg)":            row["hip_flexion_r"],
            "angle_left (deg)":             row["hip_flexion_l"],
            "angle_right (rad)":            row["hip_flexion_r_rad"],
            "angle_left (rad)":             row["hip_flexion_l_rad"],
            "filtered_angle_right (rad)":   filt_r.angle_rad,
            "filtered_angle_left (rad)":    filt_l.angle_rad,
            "filtered_velocity_right (rad/s)": filt_r.velocity_rad_per_sec,
            "filtered_velocity_left (rad/s)":  filt_l.velocity_rad_per_sec,
            "gait_phase_right (rad)":       phase_r,
            "gait_phase_left (rad)":        phase_l,
            "motor_command_right (rad/s)":  command_r,
            "motor_command_left (rad/s)":   command_l,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Step 4 – process a single CSV
# ---------------------------------------------------------------------------
def process_file(input_path: Path,
                 output_path: Path,
                 ctrl_left: WalkOnController,
                 ctrl_right: WalkOnController) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_raw  = load_and_filter(input_path)
    df_prep = prepare(df_raw)

    #df_out  = run_controllers(df_prep, ctrl_left, ctrl_right)

    df_prep.to_csv(output_path, index=False)
    logger.info(f"  ✓ {input_path.relative_to(INPUT_ROOT)}  →  {output_path.relative_to(OUTPUT_ROOT)}  ({len(df_prep)} rows)")


# ---------------------------------------------------------------------------
# Step 5 – discover all CSVs and batch-process
# ---------------------------------------------------------------------------
def run_evaluation(ctrl_left: WalkOnController,
                   ctrl_right: WalkOnController) -> None:
    csv_files = sorted(INPUT_ROOT.rglob("*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found under {INPUT_ROOT}/")
        return

    logger.info(f"Found {len(csv_files)} CSV files — processing…\n")

    for input_path in csv_files:
        # Mirror subfolder structure: evaluation_raw_data/a/b.csv → evaluation_data/a/b.csv
        relative   = input_path.relative_to(INPUT_ROOT)
        output_path = OUTPUT_ROOT / relative
        try:
            process_file(input_path, output_path, ctrl_left, ctrl_right)
        except Exception as exc:
            logger.warning(f"  ✗ {input_path.name}: {exc}")

    logger.info(f"\nDone. Results written to {OUTPUT_ROOT}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ── Instantiate your controllers here ───────────────────────────────────
    controller_left  = WalkOnController(reverse=True, filtered=False)   # add constructor args as needed
    controller_right = WalkOnController(reverse=True, filtered=False)
    # ────────────────────────────────────────────────────────────────────────

    run_evaluation(controller_left, controller_right)
