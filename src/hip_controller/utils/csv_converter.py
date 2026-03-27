"""Functions to deal with csv data.

Walking data CSV processing utilities.
Handles stair concatenation, file combining, and column filtering.


---------------------------------------------------------------------------
Example usage
---------------------------------------------------------------------------

    # 1. Concatenate all stair trials per participant
    concatenate_stairs(
        stairs_root="data/stairs",
        output_dir="output/stairs",
    )

    # 2. Combine incline walk up + down at slope 5 for participant AB01
    combine_two_files(
        file_a="data/AB01_incline_walk_up_5.csv",
        file_b="data/AB01_incline_walk_down_5.csv",
        output_path="output/AB01_incline_walk_slope5_combined.csv",
    )
"""

from pathlib import Path

import pandas as pd
from loguru import logger

KEEP_COLS = ["time", "hip_flexion_r", "hip_flexion_l"]


def filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only time, hip_flexion_r, hip_flexion_l columns."""
    available = [c for c in KEEP_COLS if c in df.columns]
    missing = set(KEEP_COLS) - set(available)
    if missing:
        logger.warning(f"  Warning: missing columns {missing}")
    return df.loc[:, available]


def concatenate_stairs(
    stairs_root: str | Path,
    output_dir: str | Path,
    participants: list[str] | None = None,
) -> None:
    """Concatenate all stair CSV files for each participant into one file.

    Scans <stairs_root>/<participant>/ for files matching
    AB##_stairs_1_*_(up|down)_angle.csv and merges them in
    numeric order into <output_dir>/<participant>_stairs_combined.csv.

    Args:
        stairs_root:  Path to the 'stairs' folder containing AB01-AB13 subdirs.
        output_dir:   Destination folder for the combined per-participant files.
        participants: Optional list of participant IDs (e.g. ['AB01', 'AB03']).
                      Defaults to all subdirectories found in stairs_root.

    """
    stairs_root = Path(stairs_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folders = (
        [stairs_root / p for p in participants]
        if participants
        else sorted(stairs_root.iterdir())
    )

    for folder in folders:
        if not folder.is_dir():
            continue
        participant = folder.name

        # Collect & sort by the trial number (the digit after 'stairs_1_')
        csv_files = sorted(
            folder.glob("*_stairs_1_*_angle.csv"),
            key=lambda p: int(p.stem.split("_stairs_1_")[1].split("_")[0]),
        )

        if not csv_files:
            logger.warning(f"[{participant}] No stair files found — skipping.")
            continue

        frames = []
        for f in csv_files:
            df = pd.read_csv(f)
            df = filter_columns(df)
            df.insert(0, "source_file", f.name)  # track origin
            frames.append(df)
            logger.info(f"  [{participant}] loaded {f.name}  ({len(df)} rows)")

        combined = pd.concat(frames, ignore_index=True)
        out_path = output_dir / f"{participant}_stairs_combined.csv"
        combined.to_csv(out_path, index=False)
        logger.info(
            f"[{participant}] → saved {out_path}  ({len(combined)} rows total)\n"
        )


def combine_two_files(
    file_a: str | Path,
    file_b: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Vertically concatenate two CSV files and keep only the required columns.

    Useful for pairing complementary conditions, e.g. incline_walk up + down
    at the same slope, or turn_left + turn_right.

    Args:
        file_a:      Path to the first CSV file.
        file_b:      Path to the second CSV file.
        output_path: Destination path for the merged CSV.

    Returns:
        The merged DataFrame.

    """
    file_a, file_b = Path(file_a), Path(file_b)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_a = filter_columns(pd.read_csv(file_a))
    df_b = filter_columns(pd.read_csv(file_b))

    df_a.insert(0, "source_file", file_a.name)
    df_b.insert(0, "source_file", file_b.name)

    combined = pd.concat([df_a, df_b], ignore_index=True)
    combined.to_csv(output_path, index=False)
    logger.info(
        f"Combined {file_a.name} + {file_b.name} → {output_path}  ({len(combined)} rows)"
    )
    return combined


def convert_xlsx_to_csv(
    input_xlsx_path: Path, output_csv_path: Path | None = None
) -> Path:
    """Convert an Excel file to CSV format.

    Reads a single Excel file (.xls or .xlsx) from the testing directory and writes
    its contents to a CSV file with the same filename stem in the same directory.
    This is useful for converting test data and measurement recordings to a
    more portable and scriptable format.

    :param Path path: Absolute path to the Excel file.
    :return: Path to the newly created CSV file with the same name as the input file but with .csv extension.
    :rtype: Path
    """
    if not input_xlsx_path.exists():
        raise FileNotFoundError(f"File not found: {input_xlsx_path}")

    if output_csv_path is None:
        output_csv_path = input_xlsx_path.with_suffix(".csv")

    logger.info(f"Reading Excel file: {input_xlsx_path}")
    data: pd.DataFrame = pd.read_excel(input_xlsx_path)

    logger.info(f"Writing CSV file: {output_csv_path}")
    data.to_csv(output_csv_path, index=False)

    return output_csv_path


def convert_zero_one_to_boolean(path: Path, column_names: list[str]) -> None:
    """Convert 0/1 values in columns to boolean.

    :param Path path: The path of the CSV file.
    :param list[str] columnnames: Names of columns which has 0 and 1 values that needs to be converted.

    :return: None
    """
    df = pd.read_csv(path)

    for column_name in column_names:
        df[column_name] = df[column_name].astype(int).astype(bool)

    # Save back to CSV
    df.to_csv(path, index=False)
