"""Functions to deal with csv data.

Walking data CSV processing utilities.
Handles stair concatenation, file combining, and column filtering.

"""

from pathlib import Path

import pandas as pd
from loguru import logger

KEEP_COLS = ["time", "hip_flexion_r", "hip_flexion_l", "source_file"]
PARTICIPANTS = [f"AB{i:02d}" for i in range(1, 14)]  # AB01–AB13

def filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only time, hip_flexion_r, hip_flexion_l, source_file columns.

    :param pd.DataFrame df: The input dataframe.
    :return: The filtered dataframe.
    :rtype: pd.DataFrame
    """
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


    :param str | Path stairs_root:  Path to the 'stairs' folder containing AB01-AB13 subdirs.
    :param str | Path output_dir:   Destination folder for the combined per-participant files.
    :param list[str] | None = None participants: Optional list of participant IDs (e.g. ['AB01', 'AB03']).
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


    :param str | Path file_a:      Path to the first CSV file.
    :param str | Path file_b:      Path to the second CSV file.
    :param str | Path output_path: Destination path for the merged CSV.

    :return: The merged DataFrame.

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








def _read_and_filter(path: Path) -> pd.DataFrame:
    """Read a CSV and retain only the required columns."""
    df = pd.read_csv(path)
    available = [c for c in KEEP_COLS if c in df.columns]
    missing = set(KEEP_COLS) - set(available)
    if missing:
       logger.warning(f"  Warning: {path.name} missing columns {missing}")
    return df[available]


def _find_file(folder: Path, participant: str, keyword: str) -> Path | None:
    """Return the first CSV in folder whose name contains participant and keyword."""
    matches = list(folder.glob(f"{participant}*{keyword}*angle.csv"))
    if not matches:
        logger.warning(f"  Warning: no file found for {participant} with '{keyword}' in {folder}")
        return None
    return matches[0]


def combine_incline_walk(
    incline_root: str | Path,
    output_dir: str | Path,
    participants: list[str] = PARTICIPANTS,
) -> None:
    """
    For each participant and each slope (5 and 10), concatenate up then down
    into a single CSV containing only time, hip_flexion_r, hip_flexion_l.


    :param incline_root: Path to the 'incline_walk' folder.
    :param output_dir:   Destination folder for combined files.
    :param participants: List of participant IDs. Defaults to AB01–AB13.
    """
    incline_root = Path(incline_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slopes = [
        ("incline_walk_5",  "up5",  "down5",  5),
        ("incline_walk_10", "up10", "down10", 10),
    ]

    for subfolder, up_kw, down_kw, slope in slopes:
        folder = incline_root / subfolder
        if not folder.exists():
            logger.warning((f"Folder not found: {folder} — skipping slope {slope}.\n"))
            continue


        for participant in participants:
            up_file   = _find_file(folder, participant, up_kw)
            down_file = _find_file(folder, participant, down_kw)

            if not up_file or not down_file:
                continue  # warning already printed inside _find_file

            df_up   = _read_and_filter(up_file);   df_up["source_file"]   = up_file.name
            df_down = _read_and_filter(down_file); df_down["source_file"] = down_file.name

            combined = pd.concat([df_up, df_down], ignore_index=True)

            out_path = output_dir / f"{participant}_incline_walk_{slope}_combined.csv"
            combined.to_csv(out_path, index=False)
            logger.info(f"  {participant}: {up_file.name} + {down_file.name} → {out_path.name}  ({len(combined)} rows)")
