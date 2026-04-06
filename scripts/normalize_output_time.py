"""Script to convert all output data's time column to start at 0 with an increment of 0.01."""

import pandas as pd
from pathlib import Path

def normalize_output_file_time(filepath: str | Path) -> pd.DataFrame:
    """Edit an output csv file so that the time column starts at 0 with an increment of 0.01.

    :param filepath: absolute path to the csv file.
    :return: converted dataframe.
    """
    timekey = "time (s)"
    df = pd.read_csv(filepath)
    df[timekey] = {i: 0.01 * i for i in range(len(df))}
    return df

def normalize_output_folder(src_path: str | Path, dest_path: str | Path) -> None:
    """Normalize time for all csv files in a given path, saving edited files in the designated path, preserving the structure.

    :param src_path: source path containing all files to be edited.
    :param dest_path: destination path for all edited files.
    :return: None
    """
    src_folder = Path(src_path)
    src_files = src_folder.rglob('*.csv')
    for src_file in src_files:
        relative_path = src_file.relative_to(src_folder)
        normalize_output_folder(src_path=relative_path, dest_path=dest_path)

        normalized_file = dest_path / relative_path.with_suffix('.csv')
        normalized_file.parent.mkdir(parents=True, exist_ok=True)
        df = normalize_output_file_time(filepath=src_file)
        df.to_csv(normalized_file, index=False)

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    normalize_output_folder(src_path="scripts/evaluation_output", dest_path="scripts/normalized_output")
