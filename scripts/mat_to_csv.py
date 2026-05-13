from pathlib import Path

import numpy as np
from scipy.io import loadmat
import pandas as pd

def mat_to_csv(mat_file: str | Path) -> pd.DataFrame:
    """Convert a mat file into a csv file, only keeping 'left' and 'right' entries.

    :param mat_file: absolute path to the mat file.
    """
    data = loadmat(mat_file)
    # Filter out MATLAB metadata keys (starting with '_')
    filtered_data = {k: v for k, v in data.items() if k[0] != '_'}

    # Extract left and right arrays
    left = filtered_data.get('left', np.array([])).squeeze()
    right = filtered_data.get('right', np.array([])).squeeze()

    # Convert to 2D arrays if needed
    if left.size > 0 and left.ndim == 1:
        left = left.reshape(-1, 2)
    elif left.size == 0:
        left = np.empty((0, 2))

    if right.size > 0 and right.ndim == 1:
        right = right.reshape(-1, 2)
    elif right.size == 0:
        right = np.empty((0, 2))

    # Create DataFrames
    left_df = pd.DataFrame(left, columns=['left_col1', 'left_col2']) if left.shape[0] > 0 else pd.DataFrame()
    right_df = pd.DataFrame(right, columns=['right_col1', 'right_col2']) if right.shape[0] > 0 else pd.DataFrame()

    # Determine max rows and reindex to add NaN padding
    max_rows = max(len(left_df), len(right_df))
    if max_rows > 0:
        if len(left_df) < max_rows:
            left_df = left_df.reindex(range(max_rows))
        if len(right_df) < max_rows:
            right_df = right_df.reindex(range(max_rows))

    # Concatenate horizontally
    return pd.concat([left_df, right_df], axis=1)

def convert_all_mat_to_csv(mat_folder: str | Path, dest_folder: str | Path) -> None:
    """
    Convert all .mat files in a folder structure to CSV files.

    :param mat_folder: Source folder containing .mat files
    :param dest_folder: Destination folder for CSV files (structure is mirrored)
    """
    mat_folder = Path(mat_folder)
    dest_folder = Path(dest_folder)

    # Find all .mat files recursively
    mat_files = mat_folder.rglob('*.mat')

    for mat_file in mat_files:
        # Get relative path from source folder
        relative_path = mat_file.relative_to(mat_folder)
        convert_all_mat_to_csv(mat_folder=relative_path, dest_folder=dest_folder)

        # Create destination file path with .csv extension
        csv_file = dest_folder / relative_path.with_suffix('.csv')

        # Create parent directories if they don't exist
        csv_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert mat to dataframe and save as csv
        df = mat_to_csv(mat_file)
        df.to_csv(csv_file, index=False)

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    convert_all_mat_to_csv(mat_folder="scripts/evaluation_data/organized",dest_folder="scripts/evaluation_data/converted")
