"""Functions to deal with csv data."""

from pathlib import Path

import pandas as pd
from loguru import logger


def convert_xlsx_to_csv(
    input_xlsx_path: Path, output_csv_path: Path | None = None
) -> Path:
    """Convert an Excel file to CSV format.

    Reads a single Excel file (.xls or .xlsx) from the testing directory and writes
    its contents to a CSV file with the same filename stem in the same directory.
    This is useful for converting test data and measurement recordings to a
    more portable and scriptable format.

    :param Path input_xlsx_path: Absolute path to the input Excel file.
    :param Path | None output_csv_path: Absolute path to the output CSV file.

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
    :param list[str] column_names: Names of columns which have 0 and 1 values that need to be converted.

    :return: None
    """
    df = pd.read_csv(path)

    for column_name in column_names:
        df[column_name] = df[column_name].astype(int).astype(bool)

    # Save back to CSV
    df.to_csv(path, index=False)
