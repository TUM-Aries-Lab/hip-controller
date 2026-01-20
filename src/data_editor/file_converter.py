"""Excel-to-CSV file converter.

This module provides a small command-line utility and library function
to convert a single Excel file (``.xls`` or ``.xlsx``) into a CSV file.

Paths passed via the CLI are resolved relative to the project's
``data/sensor_data/`` directory unless an absolute path is provided.

Command-line usage
==================

Convert an Excel file to CSV::

    python -m data_editor.file_converter \
        --file high_level_testing/valid_trigger_left_2026_01_15.xlsx

Programmatic usage
==================

::

    from pathlib import Path
    from data_editor.file_converter import convert_xlsx_to_csv

    csv_path = convert_xlsx_to_csv(Path("data.xlsx"))
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from hip_controller.definitions import SENSOR_DATA_DIR


def convert_xlsx_to_csv(
    xlsx_path: Path,
    csv_path: Path | None = None,
    sheet_name: int | None = 0,
) -> Path:
    """Convert a single Excel file to CSV.

    :param xlsx_path: Path to the input Excel file.
    :param csv_path: Optional output CSV path.
    :param sheet_name: Sheet name or index to read.
    :returns: Path to the written CSV file.
    :raises FileNotFoundError: If the input file does not exist.
    """
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Input file not found: {xlsx_path}")

    output_path: Path = csv_path or xlsx_path.with_suffix(".csv")

    logger.info(f"Reading Excel {xlsx_path} (sheet={sheet_name})")
    data: pd.DataFrame | dict[str, pd.DataFrame] = pd.read_excel(
        xlsx_path,
        sheet_name=sheet_name,
    )

    if isinstance(data, dict):
        first_sheet: str = next(iter(data))
        logger.warning(f"Multiple sheets found; using first sheet: {first_sheet}")
        data = data[first_sheet]

    logger.info(f"Writing CSV {output_path}")
    data.to_csv(output_path, index=False)

    return output_path


def main() -> None:  # pragma: no cover
    """Command-line entry point for Excel-to-CSV conversion."""
    parser = argparse.ArgumentParser(
        description="Convert a single Excel file to CSV",
    )

    parser.add_argument(
        "--file",
        type=Path,
        required=True,
        help="Path to Excel file (absolute or relative to sensor_data/)",
    )

    parser.add_argument(
        "--sheet",
        type=int,
        default=0,
        help="Excel sheet name or index to read",
    )

    args = parser.parse_args(sys.argv)

    # Resolve relative paths against SENSOR_DATA_DIR
    file_path: Path = args.file
    if not file_path.is_absolute():
        file_path = SENSOR_DATA_DIR / file_path

    output_path: Path = convert_xlsx_to_csv(
        file_path,
        sheet_name=args.sheet,
    )

    logger.info(f"Converted {file_path} -> {output_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
