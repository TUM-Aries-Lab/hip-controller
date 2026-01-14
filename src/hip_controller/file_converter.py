"""Utilities to convert a single Excel file to CSV and to read CSV sensor data.

Simple usage examples:
  Convert a single Excel file:
    python -m hip_controller.file_converter --file data/sensor_data/file.xlsx --convert

  Read a CSV file and print its head:
    python -m hip_controller.file_converter --file data/sensor_data/file.csv --read

Programmatic usage:
  from hip_controller.file_converter import convert_xlsx_to_csv, read_csv
  csv = convert_xlsx_to_csv(Path('data.xlsx'))
  df = read_csv(csv)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

LOG = logging.getLogger(__name__)


def convert_xlsx_to_csv(
    xlsx_path: Path,
    csv_path: Path | None = None,
    sheet_name: int | str | None = 0,
) -> Path:
    """Convert a single Excel file to CSV.

    Args:
        xlsx_path: path to the input .xls or .xlsx file.
        csv_path: optional path for output CSV. If omitted, same stem with .csv in same folder.
        sheet_name: sheet to read (passed to pandas.read_excel).

    Returns:
        Path to the written CSV file.

    """
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Input file not found: {xlsx_path}")

    if csv_path is None:
        csv_path = xlsx_path.with_suffix(".csv")
    csv_path = Path(csv_path)

    LOG.info("Reading excel %s (sheet=%s)", xlsx_path, sheet_name)
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    # pandas.read_excel returns a DataFrame when a single sheet is read,
    # but returns a dict[str, DataFrame] when multiple sheets are requested
    # (or sheet_name=None). Handle that here by choosing the first sheet.
    if isinstance(df, dict):
        first_sheet = next(iter(df))
        LOG.warning(
            "read_excel returned multiple sheets; using first sheet: %s", first_sheet
        )
        df = df[first_sheet]

    LOG.info("Writing csv %s", csv_path)
    df.to_csv(csv_path, index=False)
    return csv_path


def read_csv(csv_path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV file into a pandas DataFrame.

    Any extra kwargs are forwarded to `pandas.read_csv`.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path, **kwargs)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    """Use Main function for command-line."""
    parser = argparse.ArgumentParser(
        description="Convert a single Excel file to CSV or read a CSV"
    )
    parser.add_argument("--file", type=Path, help="Single file to convert or read")
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Convert given Excel file to CSV (use with --file)",
    )
    parser.add_argument(
        "--read",
        action="store_true",
        help="Read and print head of CSV (use with --file)",
    )
    parser.add_argument("--sheet", default=0, help="Excel sheet name or index to read")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.file:
        parser.print_help()
        return 1

    try:
        if args.convert:
            out = convert_xlsx_to_csv(args.file, sheet_name=args.sheet)
            LOG.info("Converted %s -> %s", args.file, out)
            return 0

        if args.read:
            df = read_csv(args.file)
            LOG.info(df.head())
            return 0

        parser.print_help()
        return 1
    except Exception as exc:
        LOG.error("Operation failed: %s", exc)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
