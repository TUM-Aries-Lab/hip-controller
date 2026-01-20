"""Tests for the file_converter module."""

from pathlib import Path

import pandas as pd

from data_editor.file_converter import convert_xlsx_to_csv


def _make_excel(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    """Write an Excel file with the given sheets.

    :param path: Output Excel file path.
    :param sheets: Mapping of sheet name to DataFrame.
    :return: None
    """
    with pd.ExcelWriter(path) as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)


def test_convert_xlsx_to_csv_creates_csv(tmp_path: Path) -> None:
    """Test converting an Excel file to CSV.

    :param path: Temporary output Excel file path for testing.
    :return: None
    """
    df: pd.DataFrame = pd.DataFrame(
        {"angle": [0.1, -0.2], "velocity": [1.0, -0.5]},
    )
    xlsx: Path = tmp_path / "test.xlsx"
    _make_excel(xlsx, {"Sheet1": df})

    out: Path = convert_xlsx_to_csv(xlsx)
    assert out.exists()

    read: pd.DataFrame = pd.read_csv(out)
    pd.testing.assert_frame_equal(read, df)


def test_convert_xlsx_with_multiple_sheets_uses_first(tmp_path: Path) -> None:
    """Test that the first sheet is used when multiple sheets exist.

    :param path: Temporary output Excel file path for testing.
    :return: None
    """
    df1: pd.DataFrame = pd.DataFrame({"a": [1, 2]})
    df2: pd.DataFrame = pd.DataFrame({"a": [3, 4]})
    xlsx: Path = tmp_path / "multi.xlsx"
    _make_excel(xlsx, {"first": df1, "second": df2})

    out: Path = convert_xlsx_to_csv(xlsx, sheet_name=None)
    read: pd.DataFrame = pd.read_csv(out)

    pd.testing.assert_frame_equal(read, df1)
