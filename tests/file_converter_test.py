"""Tests for the file_converter module."""

from pathlib import Path

import pandas as pd
import pytest

from hip_controller.file_converter import convert_xlsx_to_csv, main, read_csv


def _make_excel(path: Path, sheets: dict[str, pd.DataFrame]):
    """Write an Excel file with the given sheets (sheet_name -> DataFrame)."""
    with pd.ExcelWriter(path) as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)


def test_convert_xlsx_to_csv_creates_csv(tmp_path: Path) -> None:
    """Test converting an Excel file to CSV."""
    df = pd.DataFrame({"angle": [0.1, -0.2], "velocity": [1.0, -0.5]})
    xlsx = tmp_path / "test.xlsx"
    _make_excel(xlsx, {"Sheet1": df})

    out = convert_xlsx_to_csv(xlsx)
    assert out.exists()

    read = pd.read_csv(out)
    pd.testing.assert_frame_equal(read, df)


def test_convert_xlsx_with_multiple_sheets_uses_first(tmp_path: Path) -> None:
    """Test that converting an Excel file with multiple sheets uses the first sheet."""
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"a": [3, 4]})
    xlsx = tmp_path / "multi.xlsx"
    _make_excel(xlsx, {"first": df1, "second": df2})

    out = convert_xlsx_to_csv(xlsx, sheet_name=None)
    assert out.exists()
    read = pd.read_csv(out)
    pd.testing.assert_frame_equal(read, df1)


def test_read_csv_missing_raises(tmp_path: Path) -> None:
    """Test that reading a missing CSV raises FileNotFoundError."""
    missing = tmp_path / "nope.csv"
    with pytest.raises(FileNotFoundError):
        read_csv(missing)


def test_main_convert_and_read_cli(tmp_path: Path, capsys) -> None:
    """Test the main CLI function for converting and reading files."""
    df = pd.DataFrame({"angle": [0.0, 1.0], "velocity": [0.5, -0.5]})
    xlsx = tmp_path / "cli_test.xlsx"
    _make_excel(xlsx, {"Sheet1": df})

    # run convert
    ret = main(["--file", str(xlsx), "--convert"])  # type: ignore[arg-type]
    assert ret == 0
    csv_path = xlsx.with_suffix(".csv")
    assert csv_path.exists()

    # run read (should log and return 0)
    ret2 = main(["--file", str(csv_path), "--read"])  # type: ignore[arg-type]
    assert ret2 == 0
