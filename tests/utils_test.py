"""Test the utils module."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from hip_controller.definitions import DEFAULT_LOG_FILENAME, ColumnName, LogLevel
from hip_controller.utils import convert_xlsx_to_csv, setup_logger


def test_logger_init() -> None:
    """Test logger initialization."""
    with TemporaryDirectory() as log_dir:
        log_dir_path = Path(log_dir)
        log_filepath = setup_logger(filename=DEFAULT_LOG_FILENAME, log_dir=log_dir_path)
        assert Path(log_filepath).exists()
    assert not Path(log_filepath).exists()


def test_log_level() -> None:
    """Test the log level.

    :return: None
    """
    # Act
    log_levels = list(LogLevel())

    # Assert
    assert type(log_levels) is list


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

    :param tmp_path: Temporary output Excel file path for testing.
    :return: None
    """
    df: pd.DataFrame = pd.DataFrame(
        {ColumnName.ANGLE: [0.1, -0.2], ColumnName.VELOCITY: [1.0, -0.5]},
    )
    xlsx: Path = tmp_path / "test.xlsx"
    _make_excel(xlsx, {"Sheet1": df})

    out: Path = convert_xlsx_to_csv(xlsx)
    assert out.exists()

    read: pd.DataFrame = pd.read_csv(out)
    pd.testing.assert_frame_equal(read, df)

    def test_convert_xlsx_to_csv_file_not_found(tmp_path: Path) -> None:
        """Test that ``convert_xlsx_to_csv`` raises ``FileNotFoundError`` when the input Excel file does not exist.

        :param tmp_path: Temporary directory provided by pytest.
        :return: None
        :raises FileNotFoundError: If the Excel file does not exist.
        """

    missing_file: Path = tmp_path / "does_not_exist.xlsx"

    with pytest.raises(FileNotFoundError):
        convert_xlsx_to_csv(missing_file)
