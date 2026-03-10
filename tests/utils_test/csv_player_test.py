"""Test for the csv player module."""

from pathlib import Path

import pandas as pd
from pandas import DataFrame, ExcelWriter, read_csv, testing
from pytest import raises

from hip_controller.definitions import ExosuitData, RecordedSensorData, SensorSignal
from hip_controller.plotter.csv_player import CSVPlayer, convert_xlsx_to_csv


def create_test_csv(path: Path):
    """Create a data frame to test the CSV player."""
    df = pd.DataFrame(
        {
            RecordedSensorData.timestamp: [0.0, 0.1, 0.2],
            RecordedSensorData.ang_left: [1.0, 2.0, 3.0],
            RecordedSensorData.vel_left: [0.1, 0.2, 0.3],
            RecordedSensorData.ang_right: [4.0, 5.0, 6.0],
            RecordedSensorData.vel_right: [0.4, 0.5, 0.6],
        }
    )
    df.to_csv(path, index=False)


def test_csv_player_has_next(tmp_path):
    """Test the has_next function for the CSV Player."""
    csv_path = tmp_path / "test.csv"
    create_test_csv(csv_path)

    player = CSVPlayer(csv_path)

    assert player.has_next_line() is True

    player.get_sensor_data_from_csv()
    player.get_sensor_data_from_csv()
    player.get_sensor_data_from_csv()

    assert player.has_next_line() is False


def test_csv_player_reads_rows_in_order(tmp_path):
    """Test the read row function for the CSV Player."""
    csv_path = tmp_path / "test.csv"
    create_test_csv(csv_path)

    player = CSVPlayer(csv_path)

    t0 = player.get_sensor_data_from_csv()
    t1 = player.get_sensor_data_from_csv()

    assert t0 == ExosuitData(
        timestamp=0.0, left=SensorSignal(1.0, 0.1), right=SensorSignal(4.0, 0.4)
    )
    assert t1 == ExosuitData(
        timestamp=0.1, left=SensorSignal(2.0, 0.2), right=SensorSignal(5.0, 0.5)
    )


def test_csv_player_index_increments(tmp_path):
    """Test the incrementation of indices for the CSV Player."""
    csv_path = tmp_path / "test.csv"
    create_test_csv(csv_path)

    player = CSVPlayer(csv_path)

    assert player.counter == 0
    player.get_sensor_data_from_csv()
    assert player.counter == 1


def _make_excel(path: Path, sheets: dict[str, DataFrame]) -> None:
    """Write an Excel file with the given sheets.

    :param path: Output Excel file path.
    :param sheets: Mapping of sheet name to DataFrame.
    :return: None
    """
    with ExcelWriter(path) as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)


def test_convert_xlsx_to_csv_creates_csv(tmp_path: Path) -> None:
    """Test converting an Excel file to CSV.

    :param tmp_path: Temporary output Excel file path for testing.
    :return: None
    """
    df: DataFrame = DataFrame(
        {"FirstColumn": [0.1, -0.2], "SecondColumn": [1.0, -0.5]},
    )
    xlsx: Path = tmp_path / "test.xlsx"
    _make_excel(xlsx, {"Sheet1": df})

    out: Path = convert_xlsx_to_csv(xlsx)
    assert out.exists()

    read: DataFrame = read_csv(out)
    testing.assert_frame_equal(read, df)


def test_convert_xlsx_to_csv_file_not_found(tmp_path: Path) -> None:
    """Test that ``convert_xlsx_to_csv`` raises ``FileNotFoundError`` when the input Excel file does not exist.

    :param tmp_path: Temporary directory provided by pytest.
    :return: None
    :raises FileNotFoundError: If the Excel file does not exist.
    """
    missing_file: Path = tmp_path / "does_not_exist.xlsx"

    with raises(FileNotFoundError):
        convert_xlsx_to_csv(missing_file)
