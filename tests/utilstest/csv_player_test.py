"""Test for the csv player module."""

from pathlib import Path

import pandas as pd

from hip_controller.definitions import RecordedSensorData, SensorData
from hip_controller.utils.csv_player import CSVPlayer


def create_test_csv(path: Path):
    """Create a data frame to test the CSV player."""
    df = pd.DataFrame(
        {
            RecordedSensorData.TIMESTAMP: [0.0, 0.1, 0.2],
            RecordedSensorData.ANG_LEFT: [1.0, 2.0, 3.0],
            RecordedSensorData.VEL_LEFT: [0.1, 0.2, 0.3],
            RecordedSensorData.ANG_RIGHT: [4.0, 5.0, 6.0],
            RecordedSensorData.VEL_RIGHT: [0.4, 0.5, 0.6],
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

    assert t0 == SensorData(0.0, 1.0, 0.1, 4.0, 0.4)
    assert t1 == SensorData(0.1, 2.0, 0.2, 5.0, 0.5)


def test_csv_player_index_increments(tmp_path):
    """Test the incrementation of indices for the CSV Player."""
    csv_path = tmp_path / "test.csv"
    create_test_csv(csv_path)

    player = CSVPlayer(csv_path)

    assert player.i == 0
    player.get_sensor_data_from_csv()
    assert player.i == 1
