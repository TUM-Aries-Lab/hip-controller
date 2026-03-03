"""Stateful CSV player that simulates real-time data arrival."""

from pathlib import Path

import pandas as pd
from loguru import logger
from pandas import read_csv

from hip_controller.definitions import ExosuitData, RecordedSensorData, SensorSignal


class CSVPlayer:
    """The CSV file is loaded fully once using pandas.

    Each call to `next()` returns exactly one row, mimicking a single real-time sensor update.

    This design matches real-time GUI patterns:
    - no blocking I/O
    - deterministic stepping
    - easy replacement with a real sensor stream later
    """

    def __init__(self, csv_path: Path) -> None:
        """Load the CSV file into memory.

        :param str csv_path: Path to the CSV file containing time, angle, and velocity columns. Default takes the file path from RecordedSensorData setup in definitions.
        """
        logger.info(f"Loading CSV file '{csv_path}'.")
        self.dataframe = read_csv(csv_path)
        self.counter = 0

        self.has_timestamp: bool = (
            RecordedSensorData.timestamp in self.dataframe.columns
        )

    def has_next_line(self) -> bool:
        """Check whether more data is available.

        :return: True if there are remaining rows to read.
        :rtype: bool
        """
        return self.counter < len(self.dataframe)

    def get_sensor_data_from_csv(self) -> ExosuitData:
        """Get the recorded data from csv line by line.

        :return: timestamp, angle_left, velocity_left, angle_right, velocity_right packed together as a dataclass
        :rtype: Sensordata
        """
        row = self.dataframe.iloc[self.counter]
        self.counter += 1
        if self.has_timestamp:
            timestamp = float(row[RecordedSensorData.timestamp])
        else:
            timestamp = self.counter / RecordedSensorData.fake_frequency_hz

        return ExosuitData(
            timestamp=timestamp,
            left=SensorSignal(
                angle_rad=float(row[RecordedSensorData.ang_left]),
                velocity_rad_per_sec=float(row[RecordedSensorData.vel_left]),
            ),
            right=SensorSignal(
                angle_rad=float(row[RecordedSensorData.ang_right]),
                velocity_rad_per_sec=float(row[RecordedSensorData.vel_right]),
            ),
        )

    def get_data_from_csv(
        self, input_name: str, output_name: str
    ) -> tuple[float, float, float]:
        """Get the recorded data from csv line by line.

        :return: timestamp, angle_left, velocity_left, angle_right, velocity_right packed together as a dataclass
        :rtype: Sensordata
        """
        row = self.dataframe.iloc[self.counter]
        self.counter += 1
        if self.has_timestamp:
            timestamp = float(row[RecordedSensorData.timestamp])
        else:
            timestamp = self.counter / RecordedSensorData.fake_frequency_hz

        return timestamp, float(row[input_name]), float(row[output_name])


def convert_xlsx_to_csv(path: Path) -> Path:
    """Convert an Excel file to CSV format.

    Reads a single Excel file (.xls or .xlsx) from the testing directory and writes
    its contents to a CSV file with the same filename stem in the same directory.
    This is useful for converting test data and measurement recordings to a
    more portable and scriptable format.

    :param Path path: Absolute path to the Excel file.
    :return: Path to the newly created CSV file with the same name as the input file but with .csv extension.
    :rtype: Path
    """
    xlsx_path = path

    if not xlsx_path.exists():
        raise FileNotFoundError(f"File not found: {xlsx_path}")

    output_path = xlsx_path.with_suffix(".csv")

    logger.info(f"Reading Excel file: {xlsx_path}")
    data: pd.DataFrame = pd.read_excel(xlsx_path)

    logger.info(f"Writing CSV file: {output_path}")
    data.to_csv(output_path, index=False)

    return output_path


def convert_zero_one_to_boolean(path: Path, column_names: list[str]) -> None:
    """Convert 0/1 values in columns to boolean.

    :param Path path: The path of the CSV file.
    :param list[str] columnnames: Names of columns which has 0 and 1 values that needs to be converted.

    :return: None
    """
    df = pd.read_csv(path)

    for column_name in column_names:
        df[column_name] = df[column_name].astype(int).astype(bool)

    # Save back to CSV
    df.to_csv(path, index=False)
