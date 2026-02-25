"""Stateful CSV player that simulates real-time data arrival."""

from pathlib import Path

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
