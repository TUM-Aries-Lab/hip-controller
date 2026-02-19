"""Stateful CSV player that simulates real-time data arrival."""

from pathlib import Path

from pandas import read_csv

from hip_controller.definitions import RecordedSensorData


class CSVPlayer:
    """The CSV file is loaded fully once using pandas.

    Each call to `next()` returns exactly one row, mimicking a single real-time sensor update.

    This design matches real-time GUI patterns:
    - no blocking I/O
    - deterministic stepping
    - easy replacement with a real sensor stream later
    """

    def __init__(self, csv_path: Path):
        """Load the CSV file into memory.

        :param str csv_path: Path to the CSV file containing time, angle, and velocity columns.
        """
        self.df = read_csv(csv_path)
        self.i = 0

        self.has_timestamp: bool = RecordedSensorData.TIMESTAMP in self.df.columns

    def has_next(self) -> bool:
        """Check whether more data is available.

        :return: True if there are remaining rows to read.
        :rtype: bool
        """
        return self.i < len(self.df)

    def get_sensor_data_from_csv(self):
        """Get the recorded data from csv line by line.

        :return: timestamp, angle_left, velocity_left, angle_right, velocity_right
        :rtype: float, float, float, float, float
        """
        row = self.df.iloc[self.i]
        self.i += 1
        if self.has_timestamp:
            timestamp = float(row[RecordedSensorData.TIMESTAMP])
        else:
            timestamp = self.i / RecordedSensorData.FREQUENCY_HZ
        # sensor values
        ang_left = float(row[RecordedSensorData.ANG_LEFT])
        vel_left = float(row[RecordedSensorData.VEL_LEFT])
        ang_right = float(row[RecordedSensorData.ANG_RIGHT])
        vel_right = float(row[RecordedSensorData.VEL_RIGHT])

        return timestamp, ang_left, vel_left, ang_right, vel_right
