"""Stateful CSV player that simulates real-time data arrival."""

from pathlib import Path

from loguru import logger
from pandas import read_csv
from dataclasses import dataclass
from hip_controller.definitions import RecordedSensorData


@dataclass
class ComparisonData:
    """Data class for comparison of a function's actual outputs and expected outputs with given inputs."""
    timestamp: float
    given_input: float
    expected_output: float
    actual_output:float| None = None

class ScriptPlayer:
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


    def get_data_from_csv(
        self, input_name: str, expected_output_name: str
    ) -> ComparisonData:
        """Get the data input and expected output from csv line by line to compare output executed from input and expected output.

        :param str input_name: Name of the input column.
        :param str expected_output_name: Name of the output column.

        :return: timestamp, float value of given input, float value of given output
        :rtype: tuple[float, float, float]
        """
        row = self.dataframe.iloc[self.counter]
        self.counter += 1
        if self.has_timestamp:
            timestamp = float(row[RecordedSensorData.timestamp])
        else:
            timestamp = self.counter / RecordedSensorData.fake_frequency_hz

        return ComparisonData(timestamp=timestamp, given_input=float(row[input_name]), expected_output=float(row[expected_output_name]))
