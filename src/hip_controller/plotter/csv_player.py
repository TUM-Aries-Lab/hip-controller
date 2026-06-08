"""Stateful CSV player that simulates real-time data arrival."""

from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from pandas import read_csv

from hip_controller.definitions import ExosuitData, RecordedSensorData, SensorSignal

# Default classification value used when the Classification Left / Right
# columns are absent (or contain NaN / unmappable values). Maps to Level
# Ground mode in the application-level dispatch.
DEFAULT_CLASSIFICATION = 0

# Accepted header names for each logical column. The first entry is the
# canonical name (matches RecordedSensorData where applicable) and is what
# the tests / data pipeline write out; the others are tolerated on input so
# externally produced CSVs (e.g. MATLAB exports) don't need to be renamed
# before playback.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": (RecordedSensorData.timestamp, "Time [s]", "time"),
    "ang_left": (RecordedSensorData.ang_left, "Angle Left Raw [rad]"),
    "ang_right": (RecordedSensorData.ang_right, "Angle Right Raw [rad]"),
    "vel_left": (RecordedSensorData.vel_left, "Vel Left Raw [rad]"),
    "vel_right": (RecordedSensorData.vel_right, "Vel Right Raw [rad]"),
    "main_switch": (RecordedSensorData.main_switch, "Main Switch"),
    "classification_left": ("classification_left", "Classification Left"),
    "classification_right": ("classification_right", "Classification Right"),
}


@dataclass
class PlayerStep:
    """One row of CSV playback: sensor signals + per-sample control inputs.

    :sensor_data: Raw (or recorded) left/right angle and velocity signals
        plus the timestamp shared by both legs.
    :main_switch: Whether the controller should run this sample (``True``) or
        be held idle (``False``). Defaults to ``True`` when the column is absent.
    :classification_left: Locomotion-mode classification for the left leg
        (0 = Level Ground, 1 = Ascend Stairs, 2 = Descend Stairs). Defaults to
        :data:`DEFAULT_CLASSIFICATION` when the column is absent.
    :classification_right: Locomotion-mode classification for the right leg.
    """

    sensor_data: ExosuitData
    main_switch: bool
    classification_left: int
    classification_right: int


def _resolve_column(
    available_columns: list[str], candidates: tuple[str, ...]
) -> str | None:
    """Return the first candidate header present in ``available_columns``.

    :param list[str] available_columns: Column names found in the loaded CSV.
    :param tuple[str, ...] candidates: Accepted header names for one logical
        column, ordered by preference (canonical first).
    :return: The matching column name, or ``None`` if none of the candidates
        are present.
    :rtype: str | None
    """
    for name in candidates:
        if name in available_columns:
            return name
    return None


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
        # sep=None + engine='python' lets pandas sniff the delimiter, so both
        # comma- and semicolon-separated files load without manual configuration.
        # decimal=',' covers European-locale exports (e.g. MATLAB on German
        # systems) that write "3,14" instead of "3.14".
        self.dataframe = read_csv(csv_path, sep=None, engine="python", decimal=",")
        # Strip incidental whitespace from headers so " angle_left (rad)" still
        # matches "angle_left (rad)".
        self.dataframe.columns = [str(c).strip() for c in self.dataframe.columns]
        self.counter = 0

        available = list(self.dataframe.columns)
        self._col_timestamp = _resolve_column(available, COLUMN_ALIASES["timestamp"])
        self._col_ang_left = _resolve_column(available, COLUMN_ALIASES["ang_left"])
        self._col_ang_right = _resolve_column(available, COLUMN_ALIASES["ang_right"])
        self._col_vel_left = _resolve_column(available, COLUMN_ALIASES["vel_left"])
        self._col_vel_right = _resolve_column(available, COLUMN_ALIASES["vel_right"])
        self._col_main_switch = _resolve_column(
            available, COLUMN_ALIASES["main_switch"]
        )
        self._col_classification_left = _resolve_column(
            available, COLUMN_ALIASES["classification_left"]
        )
        self._col_classification_right = _resolve_column(
            available, COLUMN_ALIASES["classification_right"]
        )

        if self._col_ang_left is None or self._col_ang_right is None:
            raise KeyError(
                "CSV is missing a left/right hip-angle column. Expected one of "
                f"{COLUMN_ALIASES['ang_left']} and one of "
                f"{COLUMN_ALIASES['ang_right']}. Found columns: {available}"
            )

    @property
    def has_timestamp(self) -> bool:
        """Whether a timestamp column was found in the CSV."""
        return self._col_timestamp is not None

    def has_next_line(self) -> bool:
        """Check whether more data is available.

        :return: True if there are remaining rows to read.
        :rtype: bool
        """
        return self.counter < len(self.dataframe)

    def get_sensor_data_from_csv(self) -> PlayerStep:
        """Get the recorded data from csv line by line.

        Velocity columns are optional: when missing, ``velocity_rad_per_sec`` is
        set to 0.0 and the controller is expected to derive velocity from the
        raw angle internally (run with ``filtered=False``).

        The main switch column is optional: when missing it defaults to ``True``
        (controller always active).

        The classification columns are optional: when missing they default to
        :data:`DEFAULT_CLASSIFICATION` (Level Ground).

        :return: :class:`PlayerStep` bundling sensor signals, main switch and
            per-leg locomotion classifications for this sample.
        :rtype: PlayerStep
        """
        row = self.dataframe.iloc[self.counter]
        self.counter += 1
        if self._col_timestamp is not None:
            timestamp = float(row[self._col_timestamp])
        else:
            timestamp = self.counter / RecordedSensorData.fake_frequency_hz

        vel_left = (
            float(row[self._col_vel_left]) if self._col_vel_left is not None else 0.0
        )
        vel_right = (
            float(row[self._col_vel_right]) if self._col_vel_right is not None else 0.0
        )
        main_switch = (
            bool(row[self._col_main_switch])
            if self._col_main_switch is not None
            else True
        )
        classification_left = (
            int(row[self._col_classification_left])
            if self._col_classification_left is not None
            else DEFAULT_CLASSIFICATION
        )
        classification_right = (
            int(row[self._col_classification_right])
            if self._col_classification_right is not None
            else DEFAULT_CLASSIFICATION
        )

        exosuit_data = ExosuitData(
            left=SensorSignal(
                timestamp=timestamp,
                angle_rad=float(row[self._col_ang_left]),
                velocity_rad_per_sec=vel_left,
            ),
            right=SensorSignal(
                timestamp=timestamp,
                angle_rad=float(row[self._col_ang_right]),
                velocity_rad_per_sec=vel_right,
            ),
        )
        return PlayerStep(
            sensor_data=exosuit_data,
            main_switch=main_switch,
            classification_left=classification_left,
            classification_right=classification_right,
        )
