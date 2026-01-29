"""Configure the logger."""

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from hip_controller.definitions import (
    DATE_FORMAT,
    DEFAULT_LOG_FILENAME,
    DEFAULT_LOG_LEVEL,
    ENCODING,
    LOG_DIR,
    TESTING_DIR,
)


def create_timestamped_filepath(suffix: str, output_dir: Path, prefix: str) -> Path:
    """Generate a timestamped filename.

    :param suffix: Suffix to append to the timestamped filename.
    :param output_dir: Output directory.
    :param prefix: Prefix to append to the timestamped filename.
    :return: Path to the timestamped filename.
    """
    timestamp = datetime.now().strftime(DATE_FORMAT)
    filepath = output_dir / f"{prefix}_{timestamp}.{suffix}"
    filepath.parent.mkdir(parents=True, exist_ok=True)  # create dirs if missing
    filepath.touch(exist_ok=True)  # create empty file (don't overwrite)
    return filepath


def setup_logger(
    filename: str = DEFAULT_LOG_FILENAME,
    stderr_level: str = DEFAULT_LOG_LEVEL,
    log_level: str = DEFAULT_LOG_LEVEL,
    log_dir: Path | None = None,
) -> Path:
    """Configure the logger.

    :param filename: Name of the file to create.
    :param stderr_level: Logging level to use.
    :param log_level: Logging level to use.
    :param log_dir: Logging directory to use.
    :return: Path to the created logfile.
    """
    logger.remove()

    if log_dir is None:
        log_filepath = LOG_DIR
    else:
        log_filepath = log_dir
    filepath_with_time = create_timestamped_filepath(
        output_dir=log_filepath, prefix=filename, suffix="log"
    )
    logger.add(sys.stderr, level=stderr_level)
    logger.add(filepath_with_time, level=log_level, encoding=ENCODING, enqueue=True)
    logger.info(f"Logging to '{filepath_with_time}'.")
    return filepath_with_time


def get_sensor_data() -> tuple[float, float]:
    """Get fake sensor data."""
    logger.debug("Getting fake sensor data.")
    return np.sin(time.monotonic() / 2), np.cos(time.monotonic() / 2)


def convert_xlsx_to_csv(path: Path) -> Path:
    """Convert an Excel file to CSV format.

    Reads a single Excel file (.xls or .xlsx) from the testing directory and writes
    its contents to a CSV file with the same filename stem in the same directory.
    This is useful for converting test data and measurement recordings to a
    more portable and scriptable format.

    :param Path path:
        Relative path to the Excel file from the TESTING_DIR root.
        Example: 'controller_test/high_level_controller/high_level_testing_data/gait_phase_left_2026_01_21.xlsx'
    :raises FileNotFoundError:
        If the specified Excel file does not exist in the TESTING_DIR.
    :return:
        Path to the newly created CSV file with the same name as the input file
        but with .csv extension.
    :rtype: Path

    .. note::
        The Excel file must be located under the TESTING_DIR. The resulting CSV
        file is written to the same directory with the same stem.

    .. rubric:: Example

    ::

        from hip_controller.utils import convert_xlsx_to_csv
        from pathlib import Path

        csv_path = convert_xlsx_to_csv(
            path=Path('controller_test/high_level_controller/high_level_testing_data/gait_phase_left_2026_01_21.xlsx'))
    """
    xlsx_path = TESTING_DIR / path

    if not xlsx_path.exists():
        raise FileNotFoundError(f"File not found: {xlsx_path}")

    output_path = xlsx_path.with_suffix(".csv")

    logger.info(f"Reading Excel file: {xlsx_path}")
    data: pd.DataFrame = pd.read_excel(xlsx_path)

    logger.info(f"Writing CSV file: {output_path}")
    data.to_csv(output_path, index=False)

    return output_path
