"""Main function of the controller.

Reads data from a CSV file with CSV player with timestamp and compute the controller steps.
"""

import argparse  # pragma: no cover
from pathlib import Path  # pragma: no cover

from hip_controller.definitions import (
    DEFAULT_LOG_LEVEL,
    BasicConfig,
    LogLevel,
)  # pragma: no cover
from scripts.simulator import simulate_controller_with_data


def main(
    log_level: str = DEFAULT_LOG_LEVEL,
    stderr_level: str = DEFAULT_LOG_LEVEL,
    csv_path: Path = BasicConfig.read_data_from_path,
) -> None:  # pragma: no cover
    """Run the main pipeline.

    :param log_level: The log level to use.
    :param stderr_level: The std err level to use.
    :param str csv_path: Path to the CSV file used for simulated real-time playback. The user could pass in the path of a file as well.
    :return: None

    # Example
        -  simulate_controller_with_data(log_level=log_level, stderr_level=stderr_level, csv_path=BasicConfig.read_data_from_path)
    """
    simulate_controller_with_data(
        log_level=log_level, stderr_level=stderr_level, csv_path=csv_path
    )


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser("Run the pipeline.")
    parser.add_argument(
        "--log-level",
        "-l",
        default=DEFAULT_LOG_LEVEL,
        choices=list(LogLevel()),
        help="Set the log level.",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--stderr-level",
        "-s",
        default=DEFAULT_LOG_LEVEL,
        choices=list(LogLevel()),
        help="Set the std err level.",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--file-path",
        "-p",
        default=Path(BasicConfig.read_data_from_path),
        choices=list(LogLevel()),
        help="Path to the CSV file used for simulated real-time playback. The file has to contain columns name 'angle_left (rad)', 'vel_left (rad/s)', 'angle_right (rad)', 'vel_right (rad/s)', additinally 'time (s)'.",
        required=False,
        type=Path,
    )
    args = parser.parse_args()

    main(
        log_level=args.log_level,
        stderr_level=args.stderr_level,
        csv_path=args.file_path,
    )
