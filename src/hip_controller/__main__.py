"""Sample doc string."""

import argparse  # pragma: no cover
import time

from loguru import logger  # pragma: no cover

from hip_controller.app import WalkOnController  # pragma: no cover
from hip_controller.definitions import DEFAULT_LOG_LEVEL, LogLevel  # pragma: no cover
from hip_controller.utils import get_sensor_data, setup_logger  # pragma: no cover


def main(
    log_level: str = DEFAULT_LOG_LEVEL, stderr_level: str = DEFAULT_LOG_LEVEL
) -> None:  # pragma: no cover
    """Run the main pipeline.

    :param log_level: The log level to use.
    :param stderr_level: The std err level to use.
    :return: None
    """
    setup_logger(log_level=log_level, stderr_level=stderr_level)

    controller = WalkOnController()
    freq = 1.0

    try:
        while True:
            theta, theta_dot = get_sensor_data()
            controller.step(
                theta=theta, theta_dot=theta_dot, timestamp=time.monotonic()
            )
            time.sleep(1 / freq)
    except KeyboardInterrupt:
        logger.success("User interrupted.")


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
    args = parser.parse_args()

    main(log_level=args.log_level, stderr_level=args.stderr_level)
