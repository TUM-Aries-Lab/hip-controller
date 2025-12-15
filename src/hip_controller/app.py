"""Sample doc string."""

from loguru import logger

from hip_controller.definitions import DEFAULT_LOG_LEVEL
from hip_controller.utils import setup_logger


def main(
    log_level: str = DEFAULT_LOG_LEVEL, stderr_level: str = DEFAULT_LOG_LEVEL
) -> None:
    """Run the main pipeline.

    :param log_level: The log level to use.
    :param stderr_level: The std err level to use.
    :return: None
    """
    setup_logger(log_level=log_level, stderr_level=stderr_level)
    logger.info("Hello, world!")
