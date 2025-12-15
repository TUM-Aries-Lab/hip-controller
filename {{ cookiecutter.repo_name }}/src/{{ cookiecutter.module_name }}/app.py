"""Sample doc string."""

import argparse

from {{ cookiecutter.module_name }}.definitions import DEFAULT_LOG_LEVEL, LogLevel
from {{ cookiecutter.module_name }}.utils import setup_logger
from loguru import logger


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