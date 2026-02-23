"""Main function of the controller.

Reads data from a CSV file with CSV player with timestamp and compute the controller steps.
"""

# pragma: no cover
import argparse
import signal

from loguru import logger
from pyqtgraph import QtCore, QtWidgets

from hip_controller.app import ExoController
from hip_controller.definitions import DEFAULT_LOG_LEVEL, LogLevel, RecordedSensorData
from hip_controller.utils.csv_player import CSVPlayer
from hip_controller.utils.utils import setup_logger


def main(
    log_level: str = DEFAULT_LOG_LEVEL, stderr_level: str = DEFAULT_LOG_LEVEL
) -> None:  # pragma: no cover
    """Run the main pipeline.

    :param log_level: The log level to use.
    :param stderr_level: The std err level to use.
    :return: None
    """
    setup_logger(log_level=log_level, stderr_level=stderr_level)

    app = QtWidgets.QApplication([])

    csv_player = CSVPlayer(csv_path=RecordedSensorData.FILEPATH)
    controller = ExoController()
    timer = QtCore.QTimer()

    def update():
        if not csv_player.has_next_line():
            timer.stop()
            return

        sensordata = csv_player.get_sensor_data_from_csv()

        # setInterval in miliseconds. Update each 10ms
        timer.setInterval(10)

        controller.step(sensordata=sensordata)

    def sigint_handler(signal, frame):
        logger.success("Keyboard interrupted with ^C.")
        timer.stop()
        app.quit()

    timer.timeout.connect(slot=update)
    signal.signal(signal.SIGINT, sigint_handler)
    timer.start(0)
    app.exec()


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
