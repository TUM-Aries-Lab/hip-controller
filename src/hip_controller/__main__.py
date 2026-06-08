"""Main function of the controller.

Reads data from a CSV file with CSV player with timestamp and compute the controller steps.
"""

import argparse  # pragma: no cover
import signal  # pragma: no cover
from pathlib import Path  # pragma: no cover

from loguru import logger
from pandas import DataFrame
from pyqtgraph import QtCore, QtWidgets  # pragma: no cover

from hip_controller.control.app import WalkOnController
from hip_controller.control.motor_reference_control.amplitude_modulation import (
    AscendStairsMode,
    DescendStairsMode,
    LevelGroundMode,
    ModeStrategy,
)
from hip_controller.definitions import (
    DEFAULT_LOG_LEVEL,
    BasicConfig,
    LogLevel,
    RecordedSensorData,
)  # pragma: no cover
from hip_controller.plotter.csv_inspector import plot as csv_inspector_plot
from hip_controller.plotter.csv_player import CSVPlayer
from hip_controller.utils.utils import setup_logger

MOTOR_LEFT_COLUMN = "motor_command_left (rad)"
MOTOR_RIGHT_COLUMN = "motor_command_right (rad)"
FILTERED_ANG_LEFT_COLUMN = "filtered_angle_left (rad)"
FILTERED_ANG_RIGHT_COLUMN = "filtered_angle_right (rad)"
FILTERED_VEL_LEFT_COLUMN = "filtered_vel_left (rad/s)"
FILTERED_VEL_RIGHT_COLUMN = "filtered_vel_right (rad/s)"
PORTRAIT_RADIUS_LEFT_COLUMN = "Portrait Radius Left"
PORTRAIT_RADIUS_RIGHT_COLUMN = "Portrait Radius Right"
SCALED_PORTRAIT_RADIUS_LEFT_COLUMN = "Scaled Portrait Radius Left"
SCALED_PORTRAIT_RADIUS_RIGHT_COLUMN = "Scaled Portrait Radius Right"
SIGMOID_SCALING_LEFT_COLUMN = "Sigmoid Scaling Left"
SIGMOID_SCALING_RIGHT_COLUMN = "Sigmoid Scaling Right"
SCALED_SIGMOID_SCALING_LEFT_COLUMN = "Scaled Sigmoid Scaling Left"
SCALED_SIGMOID_SCALING_RIGHT_COLUMN = "Scaled Sigmoid Scaling Right"
AMPLITUDE_LEFT_COLUMN = "Amplitude Left"
AMPLITUDE_RIGHT_COLUMN = "Amplitude Right"
GAIT_PHASE_LEFT_COLUMN = "Gait Phase Left (rad)"
GAIT_PHASE_RIGHT_COLUMN = "Gait Phase Right (rad)"
MOTION_MAPPING_LEFT_COLUMN = "Motion Mapping Left"
MOTION_MAPPING_RIGHT_COLUMN = "Motion Mapping Right"
VELOCITY_SURROGATE_LEFT_COLUMN = "Velocity Surrogate Left (rad/s)"
VELOCITY_SURROGATE_RIGHT_COLUMN = "Velocity Surrogate Right (rad/s)"
VELOCITY_LPF_ANGLE_LEFT_COLUMN = "Velocity-LPF Angle Left (rad)"
VELOCITY_LPF_ANGLE_RIGHT_COLUMN = "Velocity-LPF Angle Right (rad)"
DRIFT_REMOVED_ANGLE_LEFT_COLUMN = "Drift-Removed Angle Left (rad)"
DRIFT_REMOVED_ANGLE_RIGHT_COLUMN = "Drift-Removed Angle Right (rad)"

# Integer classification -> locomotion mode. Instances are cached so we don't
# rebuild a ModeStrategy on every sample. Unknown values fall back to Level
# Ground (the safe default that matches pre-classification behavior).
_MODES_BY_CLASSIFICATION: dict[int, ModeStrategy] = {
    0: LevelGroundMode(),
    1: AscendStairsMode(),
    2: DescendStairsMode(),
}


def _mode_for(classification: int) -> ModeStrategy:
    """Map a classification integer to its locomotion-mode strategy."""
    return _MODES_BY_CLASSIFICATION.get(classification, _MODES_BY_CLASSIFICATION[0])


def main(  # noqa: PLR0915, C901
    log_level: str = DEFAULT_LOG_LEVEL,
    stderr_level: str = DEFAULT_LOG_LEVEL,
    csv_path: Path = BasicConfig.read_data_from_path,
    fast: bool = False,
) -> None:  # pragma: no cover
    """Run the main pipeline.

    :param log_level: The log level to use.
    :param stderr_level: The std err level to use.
    :param str csv_path: Path to the CSV file used for simulated real-time playback. The user could pass in the path of a file as well.
    :param bool fast: When True, skip the live phase-portrait plots, process every CSV
        row as fast as Python can, then open the resulting output CSV in the
        :func:`hip_controller.plotter.csv_inspector.plot` window. When False
        (default), runs in real-time with the live plot windows.
    :return: None

    # Example
        -  simulate_controller_with_data(log_level=log_level, stderr_level=stderr_level, csv_path=BasicConfig.read_data_from_path)
    """
    setup_logger(log_level=log_level, stderr_level=stderr_level)

    # QApplication is created unconditionally: fast mode still needs one for the
    # csv_inspector_plot() call at the end, and live mode needs one for the
    # phase-portrait windows. Reusing a single instance avoids a second
    # QApplication construction inside csv_inspector_plot.
    app = QtWidgets.QApplication([])

    player = CSVPlayer(csv_path)
    plot = not fast
    config = BasicConfig(
        filtered=False, left_limb_plot=plot, right_limb_plot=plot
    )

    controller_left = WalkOnController(left_limb=True, config=config)
    controller_right = WalkOnController(left_limb=False, config=config)
    
    timer = QtCore.QTimer()

    # Track the previous main-switch state so we can reset the controllers on a
    # falling edge (1 -> 0). The reset puts the preprocessor back into its
    # "first call" state so that, on the next rising edge, velocity derivation
    # starts fresh from the raw angle.
    state = {"prev_switch": False}

    # Buffer of per-sample rows (inputs + motor commands) written to disk when
    # playback finishes or the user interrupts with Ctrl+C. Values are mostly
    # floats; main_switch and classification_* are ints, hence the wider type.
    output_rows: list[dict[str, float | int]] = []
    output_path = csv_path.with_name(f"{csv_path.stem}_output.csv").resolve()
    logger.info(f"Simulation results will be written to '{output_path}'.")

    def save_results() -> None:
        """Persist the accumulated input/output rows to a CSV next to the input file."""
        if not output_rows:
            return
        DataFrame(output_rows).to_csv(output_path, index=False)
        logger.success(f"Saved {len(output_rows)} simulation rows to '{output_path}'.")

    def process_step() -> bool:
        """Pull one row from the CSV, run the controllers, append to ``output_rows``.

        :return: ``True`` if a row was processed, ``False`` at end-of-file.
        :rtype: bool
        """
        if not player.has_next_line():
            return False

        step = player.get_sensor_data_from_csv()
        sensor_data = step.sensor_data
        main_switch = step.main_switch

        controller_left.amplitude_modulation.set_mode(
            _mode_for(step.classification_left)
        )
        controller_right.amplitude_modulation.set_mode(
            _mode_for(step.classification_right)
        )

        if main_switch:
            motor_command_left = controller_left.step(sensor_data.left)
            motor_command_right = controller_right.step(sensor_data.right)
        else:
            if state["prev_switch"]:
                controller_left.reset()
                controller_right.reset()
            motor_command_left = 0.0
            motor_command_right = 0.0

        state["prev_switch"] = main_switch

        # last_filtered_signal / last_intermediates / etc. are None before the
        # first step or after a reset; write NaN in those cases via float('nan')
        # so downstream consumers can distinguish "no value" from a real zero.
        # Locals (rather than chained attribute access) so pyright can narrow
        # the Optionals reliably when building the row dict below.
        filt_left = controller_left.last_filtered_signal
        filt_right = controller_right.last_filtered_signal
        amp_left = controller_left.amplitude_modulation.last_intermediates
        amp_right = controller_right.amplitude_modulation.last_intermediates
        pre_left = controller_left.pre_processor
        pre_right = controller_right.pre_processor
        vel_surrogate_left = pre_left.last_velocity_surrogate_rad_per_sec
        vel_surrogate_right = pre_right.last_velocity_surrogate_rad_per_sec
        vel_lpf_left = pre_left.last_velocity_lpf_angle_rad
        vel_lpf_right = pre_right.last_velocity_lpf_angle_rad
        drift_left = pre_left.last_drift_removed_angle_rad
        drift_right = pre_right.last_drift_removed_angle_rad
        gait_phase_left = controller_left.last_gait_phase_rad
        gait_phase_right = controller_right.last_gait_phase_rad
        mapping_left = controller_left.motion_reference_controller.last_mapping_value
        mapping_right = controller_right.motion_reference_controller.last_mapping_value
        nan = float("nan")
        # SensorSignal.timestamp is Optional in the dataclass; CSVPlayer always
        # synthesizes one if the column is absent, so this is effectively never
        # None in practice — but pyright can't see that.
        timestamp_value = (
            sensor_data.left.timestamp
            if sensor_data.left.timestamp is not None
            else nan
        )

        output_rows.append(
            {
                RecordedSensorData.timestamp: timestamp_value,
                RecordedSensorData.ang_left: sensor_data.left.angle_rad,
                RecordedSensorData.ang_right: sensor_data.right.angle_rad,
                RecordedSensorData.vel_left: sensor_data.left.velocity_rad_per_sec,
                RecordedSensorData.vel_right: sensor_data.right.velocity_rad_per_sec,
                RecordedSensorData.main_switch: int(main_switch),
                "classification_left": step.classification_left,
                "classification_right": step.classification_right,
                FILTERED_ANG_LEFT_COLUMN: filt_left.angle_rad if filt_left else nan,
                FILTERED_VEL_LEFT_COLUMN: (
                    filt_left.velocity_rad_per_sec if filt_left else nan
                ),
                FILTERED_ANG_RIGHT_COLUMN: (
                    filt_right.angle_rad if filt_right else nan
                ),
                FILTERED_VEL_RIGHT_COLUMN: (
                    filt_right.velocity_rad_per_sec if filt_right else nan
                ),
                VELOCITY_SURROGATE_LEFT_COLUMN: (
                    vel_surrogate_left if vel_surrogate_left is not None else nan
                ),
                VELOCITY_SURROGATE_RIGHT_COLUMN: (
                    vel_surrogate_right if vel_surrogate_right is not None else nan
                ),
                VELOCITY_LPF_ANGLE_LEFT_COLUMN: (
                    vel_lpf_left if vel_lpf_left is not None else nan
                ),
                VELOCITY_LPF_ANGLE_RIGHT_COLUMN: (
                    vel_lpf_right if vel_lpf_right is not None else nan
                ),
                DRIFT_REMOVED_ANGLE_LEFT_COLUMN: (
                    drift_left if drift_left is not None else nan
                ),
                DRIFT_REMOVED_ANGLE_RIGHT_COLUMN: (
                    drift_right if drift_right is not None else nan
                ),
                PORTRAIT_RADIUS_LEFT_COLUMN: (
                    amp_left.portrait_radius if amp_left else nan
                ),
                PORTRAIT_RADIUS_RIGHT_COLUMN: (
                    amp_right.portrait_radius if amp_right else nan
                ),
                SCALED_PORTRAIT_RADIUS_LEFT_COLUMN: (
                    amp_left.scaled_portrait_radius if amp_left else nan
                ),
                SCALED_PORTRAIT_RADIUS_RIGHT_COLUMN: (
                    amp_right.scaled_portrait_radius if amp_right else nan
                ),
                SIGMOID_SCALING_LEFT_COLUMN: (
                    amp_left.sigmoid_scaling if amp_left else nan
                ),
                SIGMOID_SCALING_RIGHT_COLUMN: (
                    amp_right.sigmoid_scaling if amp_right else nan
                ),
                SCALED_SIGMOID_SCALING_LEFT_COLUMN: (
                    amp_left.scaled_sigmoid_scaling if amp_left else nan
                ),
                SCALED_SIGMOID_SCALING_RIGHT_COLUMN: (
                    amp_right.scaled_sigmoid_scaling if amp_right else nan
                ),
                AMPLITUDE_LEFT_COLUMN: amp_left.amplitude if amp_left else nan,
                AMPLITUDE_RIGHT_COLUMN: amp_right.amplitude if amp_right else nan,
                GAIT_PHASE_LEFT_COLUMN: (
                    gait_phase_left if gait_phase_left is not None else nan
                ),
                GAIT_PHASE_RIGHT_COLUMN: (
                    gait_phase_right if gait_phase_right is not None else nan
                ),
                MOTION_MAPPING_LEFT_COLUMN: (
                    mapping_left if mapping_left is not None else nan
                ),
                MOTION_MAPPING_RIGHT_COLUMN: (
                    mapping_right if mapping_right is not None else nan
                ),
                MOTOR_LEFT_COLUMN: motor_command_left,
                MOTOR_RIGHT_COLUMN: motor_command_right,
            }
        )
        return True

    if fast:
        # Process every row as fast as Python allows, save once, then hand
        # off the result file to the CSV inspector for visual inspection.
        while process_step():
            pass
        save_results()
        logger.info("Opening result in CSV inspector.")
        csv_inspector_plot(output_path)
        return

    # Live mode: drive the controllers from a Qt timer so the plot windows
    # update in real time.
    def update() -> None:
        """Qt timer slot: process one row and reschedule the timer."""
        if not process_step():
            timer.stop()
            save_results()
            return
        # setInterval in miliseconds. Update each 10ms
        timer.setInterval(10)

    def sigint_handler(signal, frame) -> None:
        """Handle SIGINT (Ctrl+C) gracefully."""
        logger.success("Keyboard interrupted with ^C.")
        timer.stop()
        save_results()
        app.quit()

    timer.timeout.connect(slot=update)
    signal.signal(signal.SIGINT, sigint_handler)

    # Save results no matter how the app exits: end-of-CSV in update(),
    # Ctrl+C in sigint_handler, or the user closing the plot windows. The
    # aboutToQuit signal fires once at shutdown for all of these paths;
    # save_results is idempotent (returns early when output_rows is empty),
    # so duplicate calls from the EOF/Ctrl+C paths are harmless.
    app.aboutToQuit.connect(save_results)

    # PyQt's event loop is implemented in C and doesn't yield to the Python
    # interpreter often enough for signal handlers (Ctrl+C) to be delivered.
    # A no-op QTimer firing every 200 ms forces a return to Python so the
    # SIGINT handler installed above actually runs.
    keepalive = QtCore.QTimer()
    keepalive.timeout.connect(lambda: None)
    keepalive.start(200)

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

    parser.add_argument(
        "--file-path",
        "-p",
        default=Path(BasicConfig.read_data_from_path),
        help=(
            "Path to the CSV file used for simulated real-time playback. "
            "Required columns: 'angle_left (rad)', 'angle_right (rad)'. "
            "Optional columns: 'time (s)' (else synthesized from sample index), "
            "'main_switch' (0/1 per row; defaults to 1 when absent), "
            "'vel_left (rad/s)', 'vel_right (rad/s)' (else velocity is derived "
            "from the raw angle by the controller's preprocessor), "
            "'classification_left' / 'classification_right' (0=Level Ground, "
            "1=Ascend Stairs, 2=Descend Stairs; defaults to 0 when absent). "
            "Alternative header names are accepted, see CSVPlayer.COLUMN_ALIASES."
        ),
        required=False,
        type=Path,
    )
    parser.add_argument(
        "--fast",
        "-f",
        action="store_true",
        help=(
            "Run the simulation as fast as possible without the live phase-"
            "portrait plot windows, then open the result CSV in the inspector. "
            "Default (omitted) is live mode with real-time plots."
        ),
    )
    args = parser.parse_args()

    main(
        log_level=args.log_level,
        stderr_level=args.stderr_level,
        csv_path=args.file_path,
        fast=args.fast,
    )
