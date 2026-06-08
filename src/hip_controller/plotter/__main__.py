"""Command-line entry point for the modular CSV plotter.

Usage::

    python -m hip_controller.plotter path/to/file.csv [--frequency 100]
    python -m hip_controller.plotter path/to/file.csv --no-time-only-zoom
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hip_controller.definitions import BasicConfig
from hip_controller.plotter.csv_inspector import plot


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and launch the CSV inspector.

    :param argv: optional argument list (defaults to ``sys.argv[1:]``); exposed
        to make the entry point easy to drive from tests.
    :type argv: list[str] or None
    :return: process exit code (always 0 once the GUI window closes).
    :rtype: int
    """
    parser = argparse.ArgumentParser(
        prog="python -m hip_controller.plotter",
        description=(
            "Modular CSV plotter (Simulink-Data-Inspector-style) for the "
            "hip-controller package. The X axis is synthesized from "
            "--frequency; no time column is required in the CSV."
        ),
    )
    parser.add_argument(
        "csv",
        type=Path,
        help="Path to a CSV file with a header row.",
    )
    parser.add_argument(
        "--frequency",
        type=int,
        default=BasicConfig.frequency,
        help=(
            "Sampling frequency in Hz used to synthesize the time axis. "
            f"Defaults to BasicConfig.frequency ({BasicConfig.frequency})."
        ),
    )
    parser.add_argument(
        "--no-time-only-zoom",
        dest="time_only_zoom",
        action="store_false",
        help=(
            "Start with both X and Y zoom enabled. By default the Y axis is "
            "locked and only the time axis responds to the mouse wheel."
        ),
    )
    parser.set_defaults(time_only_zoom=True)
    args = parser.parse_args(argv)

    plot(
        csv_path=args.csv,
        frequency_hz=args.frequency,
        time_only_zoom=args.time_only_zoom,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
