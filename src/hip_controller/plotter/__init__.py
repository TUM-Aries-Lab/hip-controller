"""Plotting utilities for the hip controller package."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    # Import-time only for type checkers; at runtime __getattr__ below
    # resolves `plot` lazily so Qt stays optional.
    from hip_controller.plotter.csv_inspector import plot

__all__ = ["plot"]


def __getattr__(name: str) -> Any:
    """Resolve ``plot`` lazily so importing this package does not require Qt.

    ``csv_inspector`` defines Qt widget subclasses at module level, so importing
    it needs a working PyQt6 install. Eagerly importing it here meant that
    ``hip_controller.plotter.csv_player`` -- which is pure pandas -- could not be
    imported on a headless machine either. Deferring keeps the GUI entry point
    available without imposing Qt on the data-only modules.

    :param str name: attribute being looked up on the package.
    :return: the requested attribute.
    :rtype: Any
    :raises AttributeError: if ``name`` is not exported by this package.
    """
    if name == "plot":
        from hip_controller.plotter.csv_inspector import plot

        return plot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
