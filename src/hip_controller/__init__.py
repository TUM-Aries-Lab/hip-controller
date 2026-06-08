"""Hip flexion exosuit controller package.

Provides gait phase estimation, amplitude modulation, motor reference
generation, and signal processing for the fully actuated hip flexion exosuit.

The version number is automatically imported from the pyproject.toml file.
"""

import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError as err:
        raise ImportError(
            "Python 3.10 requires the 'tomli' package: pip install tomli"
        ) from err
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version("hip-controller")
except PackageNotFoundError:
    try:
        # this path leads to: src/hip_controller/__init__.py → src/ → repo_root/ → pyproject.toml
        pyproject = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
        with open(pyproject, "rb") as f:
            __version__ = dict(tomllib.load(f))["project"]["version"]
    except FileNotFoundError:
        __version__ = "0.0.0+unknown"
