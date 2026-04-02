"""Hip flexion exosuit controller package.

Provides gait phase estimation, amplitude modulation, motor reference
generation, and signal processing for the fully actuated hip flexion exosuit.

The version number is automatically imported from the pyproject.toml file.
"""

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version("hip-controller")
except PackageNotFoundError:
    # this path leads to: src/hip_controller/__init__.py → src/ → repo_root/ → pyproject.toml
    pyproject = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
    with open(pyproject, "rb") as f:
        __version__ = dict(tomllib.load(f))["project"]["version"]
