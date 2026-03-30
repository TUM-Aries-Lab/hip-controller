"""Import version number automatically."""

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path  # pragma no cover

try:
    __version__ = version("hip-controller")
except PackageNotFoundError:
    pyproject = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
    with open(pyproject, "rb") as f:
        __version__ = dict(tomllib.load(f))["project"]["version"]
