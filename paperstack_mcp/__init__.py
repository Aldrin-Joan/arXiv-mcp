"""Auf alias package for setuptools/pyproject import compatibility.

This package is an alias to the legacy `arxiv_mcp` top-level package,
and provides a `paperstack_mcp` import path for installers.
"""

from arxiv_mcp import *  # noqa: F401,F403
from importlib.metadata import version, PackageNotFoundError

# keep the same version as pyproject
try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0"
