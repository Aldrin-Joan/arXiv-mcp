"""Top-level convenience package for arxiv-mcp.

This package forwards to the internal `src` server implementation.
"""

from src.console import entrypoint
from src.models import *
from src.arxiv_client import *
from src.pdf_fetcher import *
from src.pdf_parser import *
from src.context_builder import *
from src.mcp_server import *
import src as src

__all__ = [
    "entrypoint",
    "src",
]
