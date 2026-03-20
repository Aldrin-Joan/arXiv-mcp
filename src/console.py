from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from src.mcp_server import main


def configure_env_from_args(args: argparse.Namespace) -> None:
    if args.download_dir:
        os.environ["ARXIV_DOWNLOAD_DIR"] = str(Path(args.download_dir).expanduser().resolve())
    if args.keep_pdfs is not None:
        os.environ["ARXIV_KEEP_PDFS"] = "true" if args.keep_pdfs else "false"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="arxiv-mcp", description="arXiv MCP Tool Server")
    parser.add_argument("--download-dir", help="Directory to cache downloaded PDFs (default=env ARXIV_DOWNLOAD_DIR or temp)")
    parser.add_argument("--keep-pdfs", choices=["true", "false"], default=None, help="Control whether PDFs are kept after processing")
    parser.add_argument("--env", action="store_true", help="Print effective environment and exit")
    return parser.parse_args()


def entrypoint() -> None:
    args = parse_args()
    if args.env:
        print("ARXIV_DOWNLOAD_DIR=", os.getenv("ARXIV_DOWNLOAD_DIR"))
        print("ARXIV_KEEP_PDFS=", os.getenv("ARXIV_KEEP_PDFS"))
        return

    if args.keep_pdfs is not None:
        os.environ["ARXIV_KEEP_PDFS"] = args.keep_pdfs

    if args.download_dir:
        os.environ["ARXIV_DOWNLOAD_DIR"] = str(Path(args.download_dir).expanduser().resolve())

    asyncio.run(main())
