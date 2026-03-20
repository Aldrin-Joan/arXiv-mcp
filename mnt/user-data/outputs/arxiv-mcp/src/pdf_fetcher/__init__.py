"""
pdf_fetcher — download arXiv PDFs with retry, error handling, and caching.

Downloads PDFs to DOWNLOAD_DIR and caches them by arXiv ID to avoid
redundant network requests.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from src.models import DownloadResult, DOWNLOAD_DIR, HTTP_TIMEOUT, MAX_RETRIES
from src.logger import get_logger
import logging

log = get_logger("pdf_fetcher")

_HEADERS = {
    "User-Agent": "arxiv-mcp/1.0 (research tool; https://github.com/arxiv-mcp) Python/httpx",
}


def _pdf_path(arxiv_id: str) -> Path:
    """Return local path for a cached PDF."""
    safe_id = arxiv_id.replace("/", "_")
    return DOWNLOAD_DIR / f"{safe_id}.pdf"


def _is_cached(arxiv_id: str) -> bool:
    path = _pdf_path(arxiv_id)
    return path.exists() and path.stat().st_size > 1024  # >1KB = valid PDF


class PDFFetcher:
    """
    Downloads PDFs from arXiv and caches them locally.

    Uses httpx for async HTTP with automatic retry on transient failures.
    """

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            headers=_HEADERS,
            timeout=HTTP_TIMEOUT,
            follow_redirects=True,
        )

    async def __aenter__(self) -> "PDFFetcher":
        return self

    async def __aexit__(self, *args) -> None:
        await self._client.aclose()

    async def download(self, arxiv_id: str, force: bool = False) -> DownloadResult:
        """
        Download the PDF for a given arXiv ID.

        Args:
            arxiv_id: The arXiv paper ID (e.g., "2603.17216").
            force: If True, re-download even if cached.

        Returns:
            DownloadResult with local path and metadata.
        """
        local_path = _pdf_path(arxiv_id)

        if not force and _is_cached(arxiv_id):
            log.info("PDF already cached", arxiv_id=arxiv_id, path=str(local_path))
            return DownloadResult(
                arxiv_id=arxiv_id,
                local_path=str(local_path),
                file_size_bytes=local_path.stat().st_size,
                success=True,
            )

        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        log.info("Downloading PDF", arxiv_id=arxiv_id, url=pdf_url)

        try:
            result = await self._download_with_retry(arxiv_id, pdf_url, local_path)
            return result
        except Exception as exc:
            log.error("PDF download failed", arxiv_id=arxiv_id, error=str(exc))
            return DownloadResult(
                arxiv_id=arxiv_id,
                local_path="",
                file_size_bytes=0,
                success=False,
                error=str(exc),
            )

    async def _download_with_retry(
        self, arxiv_id: str, url: str, dest: Path
    ) -> DownloadResult:
        """Inner download with exponential backoff retry."""
        last_error: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with self._client.stream("GET", url) as response:
                    if response.status_code == 404:
                        raise FileNotFoundError(
                            f"PDF not found on arXiv (404): {url}"
                        )
                    response.raise_for_status()

                    content_type = response.headers.get("content-type", "")
                    if "pdf" not in content_type and "octet-stream" not in content_type:
                        # arXiv sometimes returns HTML for very new papers
                        raise ValueError(
                            f"Unexpected content-type '{content_type}' — PDF not ready yet"
                        )

                    with dest.open("wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            f.write(chunk)

                file_size = dest.stat().st_size
                log.info(
                    "PDF downloaded",
                    arxiv_id=arxiv_id,
                    size_kb=file_size // 1024,
                    path=str(dest),
                )
                return DownloadResult(
                    arxiv_id=arxiv_id,
                    local_path=str(dest),
                    file_size_bytes=file_size,
                    success=True,
                )

            except FileNotFoundError:
                raise  # don't retry 404s
            except Exception as exc:
                last_error = exc
                wait = 2**attempt
                log.warning(
                    "Download attempt failed",
                    attempt=attempt,
                    max=MAX_RETRIES,
                    wait=wait,
                    error=str(exc),
                )
                if attempt < MAX_RETRIES:
                    import asyncio
                    await asyncio.sleep(wait)

        raise RuntimeError(
            f"PDF download failed after {MAX_RETRIES} attempts: {last_error}"
        )

    async def aclose(self) -> None:
        await self._client.aclose()
