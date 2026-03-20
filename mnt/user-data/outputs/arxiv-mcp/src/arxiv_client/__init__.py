"""
arxiv_client — search and metadata retrieval from arXiv API.

Uses the official `arxiv` Python library and wraps it with:
  - retry logic (tenacity)
  - rate limiting
  - ID detection / validation
  - structured Pydantic output
"""
from __future__ import annotations

import asyncio
import re
import time
from typing import Optional

import arxiv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.models import (
    Author,
    PaperMetadata,
    SearchResult,
    MAX_RESULTS_DEFAULT,
    RATE_LIMIT_DELAY,
    MAX_RETRIES,
)
from src.logger import get_logger

log = get_logger("arxiv_client")

# arXiv ID pattern: YYMM.NNNNN  or  category/NNNNNNN
_ARXIV_ID_RE = re.compile(
    r"\b(?:arxiv:)?(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)\b",
    re.IGNORECASE,
)

_last_call_time: float = 0.0
_rate_lock = asyncio.Lock()


async def _rate_limit() -> None:
    """Enforce minimum delay between arXiv API calls."""
    global _last_call_time
    async with _rate_lock:
        elapsed = time.monotonic() - _last_call_time
        if elapsed < RATE_LIMIT_DELAY:
            await asyncio.sleep(RATE_LIMIT_DELAY - elapsed)
        _last_call_time = time.monotonic()


def detect_arxiv_id(text: str) -> Optional[str]:
    """
    Extract the first arXiv ID from free text.
    Returns the canonical ID (without 'arxiv:' prefix, without version).
    """
    match = _ARXIV_ID_RE.search(text)
    if match:
        raw = match.group(1)
        # Strip version suffix for canonical ID
        canonical = re.sub(r"v\d+$", "", raw, flags=re.IGNORECASE)
        log.debug("Detected arXiv ID", raw=raw, canonical=canonical)
        return canonical
    return None


def _result_to_metadata(result: arxiv.Result) -> PaperMetadata:
    """Convert an arxiv.Result object to our PaperMetadata model."""
    arxiv_id = result.entry_id.split("/abs/")[-1]
    # Strip version from stored ID
    arxiv_id = re.sub(r"v\d+$", "", arxiv_id)

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    return PaperMetadata(
        arxiv_id=arxiv_id,
        title=result.title.strip(),
        authors=[Author(name=a.name) for a in result.authors],
        abstract=result.summary.strip(),
        categories=result.categories,
        primary_category=result.primary_category,
        published=result.published.isoformat() if result.published else "",
        updated=result.updated.isoformat() if result.updated else "",
        pdf_url=pdf_url,
        entry_url=result.entry_id,
        comment=result.comment,
        journal_ref=result.journal_ref,
    )


def _result_to_search_result(result: arxiv.Result) -> SearchResult:
    meta = _result_to_metadata(result)
    snippet = meta.abstract[:300] + "..." if len(meta.abstract) > 300 else meta.abstract
    return SearchResult(
        arxiv_id=meta.arxiv_id,
        title=meta.title,
        authors=[a.name for a in meta.authors],
        abstract_snippet=snippet,
        published=meta.published,
        categories=meta.categories,
        pdf_url=meta.pdf_url,
    )


class ArxivClient:
    """
    High-level async wrapper around the arxiv Python library.

    All public methods are async coroutines and respect rate limits.
    """

    def __init__(self) -> None:
        self._client = arxiv.Client(
            page_size=50,
            delay_seconds=RATE_LIMIT_DELAY,
            num_retries=MAX_RETRIES,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        max_results: int = MAX_RESULTS_DEFAULT,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
    ) -> list[SearchResult]:
        """
        Search arXiv with a natural language query.

        Automatically detects if the query contains an arXiv ID and routes
        to `get_by_id` for precise retrieval.
        """
        detected_id = detect_arxiv_id(query)
        if detected_id:
            log.info("Query contains arXiv ID; routing to get_by_id", id=detected_id)
            meta = await self.get_by_id(detected_id)
            if meta:
                return [_result_to_search_result_from_meta(meta)]
            return []

        log.info("Searching arXiv", query=query, max_results=max_results)
        await _rate_limit()

        search_obj = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_by,
        )

        results: list[SearchResult] = []
        try:
            for result in self._client.results(search_obj):
                results.append(_result_to_search_result(result))
        except Exception as exc:
            log.error("arXiv search failed", error=str(exc))
            raise

        log.info("Search complete", count=len(results))
        return results

    async def get_by_id(self, arxiv_id: str) -> Optional[PaperMetadata]:
        """
        Fetch full metadata for a single paper by arXiv ID.
        Returns None if the paper is not found.
        """
        clean_id = re.sub(r"v\d+$", "", arxiv_id.strip())
        log.info("Fetching paper by ID", arxiv_id=clean_id)
        await _rate_limit()

        search_obj = arxiv.Search(id_list=[clean_id])
        try:
            results = list(self._client.results(search_obj))
        except Exception as exc:
            log.error("Failed to fetch paper", arxiv_id=clean_id, error=str(exc))
            raise

        if not results:
            log.warning("Paper not found", arxiv_id=clean_id)
            return None

        meta = _result_to_metadata(results[0])
        log.info("Paper fetched", title=meta.title, arxiv_id=meta.arxiv_id)
        return meta

    async def validate_id(self, arxiv_id: str) -> bool:
        """Return True if the arXiv ID resolves to a real paper."""
        result = await self.get_by_id(arxiv_id)
        return result is not None


def _result_to_search_result_from_meta(meta: PaperMetadata) -> SearchResult:
    snippet = meta.abstract[:300] + "..." if len(meta.abstract) > 300 else meta.abstract
    return SearchResult(
        arxiv_id=meta.arxiv_id,
        title=meta.title,
        authors=[a.name for a in meta.authors],
        abstract_snippet=snippet,
        published=meta.published,
        categories=meta.categories,
        pdf_url=meta.pdf_url,
    )
