from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List
from urllib.parse import urlparse

import httpx
import fitz
from bs4 import BeautifulSoup

from src.models import (
    CodeLink,
    CodeLinks,
    DOWNLOAD_DIR,
    HTTP_TIMEOUT,
    LINK_CACHE_TTL,
)
from src.pdf_fetcher import PDFFetcher
from src.pdf_parser import PDFParser as PDFParserClass
from src.logger import get_logger

log = get_logger("link_extractor")

_USER_AGENT = "arxiv-mcp/1.0 (research tool; https://github.com/arxiv-mcp) Python/httpx"


class LinkExtractor:
    """Extract links from PapersWithCode, arXiv page, and PDF for an arXiv paper."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            headers={"User-Agent": _USER_AGENT},
            timeout=HTTP_TIMEOUT,
            follow_redirects=True,
        )
        self._cache_dir = Path(DOWNLOAD_DIR) / "links"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    async def aclose(self) -> None:
        """Close underlying HTTP client."""
        await self._client.aclose()

    async def _fetch_paperswithcode(self, arxiv_id: str) -> List[CodeLink]:
        """Fetch structured code and dataset links from PapersWithCode API."""
        url = f"https://paperswithcode.com/api/v1/papers/?arxiv_id={arxiv_id}"
        try:
            resp = await self._client.get(url)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", []) if isinstance(data, dict) else []
            if not results:
                return []
            entry = results[0]
            links: List[CodeLink] = []
            for repo in entry.get("repositories", []) or []:
                repo_url = repo.get("url") or repo.get("github_url")
                if not repo_url:
                    continue
                link_type = self._classify_url(repo_url)
                links.append(
                    CodeLink(
                        url=repo_url,
                        link_type=link_type,
                        source="paperswithcode",
                        confidence=1.0,
                    )
                )

            for ds in entry.get("datasets", []) or []:
                ds_url = ds.get("url")
                if not ds_url:
                    continue
                link_type = self._classify_url(ds_url)
                links.append(
                    CodeLink(
                        url=ds_url,
                        link_type=link_type,
                        source="paperswithcode",
                        confidence=1.0,
                    )
                )

            return links

        except Exception as exc:
            log.warning(
                "Failed PapersWithCode fetch", arxiv_id=arxiv_id, error=str(exc)
            )
            return []

    async def _fetch_arxiv_page_links(self, arxiv_id: str) -> List[CodeLink]:
        """Scrape the arXiv abstract page for code/dataset links."""
        url = f"https://arxiv.org/abs/{arxiv_id}"
        try:
            resp = await self._client.get(url)
            if resp.status_code != 200:
                return []
            soup = BeautifulSoup(resp.text, "lxml")
            links: List[CodeLink] = []
            seen: set[str] = set()
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if not href:
                    continue
                normalized = self._normalize_url(href)
                if normalized in seen:
                    continue
                seen.add(normalized)
                if any(
                    x in href
                    for x in [
                        "github.com",
                        "huggingface.co",
                        "kaggle.com/datasets",
                        "colab.research.google.com",
                    ]
                ):
                    link_type = self._classify_url(href)
                    links.append(
                        CodeLink(
                            url=href,
                            link_type=link_type,
                            source="arxiv_page",
                            confidence=0.9,
                        )
                    )
            return links
        except Exception as exc:
            log.warning("Failed arXiv page fetch", arxiv_id=arxiv_id, error=str(exc))
            return []

    async def _extract_pdf_links(self, arxiv_id: str) -> List[CodeLink]:
        """Extract hyperlinks and text URLs from PDF via PDFFetcher and PDFParser."""
        pdf_links: List[CodeLink] = []

        async with PDFFetcher() as fetcher:
            download_result = await fetcher.download(arxiv_id, force=False)
            if not download_result.success:
                return []

            path = Path(download_result.local_path)

            # hyperlinks embedded in PDF pages
            try:
                doc = fitz.open(str(path))
                for page in doc:
                    for link in page.get_links():
                        uri = link.get("uri") or link.get("uri", "")
                        if not uri:
                            continue
                        link_type = self._classify_url(uri)
                        pdf_links.append(
                            CodeLink(
                                url=uri,
                                link_type=link_type,
                                source="pdf_hyperlink",
                                confidence=0.9,
                            )
                        )
                doc.close()
            except Exception as exc:
                log.warning(
                    "PDF hyperlink extraction failed", arxiv_id=arxiv_id, error=str(exc)
                )

        # text-based regex extraction
        try:
            extracted = PDFParserClass().parse(str(path), arxiv_id)
            text = extracted.full_text
            matches: dict[str, set[str]] = {
                "github_repo": set(
                    re.findall(
                        r"https?://(?:www\.)?github\.com/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+",
                        text,
                        re.IGNORECASE,
                    )
                ),
                "huggingface_model": set(
                    re.findall(
                        r"https?://(?:www\.)?huggingface\.co/[A-Za-z0-9_\-]+(?:/[A-Za-z0-9_\-]+)/?",
                        text,
                        re.IGNORECASE,
                    )
                ),
                "kaggle_dataset": set(
                    re.findall(
                        r"https?://(?:www\.)?kaggle\.com/datasets/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+",
                        text,
                        re.IGNORECASE,
                    )
                ),
            }
            for typ, urls in matches.items():
                for url in urls:
                    pdf_links.append(
                        CodeLink(
                            url=url,
                            link_type=typ,
                            source="pdf_text",
                            confidence=0.7,
                        )
                    )
        except Exception as exc:
            log.warning(
                "PDF text extraction for links failed",
                arxiv_id=arxiv_id,
                error=str(exc),
            )

        return pdf_links

    def _classify_url(self, url: str) -> str:
        """Classify URL into the charted link_type labels."""
        normalized = self._normalize_url(url)
        parsed = urlparse(normalized)
        host = parsed.netloc.lower()
        path = parsed.path.strip("/")

        if host.endswith("github.com"):
            parts = path.split("/") if path else []
            if len(parts) >= 2:
                return "github_repo"

        if "huggingface.co" in host:
            if "/datasets/" in parsed.path:
                return "huggingface_dataset"
            if "/spaces/" in parsed.path:
                return "demo"
            if path:
                return "huggingface_model"

        if "kaggle.com" in host and parsed.path.startswith("/datasets/"):
            return "kaggle_dataset"

        if any(x in host for x in ["github.com", "huggingface.co", "kaggle.com"]):
            return "project_page"

        if "demo" in parsed.path or "spaces" in parsed.path:
            return "demo"

        if not parsed.path or parsed.path == "/":
            return "project_page"

        return "other"

    def _validate_github_url(self, url: str) -> bool:
        """Validate Github URL is owner/repo, not issue/PR/profile only."""
        normalized = self._normalize_url(url)
        parsed = urlparse(normalized)
        if not parsed.netloc.lower().endswith("github.com"):
            return False
        parts = [p for p in parsed.path.strip("/").split("/") if p]
        if len(parts) != 2:
            return False
        blacklist = {"issues", "pulls", "actions", "wiki", "releases"}
        if parts[1].lower() in blacklist or parts[0].lower() in blacklist:
            return False
        return True

    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication: lowercase scheme/netloc, strip www and trailing slash."""
        url = url.strip()
        if not url:
            return ""
        parsed = urlparse(url)
        if not parsed.scheme:
            parsed = urlparse("https://" + url)

        netloc = parsed.netloc.lower().lstrip("www.")
        path = parsed.path.rstrip("/")
        normalized = f"{parsed.scheme.lower()}://{netloc}{path}"
        return normalized

    def _deduplicate(self, links: Iterable[CodeLink]) -> List[CodeLink]:
        """Deduplicate links by normalized URL preserving highest confidence."""
        best: dict[str, CodeLink] = {}
        for link in links:
            if not link.url:
                continue
            normalized = self._normalize_url(link.url)
            if not normalized:
                continue
            existing = best.get(normalized)
            if existing is None or link.confidence > existing.confidence:
                best[normalized] = link
        return list(best.values())

    def _cache_path(self, arxiv_id: str) -> Path:
        return self._cache_dir / f"{arxiv_id}.json"

    def _is_cache_valid(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            fetched = data.get("fetched_at")
            if not fetched:
                return False
            fetched_dt = datetime.fromisoformat(fetched)
            return datetime.utcnow() - fetched_dt < timedelta(seconds=LINK_CACHE_TTL)
        except Exception:
            return False

    async def extract(self, arxiv_id: str, force_refresh: bool = False) -> CodeLinks:
        """Main API: extract and return classified links for an arXiv ID."""
        cache_path = self._cache_path(arxiv_id)
        if not force_refresh and self._is_cache_valid(cache_path):
            try:
                saved = json.loads(cache_path.read_text(encoding="utf-8"))
                return CodeLinks.model_validate(saved)
            except Exception as exc:
                log.warning("Cache load failed", path=str(cache_path), error=str(exc))

        pwr_links, page_links, pdf_links = await asyncio.gather(
            self._fetch_paperswithcode(arxiv_id),
            self._fetch_arxiv_page_links(arxiv_id),
            self._extract_pdf_links(arxiv_id),
        )

        all_links = self._deduplicate(pwr_links + page_links + pdf_links)

        github_repos = [ln for ln in all_links if ln.link_type == "github_repo"]
        huggingface_links = [
            ln for ln in all_links if ln.link_type == "huggingface_model"
        ]
        dataset_links = [
            ln
            for ln in all_links
            if ln.link_type in ("huggingface_dataset", "kaggle_dataset")
        ]
        project_pages = [ln for ln in all_links if ln.link_type == "project_page"]
        other_links = [
            ln for ln in all_links if ln.link_type == "other" or ln.link_type == "demo"
        ]

        has_official_code = any(
            repo.source == "paperswithcode" and repo.link_type == "github_repo"
            for repo in pwr_links
        )
        report = CodeLinks(
            arxiv_id=arxiv_id,
            github_repos=github_repos,
            huggingface_links=huggingface_links,
            dataset_links=dataset_links,
            project_pages=project_pages,
            other_links=other_links,
            has_official_code=has_official_code,
            fetched_at=datetime.utcnow(),
        )

        try:
            cache_path.write_text(
                report.model_dump_json(by_alias=True, exclude_none=True),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("Cache write failed", path=str(cache_path), error=str(exc))

        return report
