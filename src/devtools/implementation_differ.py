from __future__ import annotations

import asyncio
import concurrent.futures
import json
import re
from datetime import datetime, timedelta


def _run_sync(coro_or_func):
    if asyncio.iscoroutine(coro_or_func):
        coro = coro_or_func
    elif callable(coro_or_func):
        result = coro_or_func()
        if asyncio.iscoroutine(result):
            coro = result
        else:
            return result
    else:
        raise TypeError("_run_sync expects a coroutine or callable returning a coroutine")

    try:
        asyncio.get_running_loop()

        def _runner():
            return asyncio.run(coro)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(_runner).result()
    except RuntimeError:
        return asyncio.run(coro)
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import tiktoken

from src.intelligence.contribution_extractor import ContributionExtractor
from src.logger import get_logger
from src.models import (
    DIFF_CACHE_TTL,
    DOWNLOAD_DIR,
    GITHUB_MAX_FILE_SIZE_KB,
    GITHUB_MAX_FILES,
    GITHUB_TOKEN,
    HTTP_TIMEOUT,
    ImplementationDiff,
    ExtractedPaper,
)
from src.pdf_fetcher import PDFFetcher
from src.pdf_parser import PDFParser

log = get_logger("implementation_differ")


class GitHubFetcher:
    """Fetch GitHub repository metadata and source files for implementation diffing."""

    def __init__(self) -> None:
        headers: Dict[str, str] = {
            "Accept": "application/vnd.github.v3+json",
        }
        if GITHUB_TOKEN:
            headers["Authorization"] = f"token {GITHUB_TOKEN}"

        self._client = httpx.AsyncClient(
            base_url="https://api.github.com",
            headers=headers,
            timeout=HTTP_TIMEOUT,
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _parse_github_url(self, url: str) -> Tuple[str, str]:
        """Extract owner and repository name from a GitHub repo URL."""
        if url.endswith(".git"):
            url = url[:-4]

        m = re.match(
            r"https?://(?:www\.)?github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)(?:/.*)?$",
            url,
        )
        if not m:
            raise ValueError(f"Invalid GitHub URL: {url}")

        owner = m.group("owner")
        repo = m.group("repo")

        if not owner or not repo:
            raise ValueError(f"Invalid GitHub URL: {url}")

        return owner, repo

    async def _check_rate_limit(self, response: httpx.Response) -> None:
        remaining_header = response.headers.get("X-RateLimit-Remaining")
        reset_header = response.headers.get("X-RateLimit-Reset")

        if not remaining_header:
            return

        try:
            remaining = int(remaining_header)
        except ValueError:
            return

        if remaining >= 5:
            return

        if not reset_header:
            raise ValueError("GitHub rate limit near exhaustion and reset time unknown")

        try:
            reset_ts = int(reset_header)
        except ValueError:
            raise ValueError("GitHub rate limit reset header invalid")

        import time

        now_ts = int(time.time())
        sleep_duration = max(0, reset_ts - now_ts)

        if sleep_duration > 60:
            raise ValueError("GitHub rate limit reached; retry after reset")

        await asyncio.sleep(sleep_duration)

    async def _get_default_branch(self, owner: str, repo: str) -> str:
        """Fetch default branch name from GitHub repository metadata."""
        path = f"/repos/{owner}/{repo}"
        response = await self._client.get(path)

        await self._check_rate_limit(response)

        if response.status_code == 404:
            raise ValueError("Repository not found or not accessible")

        response.raise_for_status()
        data = response.json()

        branch = data.get("default_branch")
        if not branch or not isinstance(branch, str):
            raise ValueError("Could not determine default branch")

        return branch

    async def _get_file_tree(
        self, owner: str, repo: str, branch: str
    ) -> List[Dict[str, Any]]:
        """Return list of Python blob entries in the Git tree."""
        path = f"/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
        response = await self._client.get(path)

        await self._check_rate_limit(response)

        response.raise_for_status()
        body = response.json()

        tree = body.get("tree")
        if tree is None or not isinstance(tree, list):
            raise ValueError("Repository tree not available")

        return [
            item
            for item in tree
            if item.get("type") == "blob"
            and isinstance(item.get("path"), str)
            and item["path"].endswith(".py")
        ]

    def _select_files(self, tree: List[Dict[str, Any]]) -> List[str]:
        """Select and rank Python source files for analysis."""
        excluded_patterns = [
            r"(^|/)test_",
            r"(^|/)setup\.py$",
            r"(^|/)conf\.py$",
            r"(^|/)docs/",
            r"(^|/)\.venv/",
        ]
        keywords = ["train", "model", "arch", "network", "loss", "main"]

        candidates: List[Tuple[int, str]] = []

        for item in tree:
            path = item.get("path")
            if not isinstance(path, str):
                continue
            if any(re.search(expr, path) for expr in excluded_patterns):
                continue
            score = sum(path.lower().count(k) for k in keywords)
            candidates.append((score, path))

        candidates.sort(key=lambda x: (-x[0], x[1]))

        selected = [path for _, path in candidates[:GITHUB_MAX_FILES]]
        return selected

    async def _fetch_file(
        self, owner: str, repo: str, branch: str, path: str
    ) -> Optional[str]:
        """Fetch file content from raw.githubusercontent.com with size protection."""
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
        response = await self._client.get(url)

        if response.status_code != 200:
            return None

        content_length = response.headers.get("content-length")
        max_bytes = GITHUB_MAX_FILE_SIZE_KB * 1024
        if content_length is not None:
            try:
                if int(content_length) > max_bytes:
                    return None
            except ValueError:
                pass

        content = response.content
        if len(content) > max_bytes:
            return None

        try:
            return content.decode("utf-8", errors="replace")
        except Exception:
            return None

    async def fetch_repo_summary(self, github_url: str) -> Dict[str, Any]:
        """Fetch repo metadata and selected Python file contents."""
        owner, repo = self._parse_github_url(github_url)
        branch = await self._get_default_branch(owner, repo)
        tree = await self._get_file_tree(owner, repo, branch)
        selected_paths = self._select_files(tree)

        files: Dict[str, str] = {}
        for p in selected_paths:
            content = await self._fetch_file(owner, repo, branch, p)
            if content is not None:
                files[p] = content

        return {
            "files": files,
            "owner": owner,
            "repo": repo,
            "branch": branch,
            "total_files_in_repo": len(tree),
        }


class ImplementationDiffer:
    """Compare paper method text with repository code and derive divergences."""

    def __init__(self) -> None:
        self._cache_dir = Path(DOWNLOAD_DIR) / "diffs"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, arxiv_id: str, owner: str, repo: str) -> Path:
        safe_key = f"{arxiv_id}_{owner}_{repo}".replace("/", "_")
        return self._cache_dir / f"{safe_key}.json"

    def _is_cache_valid(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            diffed_at = raw.get("diffed_at")
            if not diffed_at:
                return False
            ts = datetime.fromisoformat(diffed_at)
            return datetime.utcnow() - ts < timedelta(seconds=DIFF_CACHE_TTL)
        except Exception:
            return False

    def _extract_method_section(self, paper: ExtractedPaper) -> str:
        relevant = [
            c.text
            for c in paper.chunks
            if c.section_hint
            and re.search(
                r"method|approach|model|architecture|system",
                c.section_hint,
                re.IGNORECASE,
            )
        ]

        if not relevant:
            relevant = [c.text for c in paper.chunks[:2]]

        joined = "\n\n".join(relevant)

        try:
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(joined)
            if len(tokens) > 2000:
                joined = enc.decode(tokens[:2000])
        except Exception:
            if len(joined) > 16000:
                joined = joined[:16000]

        return joined

    def _build_code_content(self, files: Dict[str, str]) -> Tuple[str, int]:
        prioritized_files = sorted(
            files.items(),
            key=lambda kv: -sum(
                kv[0].lower().count(tok)
                for tok in ["train", "model", "arch", "network", "loss", "main"]
            ),
        )

        code_content = ""
        for path, content in prioritized_files:
            code_content += f"# === {path} ===\n{content}\n\n"

        token_count = 0
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            token_count = len(enc.encode(code_content))
            if token_count > 15000:
                accum = 0
                tight_content = ""
                for path, content in prioritized_files:
                    block = f"# === {path} ===\n{content}\n\n"
                    block_tokens = len(enc.encode(block))
                    if accum + block_tokens > 15000:
                        break
                    accum += block_tokens
                    tight_content += block
                code_content = tight_content
                token_count = accum
        except Exception:
            token_count = len(code_content) // 4
            if token_count > 15000:
                code_content = code_content[: 15000 * 4]
                token_count = 15000

        return code_content, token_count

    def _build_prompt(
        self,
        contributions: Any,
        method_text: str,
        code_content: str,
        title: str,
    ) -> str:
        prompt_file = Path(__file__).resolve().parent / "prompts" / "diff.txt"
        if not prompt_file.exists():
            raise FileNotFoundError(f"diff prompt not found: {prompt_file}")

        template = prompt_file.read_text(encoding="utf-8")

        return (
            template.replace("{title}", title)
            .replace("{proposed_method}", getattr(contributions, "proposed_method", ""))
            .replace("{core_claim}", getattr(contributions, "core_claim", ""))
            .replace("{method_text}", method_text)
            .replace("{code_content}", code_content)
        )

    def _call_llm(self, prompt: str) -> str:
        try:
            return _run_sync(ContributionExtractor()._call_ollama(prompt))
        except Exception as exc:
            log.warning("LLM unavailable", error=str(exc))
            return "LLM unavailable"

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        try:
            return json.loads(response)
        except Exception:
            match = re.search(r"\{.*\}", response, flags=re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass

        return {
            "divergences": [],
            "faithful_implementations": [],
            "missing_implementations": [],
            "overall_fidelity": "low",
            "summary": response[:500],
        }

    async def _fetch_paper(self, arxiv_id: str) -> ExtractedPaper:
        async with PDFFetcher() as fetcher:
            result = await fetcher.download(arxiv_id)
            if not result.success:
                raise RuntimeError(
                    f"Failed to download PDF for {arxiv_id}: {result.error}"
                )

        return PDFParser().parse(result.local_path, arxiv_id)

    def diff(
        self, arxiv_id: str, github_url: str, force_refresh: bool = False
    ) -> ImplementationDiff:
        fetcher = GitHubFetcher()
        owner, repo = fetcher._parse_github_url(github_url)
        cache_path = self._cache_path(arxiv_id, owner, repo)

        if not force_refresh and self._is_cache_valid(cache_path):
            try:
                raw = json.loads(cache_path.read_text(encoding="utf-8"))
                return ImplementationDiff.model_validate(raw)
            except Exception:
                pass

        contributions = _run_sync(lambda: ContributionExtractor().extract(arxiv_id, force_refresh=force_refresh))
        paper = _run_sync(lambda: self._fetch_paper(arxiv_id))

        method_text = self._extract_method_section(paper)

        repo_summary = _run_sync(fetcher.fetch_repo_summary(github_url))
        code_content, total_tokens = self._build_code_content(
            repo_summary.get("files", {})
        )

        prompt = self._build_prompt(
            contributions, method_text, code_content, paper.title
        )
        llm_response = self._call_llm(prompt)
        parsed = self._parse_llm_response(llm_response)

        diff_report = ImplementationDiff(
            arxiv_id=arxiv_id,
            github_url=github_url,
            paper_title=paper.title,
            divergences=[
                # pydantic will validate each dict element into Divergence if possible
                d
                for d in parsed.get("divergences", [])
            ],
            faithful_implementations=parsed.get("faithful_implementations", []),
            missing_implementations=parsed.get("missing_implementations", []),
            overall_fidelity=parsed.get("overall_fidelity", "low"),
            summary=parsed.get("summary", ""),
            code_files_analyzed=list(repo_summary.get("files", {}).keys()),
            total_code_tokens=total_tokens,
            diffed_at=datetime.utcnow(),
        )

        try:
            cache_path.write_text(
                diff_report.model_dump_json(by_alias=True, exclude_none=True),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("Diff cache write failed", path=str(cache_path), error=str(exc))

        return diff_report
