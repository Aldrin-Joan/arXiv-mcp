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
        raise TypeError(
            "_run_sync expects a coroutine or callable returning a coroutine"
        )

    try:
        asyncio.get_running_loop()

        def _runner():
            return asyncio.run(coro)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(_runner).result()
    except RuntimeError:
        return asyncio.run(coro)


from pathlib import Path
from typing import Dict
from urllib.parse import urlparse

import httpx

from src.devtools.link_extractor import LinkExtractor
from src.models import (
    CodeLinks,
    DOWNLOAD_DIR,
    GITHUB_TOKEN,
    HTTP_TIMEOUT,
    REPRO_CACHE_TTL,
    ReproducibilityReport,
    ReproducibilitySignal,
)
from src.pdf_fetcher import PDFFetcher
from src.pdf_parser import PDFParser
from src.logger import get_logger

log = get_logger("reproducibility_scorer")

_PUBLIC_COMPUTE_KEYWORDS = [
    "V100",
    "A100",
    "A6000",
    "RTX 3090",
    "RTX 4090",
    "P100",
    "T4",
    "TPU",
]

_HYPERPARAM_PHRASES = [
    "learning rate",
    "lr",
    "batch size",
    "epochs",
    "optimizer",
    "weight decay",
    "momentum",
    "dropout",
]


class ReproducibilityScorer:
    """Deterministic scoring of paper reproducibility signals."""

    def __init__(self) -> None:
        self._cache_dir = Path(DOWNLOAD_DIR) / "reproducibility"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, arxiv_id: str) -> Path:
        return self._cache_dir / f"{arxiv_id}.json"

    def _is_cache_valid(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            scored_at = raw.get("scored_at")
            if not scored_at:
                return False
            scored_dt = datetime.fromisoformat(scored_at)
            return datetime.utcnow() - scored_dt < timedelta(seconds=REPRO_CACHE_TTL)
        except Exception as exc:
            log.warning("Repro cache parse failed", path=str(path), error=str(exc))
            return False

    def _signal_code_repo(self, code_links: CodeLinks) -> ReproducibilitySignal:
        detected = bool(code_links.has_official_code or code_links.github_repos)
        evidence = ""
        if detected and code_links.github_repos:
            evidence = code_links.github_repos[0].url
        elif code_links.has_official_code:
            evidence = "official code link from PapersWithCode"

        return ReproducibilitySignal(
            name="Code repository linked",
            points_awarded=2.0 if detected else 0.0,
            points_possible=2.0,
            detected=detected,
            evidence=evidence,
        )

    def _signal_public_dataset(
        self, text: str, code_links: CodeLinks
    ) -> ReproducibilitySignal:
        detected = bool(code_links.dataset_links)
        evidence = ""

        if detected:
            evidence = code_links.dataset_links[0].url
        else:
            pattern = re.compile(r"publicly available", re.IGNORECASE)
            m = pattern.search(text)
            if m:
                evidence = text[max(0, m.start() - 50) : m.end() + 50].strip()
                detected = True

        return ReproducibilitySignal(
            name="Dataset publicly available",
            points_awarded=1.5 if detected else 0.0,
            points_possible=1.5,
            detected=detected,
            evidence=evidence,
        )

    def _signal_hyperparameters(self, text: str) -> ReproducibilitySignal:
        text_lower = text.lower()
        matches = [p for p in _HYPERPARAM_PHRASES if p in text_lower]
        detected = len(matches) >= 3
        evidence = ", ".join(matches)

        return ReproducibilitySignal(
            name="Hyperparameters reported",
            points_awarded=1.5 if detected else 0.0,
            points_possible=1.5,
            detected=detected,
            evidence=evidence,
        )

    def _signal_ablation(self, text: str) -> ReproducibilitySignal:
        m = re.search(r"ablation", text, re.IGNORECASE)
        detected = bool(m)
        evidence = ""
        if m:
            start = max(0, m.start() - 60)
            end = min(len(text), m.end() + 60)
            evidence = text[start:end]

        return ReproducibilitySignal(
            name="Ablation study present",
            points_awarded=1.0 if detected else 0.0,
            points_possible=1.0,
            detected=detected,
            evidence=evidence,
        )

    def _signal_seeds(self, text: str) -> ReproducibilitySignal:
        patterns = [r"seed\s*[=:]\s*\d+", r"random seed", r"fix(?:ed)? the seed"]
        detected = False
        evidence = ""
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                detected = True
                evidence = m.group(0)
                break

        return ReproducibilitySignal(
            name="Random seeds reported",
            points_awarded=0.5 if detected else 0.0,
            points_possible=0.5,
            detected=detected,
            evidence=evidence,
        )

    def _signal_error_bars(self, text: str) -> ReproducibilitySignal:
        patterns = [
            r"±",
            r"\\pm",
            r"std(?:ev)?",
            r"standard deviation",
            r"average of \d+ runs",
            r"mean over \d+ (?:runs|seeds)",
        ]
        detected = False
        evidence = ""

        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                detected = True
                evidence = m.group(0)
                break

        return ReproducibilitySignal(
            name="Multiple runs / error bars",
            points_awarded=0.5 if detected else 0.0,
            points_possible=0.5,
            detected=detected,
            evidence=evidence,
        )

    def _signal_compute(self, text: str) -> ReproducibilitySignal:
        detected = False
        evidence = ""

        for gpu in _PUBLIC_COMPUTE_KEYWORDS:
            if gpu.lower() in text.lower():
                detected = True
                evidence = gpu
                break

        if not detected:
            m = re.search(r"\d+\.?\d*\s*(GPU|TPU|node)", text, re.IGNORECASE)
            if m:
                detected = True
                evidence = m.group(0)

        if not detected:
            m = re.search(r"(FLOPs|GFLOPs|PFLOPs|training time)", text, re.IGNORECASE)
            if m:
                detected = True
                evidence = m.group(0)

        return ReproducibilitySignal(
            name="Hardware/compute reported",
            points_awarded=0.5 if detected else 0.0,
            points_possible=0.5,
            detected=detected,
            evidence=evidence,
        )

    def _signal_eval_code(
        self, text: str, code_links: CodeLinks
    ) -> ReproducibilitySignal:
        detected = False
        evidence = ""

        for repo in code_links.github_repos:
            if any(x in repo.url.lower() for x in ["eval", "benchmark", "metric"]):
                detected = True
                evidence = repo.url
                break

        if not detected:
            m = re.search(r"evaluation code|evaluation script", text, re.IGNORECASE)
            if m:
                detected = True
                evidence = m.group(0)

        return ReproducibilitySignal(
            name="Evaluation code available",
            points_awarded=0.5 if detected else 0.0,
            points_possible=0.5,
            detected=detected,
            evidence=evidence,
        )

    def _signal_model_weights(
        self, text: str, code_links: CodeLinks
    ) -> ReproducibilitySignal:
        detected = any(
            link.link_type == "huggingface_model"
            for link in code_links.huggingface_links
        )
        evidence = ""
        if detected:
            evidence = next(
                (
                    link.url
                    for link in code_links.huggingface_links
                    if link.link_type == "huggingface_model"
                ),
                "",
            )

        if not detected:
            m = re.search(
                r"(model weights|checkpoint|pretrained model)", text, re.IGNORECASE
            )
            if m:
                detected = True
                evidence = m.group(0)

        return ReproducibilitySignal(
            name="Pre-trained models available",
            points_awarded=0.5 if detected else 0.0,
            points_possible=0.5,
            detected=detected,
            evidence=evidence,
        )

    def _signal_license(self, code_links: CodeLinks) -> ReproducibilitySignal:
        if not code_links.github_repos:
            return ReproducibilitySignal(
                name="License present in code",
                points_awarded=0.0,
                points_possible=0.5,
                detected=False,
                evidence="no GitHub repo found",
            )

        first = code_links.github_repos[0].url
        parsed = urlparse(first)
        path_parts = [p for p in parsed.path.strip("/").split("/") if p]

        if len(path_parts) < 2:
            return ReproducibilitySignal(
                name="License present in code",
                points_awarded=0.0,
                points_possible=0.5,
                detected=False,
                evidence="invalid GitHub repo URL",
            )

        owner, repo = path_parts[0], path_parts[1].replace(".git", "")
        url = f"https://api.github.com/repos/{owner}/{repo}"

        headers: Dict[str, str] = {"Accept": "application/vnd.github.v3+json"}
        if GITHUB_TOKEN:
            headers["Authorization"] = f"token {GITHUB_TOKEN}"

        try:
            response = httpx.get(url, headers=headers, timeout=HTTP_TIMEOUT)
            if response.status_code != 200:
                return ReproducibilitySignal(
                    name="License present in code",
                    points_awarded=0.0,
                    points_possible=0.5,
                    detected=False,
                    evidence=f"GitHub API status {response.status_code}",
                )
            data = response.json()
            license_field = data.get("license")
            detected = license_field is not None
            evidence = (
                license_field.get("name")
                if isinstance(license_field, dict)
                else str(license_field)
            )
            if not detected:
                evidence = "no license in GitHub repo"

            return ReproducibilitySignal(
                name="License present in code",
                points_awarded=0.5 if detected else 0.0,
                points_possible=0.5,
                detected=detected,
                evidence=evidence,
            )

        except Exception as exc:
            log.warning("GitHub license check failed", repo=first, error=str(exc))
            return ReproducibilitySignal(
                name="License present in code",
                points_awarded=0.0,
                points_possible=0.5,
                detected=False,
                evidence=str(exc),
            )

    def _band(self, score: float) -> str:
        if score >= 8.0:
            return "Highly Reproducible"
        if score >= 5.0:
            return "Moderately Reproducible"
        if score >= 2.5:
            return "Partially Reproducible"
        return "Difficult to Reproduce"

    async def _fetch_pdf_text(self, arxiv_id: str) -> str:
        async with PDFFetcher() as fetcher:
            result = await fetcher.download(arxiv_id, force=False)
            if not result.success:
                return ""
            parser = PDFParser()
            extracted = parser.parse(result.local_path, arxiv_id)
            return extracted.full_text

    def score(
        self, arxiv_id: str, force_refresh: bool = False
    ) -> ReproducibilityReport:
        cache_path = self._cache_path(arxiv_id)

        if not force_refresh and self._is_cache_valid(cache_path):
            try:
                raw = json.loads(cache_path.read_text(encoding="utf-8"))
                return ReproducibilityReport.model_validate(raw)
            except Exception as exc:
                log.warning(
                    "Repro cache load failed", path=str(cache_path), error=str(exc)
                )

        code_links = _run_sync(
            lambda: LinkExtractor().extract(arxiv_id, force_refresh=force_refresh)
        )
        text = _run_sync(lambda: self._fetch_pdf_text(arxiv_id))

        signals = [
            self._signal_code_repo(code_links),
            self._signal_public_dataset(text, code_links),
            self._signal_hyperparameters(text),
            self._signal_ablation(text),
            self._signal_seeds(text),
            self._signal_error_bars(text),
            self._signal_compute(text),
            self._signal_eval_code(text, code_links),
            self._signal_model_weights(text, code_links),
            self._signal_license(code_links),
        ]

        score = round(sum(s.points_awarded for s in signals), 1)
        band = self._band(score)

        report = ReproducibilityReport(
            arxiv_id=arxiv_id,
            score=score,
            band=band,
            signals=signals,
            code_links=code_links,
            scored_at=datetime.utcnow(),
        )

        try:
            cache_path.write_text(
                report.model_dump_json(by_alias=True, exclude_none=True),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning(
                "Repro cache write failed", path=str(cache_path), error=str(exc)
            )

        return report
