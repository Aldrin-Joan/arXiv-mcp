"""LLM-based explanation synthesis with caching for Layer 4 workflows."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx

from src import models
from src.models import ExplanationResult, PaperContributions, PaperMetadata
from src.workflows.db import DatabaseClient

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "explain.txt"

AUDIENCE_DESCRIPTIONS = {
    "layperson": (
        "Explain with everyday analogies, avoid technical jargon, and keep it simple enough for those without a technical background."
    ),
    "undergrad": (
        "Target a strong undergraduate student who knows basic calculus and machine learning concepts, with clear method and result explanation."
    ),
    "practitioner": (
        "Focus on practical implications for engineers, including implementation considerations and integration tips."
    ),
    "researcher": (
        "Condense novelty and research contributions for domain-expert readers with emphasis on technical delta."
    ),
    "executive": (
        "Highlight business value, impact, and decisions without deep technical detail."
    ),
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Explainer:
    """Generates audience-targeted explanations through LLM and caching."""

    def __init__(
        self,
        db: DatabaseClient,
        contribution_extractor: Any,
        arxiv_client: Any,
    ) -> None:
        self.db = db
        self.contribution_extractor = contribution_extractor
        self.arxiv_client = arxiv_client
        self._ollama_url = models.OLLAMA_BASE_URL.rstrip("/")
        self._ollama_model = models.OLLAMA_MODEL

    def _load_from_cache(
        self, arxiv_id: str, audience: str
    ) -> Optional[ExplanationResult]:
        row = self.db.fetchone(
            "SELECT content FROM explanations WHERE arxiv_id = ? AND audience = ?",
            (arxiv_id, audience),
        )
        if not row:
            return None

        data = json.loads(row["content"])
        return ExplanationResult(**data)

    def _save_to_cache(self, result: ExplanationResult) -> None:
        payload = result.model_dump()
        payload["generated_at"] = result.generated_at.isoformat()

        self.db.execute(
            "INSERT OR REPLACE INTO explanations (arxiv_id, audience, content, generated_at) VALUES (?, ?, ?, ?)",
            (
                result.arxiv_id,
                result.audience,
                json.dumps(payload),
                result.generated_at.isoformat(),
            ),
        )

    def _build_prompt(
        self,
        contributions: PaperContributions,
        metadata: PaperMetadata,
        audience: str,
        focus: str,
    ) -> str:
        if not PROMPT_PATH.exists():
            raise FileNotFoundError(f"Prompt template not found: {PROMPT_PATH}")

        template = PROMPT_PATH.read_text(encoding="utf-8")
        audience_label = audience
        audience_description = AUDIENCE_DESCRIPTIONS.get(
            audience, "General audience explanation requested."
        )

        proposed_method = contributions.proposed_method or "N/A"
        key_results = (
            "; ".join(contributions.key_results) if contributions.key_results else "N/A"
        )
        abstract = metadata.abstract or ""

        if focus == "abstract_only":
            proposed_method = ""
            key_results = ""
        elif focus == "contributions_only":
            abstract = ""

        prompt = template.format(
            audience_label=audience_label,
            audience_description=audience_description,
            title=metadata.title,
            authors=", ".join([a.name for a in metadata.authors]),
            core_claim=contributions.core_claim,
            proposed_method=proposed_method,
            key_results=key_results,
            abstract=abstract,
        )

        return prompt

    async def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": self._ollama_model,
            "prompt": prompt,
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self._ollama_url}/api/generate", json=payload
            )
            response.raise_for_status()
            data = response.json()

        if isinstance(data, dict) and "response" in data:
            return str(data["response"])

        if isinstance(data, dict) and "choices" in data and data["choices"]:
            first = data["choices"][0]
            if isinstance(first, dict) and "text" in first:
                return str(first["text"])

        raise ValueError("Unexpected Ollama response format")

    def _passthrough(
        self,
        contributions: PaperContributions,
        metadata: PaperMetadata,
        audience: str,
    ) -> ExplanationResult:
        abstract_source = (metadata.abstract or "").strip() or "No abstract available."

        sentences = [
            s.strip()
            for s in re.split(r"(?<=[\.\!?])\s+", abstract_source)
            if s.strip()
        ]
        first_sentence = sentences[0] if sentences else abstract_source
        last_sentence = sentences[-1] if sentences else abstract_source

        return ExplanationResult(
            arxiv_id=contributions.arxiv_id,
            title=metadata.title,
            audience=audience,
            what_it_is=abstract_source,
            problem_solved=first_sentence,
            how_it_works="See abstract.",
            why_it_matters="See abstract.",
            key_result=last_sentence,
            reading_time_minutes=3,
            generation_method="passthrough",
            generated_at=datetime.now(timezone.utc),
        )

    def _parse_llm_response(
        self, raw_text: str, arxiv_id: str, title: str, audience: str
    ) -> ExplanationResult:
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            match = None
            for start in range(len(raw_text)):
                if raw_text[start] == "{":
                    try:
                        attempt = json.loads(raw_text[start:])
                        match = attempt
                        break
                    except json.JSONDecodeError:
                        continue
            if match is None:
                raise
            data = match

        required_keys = [
            "what_it_is",
            "problem_solved",
            "how_it_works",
            "why_it_matters",
            "key_result",
            "reading_time_minutes",
        ]

        if not all(k in data for k in required_keys):
            raise ValueError("LLM response missing required keys")

        explanation = ExplanationResult(
            arxiv_id=arxiv_id,
            title=title,
            audience=audience,
            what_it_is=str(data.get("what_it_is", "")),
            problem_solved=str(data.get("problem_solved", "")),
            how_it_works=str(data.get("how_it_works", "")),
            why_it_matters=str(data.get("why_it_matters", "")),
            key_result=str(data.get("key_result", "")),
            reading_time_minutes=int(data.get("reading_time_minutes", 2)),
            generation_method="llm",
            generated_at=datetime.now(timezone.utc),
        )

        return explanation

    async def explain(
        self,
        arxiv_id: str,
        audience: str,
        focus: str = "full",
        force_refresh: bool = False,
    ) -> ExplanationResult:
        arxiv_id = arxiv_id.strip()
        audience = audience.strip().lower()
        focus = focus.strip().lower()

        if audience not in AUDIENCE_DESCRIPTIONS:
            raise ValueError("Invalid audience")

        if focus not in {"full", "abstract_only", "contributions_only"}:
            raise ValueError("Invalid focus")

        if not force_refresh:
            cached = self._load_from_cache(arxiv_id, audience)
            if cached is not None:
                return cached

        contributions = await self.contribution_extractor.extract(
            arxiv_id, force_refresh=force_refresh
        )
        metadata = await self.arxiv_client.get_by_id(arxiv_id)
        if metadata is None:
            raise ValueError("Paper metadata not found")

        prompt = self._build_prompt(contributions, metadata, audience, focus)

        try:
            raw_response = await self._call_ollama(prompt)
            explanation = self._parse_llm_response(
                raw_response, arxiv_id, metadata.title, audience
            )
        except Exception:
            explanation = self._passthrough(contributions, metadata, audience)

        self._save_to_cache(explanation)
        return explanation
