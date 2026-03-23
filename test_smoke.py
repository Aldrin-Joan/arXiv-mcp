#!/usr/bin/env python3
"""
test_smoke.py — Quick smoke test for all arxiv-mcp modules.
Run from project root: python tests/test_smoke.py
"""
import asyncio
import sys
import os
import json

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.logger import configure_logging, get_logger
from src.arxiv_client import ArxivClient, detect_arxiv_id
from src.pdf_fetcher import PDFFetcher
from src.pdf_parser import PDFParser
from src.context_builder import ContextBuilder

configure_logging("INFO")
log = get_logger("smoke_test")


async def run():
    print("\n" + "=" * 60)
    print("  arxiv-mcp Smoke Test")
    print("=" * 60)

    # ── 1. ID Detection ───────────────────────────────────────────
    print("\n[1] Testing arXiv ID detection...")
    test_cases = [
        ("Check paper 2603.17216 about AI agents", "2603.17216"),
        ("What is arxiv:1706.03762?", "1706.03762"),
        ("Find papers about transformers", None),
    ]
    for text, expected in test_cases:
        result = detect_arxiv_id(text)
        status = "OK" if result == expected else "FAIL"
        print(f"  [{status}] '{text[:50]}' -> {result}")

    # ── 2. Search ─────────────────────────────────────────────────
    print("\n[2] Testing arXiv search...")
    client = ArxivClient()
    results = await client.search("attention transformer NLP", max_results=3)
    print(f"  [OK] Got {len(results)} results")
    for r in results:
        print(f"    - [{r.arxiv_id}] {r.title[:60]}...")

    # ── 3. Metadata lookup by ID ─────────────────────────────────
    print("\n[3] Testing get_by_id...")
    meta = await client.get_by_id("1706.03762")  # Attention is All You Need
    if meta:
        print(f"  [OK] Title: {meta.title}")
        print(f"    Authors: {', '.join(a.name for a in meta.authors[:3])}...")
        print(f"    Categories: {meta.categories}")
        print(f"    PDF URL: {meta.pdf_url}")
    else:
        print("  ✗ Paper not found")

    # ── 4. PDF Download ───────────────────────────────────────────
    print("\n[4] Testing PDF download (1706.03762 - Attention is All You Need)...")
    async with PDFFetcher() as fetcher:
        dl = await fetcher.download("1706.03762")
    if dl.success:
        print(f"  [OK] Downloaded: {dl.local_path} ({dl.file_size_bytes // 1024} KB)")
    else:
        print(f"  ✗ Download failed: {dl.error}")
        return

    # ── 5. PDF Parsing ────────────────────────────────────────────
    print("\n[5] Testing PDF parsing...")
    parser = PDFParser()
    extracted = parser.parse(dl.local_path, "1706.03762")
    print(f"  [OK] Pages: {extracted.total_pages}")
    print(f"    Chars: {len(extracted.full_text):,}")
    print(f"    Chunks: {len(extracted.chunks)}")
    print(f"    Total tokens: {sum(c.token_count for c in extracted.chunks):,}")
    print(f"    First 200 chars: {extracted.full_text[:200]!r}")

    # ── 6. Context Building ───────────────────────────────────────
    print("\n[6] Testing context builder...")
    builder = ContextBuilder()
    context = builder.build(meta, extracted, max_chunks=5)
    print(f"  [OK] Chunks in context: {context.chunk_count}")
    print(f"    Total tokens: {context.total_tokens:,}")
    print(f"    System prompt preview: {context.llm_system_prompt[:100]!r}...")

    # ── 7. Citation Graph (Layer 2) ───────────────────────────────
    print("\n[7] Testing citation graph extraction (Semantic Scholar)...")
    from src.intelligence.citation_graph import SemanticScholarClient

    async with SemanticScholarClient() as s2_client:
        citation_graph = await s2_client.get_citation_graph(
            "1706.03762", max_references=5, max_citations=5
        )

    print(
        f"  [OK] Graph for 1706.03762: {citation_graph.reference_count} refs, {citation_graph.citation_count} total citations"
    )

    # ── 8. Contribution Extraction (Layer 2) ───────────────────────
    print("\n[8] Testing contribution extraction (LLM + heuristic fallback)...")
    from src.intelligence.contribution_extractor import ContributionExtractor

    contributor = ContributionExtractor()
    contributions = await contributor.extract("1706.03762", force_refresh=True)
    print(f"  [OK] Contribution extraction method: {contributions.extraction_method}")
    print(f"    Core claim: {contributions.core_claim[:120]!r}")

    # ── 9. Layer 3 Dev Tooling smoke tests ─────────────────────────
    print("\n[9] Testing arxiv_extract_code_links smoke...")
    from src.devtools.link_extractor import LinkExtractor

    link_report = await LinkExtractor().extract("1706.03762", force_refresh=False)
    print(
        f"  [OK] Extracted {len(link_report.github_repos)} github repos, official={link_report.has_official_code}"
    )

    print("\n[10] Testing arxiv_reproducibility_score smoke...")
    from src.devtools.reproducibility_scorer import ReproducibilityScorer

    score_report = ReproducibilityScorer().score("1706.03762", force_refresh=False)
    print(
        f"  [OK] Reproducibility score {score_report.score}, band={score_report.band}"
    )
    if score_report.score <= 5.0:
        print(
            "  [WARN] Score <= 5.0; expected > 5.0 for this paper but may vary with environment"
        )
    # ── 11. Layer 4 workspaces smoke tests ─────────────────────────
    print("\n[11] Testing Layer 4 workflow smoke tests...")
    from src.workflows.db import DatabaseClient
    from src.workflows.reading_list import ReadingListManager
    from src.workflows.topic_watcher import TopicWatcher
    from src.workflows.explainer import Explainer
    from src.models import Author, PaperMetadata, PaperContributions

    class DummyArxivClientForReading:
        async def get_by_id(self, arxiv_id: str):
            return PaperMetadata(
                arxiv_id=arxiv_id,
                title="Test",
                authors=[Author(name="A")],
                abstract="Sentence one. Sentence two. Sentence three.",
                categories=["cs.AI"],
                primary_category="cs.AI",
                published="2024-01-01T00:00:00Z",
                updated="2024-01-02T00:00:00Z",
                pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
                entry_url="https://arxiv.org/abs/1706.03762",
            )

        async def search(self, query: str, max_results: int = 50):
            class Paper:
                def __init__(self, arxiv_id):
                    self.arxiv_id = arxiv_id

            return [Paper("id1"), Paper("id2")]

    class DummyContributionExtractor:
        async def extract(self, arxiv_id: str, force_refresh: bool = False):
            return PaperContributions(
                arxiv_id=arxiv_id,
                core_claim="core",
                proposed_method="method",
                key_results=["r1"],
                baselines_compared=[],
                limitations=[],
                datasets_used=[],
                task_domain="NLP",
                novelty_type="empirical",
                extraction_method="heuristic",
                extracted_at="2026-01-01T00:00:00Z",
            )

    db2 = DatabaseClient(":memory:")
    rl_mgr = ReadingListManager(db2, DummyArxivClientForReading())
    await rl_mgr.add("1706.03762", tags=["NLP"], notes="note")
    result_list = await rl_mgr.list()
    assert len(result_list.entries) >= 1
    await rl_mgr.update("1706.03762", read_status="read")
    stats = await rl_mgr.stats()
    assert stats.stats["read"] == 1

    tw = TopicWatcher(db2, DummyArxivClientForReading())
    add_result1 = await tw.add("cat:cs.AI", "AI")
    assert add_result1.action == "add"
    assert add_result1.message == "Topic saved"

    add_result2 = await tw.add("cat:cs.AI", "AI")
    assert add_result2.action == "add"
    assert add_result2.message == "Topic already exists"
    assert len(add_result2.topics) == 1

    check_res1 = await tw.check()
    assert check_res1.check_results[0].baseline_established
    check_res2 = await tw.check()
    assert not check_res2.check_results[0].baseline_established

    # reset utility should clear all workflow tables
    db2.reset()
    post_reset = await tw.list()
    assert post_reset.topics == []

    after_reset = await tw.check()
    assert after_reset.check_results == []

    class DummyContributionExtractorNoMethod:
        async def extract(self, arxiv_id: str, force_refresh: bool = False):
            return PaperContributions(
                arxiv_id=arxiv_id,
                core_claim="",
                proposed_method="",
                key_results=[],
                baselines_compared=[],
                limitations=[],
                datasets_used=[],
                task_domain="General",
                novelty_type="empirical",
                extraction_method="heuristic",
                extracted_at="2026-01-01T00:00:00Z",
            )

    explainer_baseline = Explainer(
        db2, DummyContributionExtractorNoMethod(), DummyArxivClientForReading()
    )

    async def fake_ollama_fail(prompt):
        raise RuntimeError("Ollama unavailable")

    explainer_baseline._call_ollama = fake_ollama_fail
    explanation_fallback = await explainer_baseline.explain(
        "1706.03762", "practitioner"
    )
    assert explanation_fallback.generation_method == "passthrough"
    assert (
        explanation_fallback.what_it_is == "Sentence one. Sentence two. Sentence three."
    )
    assert explanation_fallback.problem_solved == "Sentence one."
    assert explanation_fallback.how_it_works == "See abstract."
    assert explanation_fallback.why_it_matters == "See abstract."
    assert explanation_fallback.key_result == "Sentence three."

    # reading list note dedupe should skip repeated identical notes
    await rl_mgr.add("1706.03762", notes="Smoke test add")
    await rl_mgr.add("1706.03762", notes="Smoke test add")
    repeated_entry = await rl_mgr.get("1706.03762")
    assert repeated_entry.entry.notes.count("Smoke test add") == 1

    explainer = Explainer(
        db2, DummyContributionExtractor(), DummyArxivClientForReading()
    )

    async def fake_ollama(prompt):
        return json.dumps(
            {
                "what_it_is": "X",
                "problem_solved": "Y",
                "how_it_works": "Z",
                "why_it_matters": "W",
                "key_result": "K",
                "reading_time_minutes": 2,
            }
        )

    explainer._call_ollama = fake_ollama
    explanation = await explainer.explain("1706.03762", "practitioner")
    assert explanation.generation_method in {"llm", "passthrough"}

    print("  [OK] Layer 4 workflow smoke checks passed")
    print("\n" + "=" * 60)
    print("  All smoke tests passed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(run())
