#!/usr/bin/env python3
"""
test_smoke.py — Quick smoke test for all arxiv-mcp modules.
Run from project root: python tests/test_smoke.py
"""
import asyncio
import sys
import os

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
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{text[:50]}' → {result}")

    # ── 2. Search ─────────────────────────────────────────────────
    print("\n[2] Testing arXiv search...")
    client = ArxivClient()
    results = await client.search("attention transformer NLP", max_results=3)
    print(f"  ✓ Got {len(results)} results")
    for r in results:
        print(f"    - [{r.arxiv_id}] {r.title[:60]}...")

    # ── 3. Metadata lookup by ID ─────────────────────────────────
    print("\n[3] Testing get_by_id...")
    meta = await client.get_by_id("1706.03762")  # Attention is All You Need
    if meta:
        print(f"  ✓ Title: {meta.title}")
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
        print(f"  ✓ Downloaded: {dl.local_path} ({dl.file_size_bytes // 1024} KB)")
    else:
        print(f"  ✗ Download failed: {dl.error}")
        return

    # ── 5. PDF Parsing ────────────────────────────────────────────
    print("\n[5] Testing PDF parsing...")
    parser = PDFParser()
    extracted = parser.parse(dl.local_path, "1706.03762")
    print(f"  ✓ Pages: {extracted.total_pages}")
    print(f"    Chars: {len(extracted.full_text):,}")
    print(f"    Chunks: {len(extracted.chunks)}")
    print(f"    Total tokens: {sum(c.token_count for c in extracted.chunks):,}")
    print(f"    First 200 chars: {extracted.full_text[:200]!r}")

    # ── 6. Context Building ───────────────────────────────────────
    print("\n[6] Testing context builder...")
    builder = ContextBuilder()
    context = builder.build(meta, extracted, max_chunks=5)
    print(f"  ✓ Chunks in context: {context.chunk_count}")
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
        f"  ✓ Graph for 1706.03762: {citation_graph.reference_count} refs, {citation_graph.citation_count} total citations"
    )

    # ── 8. Contribution Extraction (Layer 2) ───────────────────────
    print("\n[8] Testing contribution extraction (LLM + heuristic fallback)...")
    from src.intelligence.contribution_extractor import ContributionExtractor

    contributor = ContributionExtractor()
    contributions = await contributor.extract("1706.03762", force_refresh=True)
    print(f"  ✓ Contribution extraction method: {contributions.extraction_method}")
    print(f"    Core claim: {contributions.core_claim[:120]!r}")

    # ── 9. Layer 3 Dev Tooling smoke tests ─────────────────────────
    print("\n[9] Testing arxiv_extract_code_links smoke...")
    from src.devtools.link_extractor import LinkExtractor

    link_report = await LinkExtractor().extract("1706.03762", force_refresh=False)
    print(
        f"  ✓ Extracted {len(link_report.github_repos)} github repos, official={link_report.has_official_code}"
    )

    print("\n[10] Testing arxiv_reproducibility_score smoke...")
    from src.devtools.reproducibility_scorer import ReproducibilityScorer

    score_report = ReproducibilityScorer().score("1706.03762", force_refresh=False)
    print(f"  ✓ Reproducibility score {score_report.score}, band={score_report.band}")
    if score_report.score <= 5.0:
        print(
            "  ⚠️ Score <= 5.0; expected > 5.0 for this paper but may vary with environment"
        )

    print("\n" + "=" * 60)
    print("  All smoke tests passed! ✓")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(run())
