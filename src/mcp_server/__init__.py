"""
mcp_server — arXiv MCP server exposing research paper tools.

Tools exposed:
  - search_arxiv(query, max_results?)
  - get_paper_by_id(arxiv_id)
  - download_pdf(arxiv_id, force?)
  - extract_text(arxiv_id)
  - get_paper_context(arxiv_id, max_chunks?)

Run with:
  python -m src.mcp_server
"""

# pragma: no cover
from __future__ import annotations

import atexit
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from src.arxiv_client import ArxivClient
from src.pdf_fetcher import PDFFetcher
from src.pdf_parser import PDFParser
from src.context_builder import ContextBuilder
from src.logger import get_logger, configure_logging
from src.models import ARXIV_DB_PATH, KEEP_PDFS
from src.devtools.link_extractor import LinkExtractor
from src.devtools.reproducibility_scorer import ReproducibilityScorer
from src.workflows.db import DatabaseClient
from src.workflows.explainer import Explainer
from src.workflows.reading_list import ReadingListManager
from src.workflows.topic_watcher import TopicWatcher
from src.devtools.implementation_differ import ImplementationDiffer
from src.intelligence.citation_graph import SemanticScholarClient
from src.intelligence.contribution_extractor import ContributionExtractor
from src.intelligence.paper_comparator import PaperComparator
from src.intelligence.semantic_index import SemanticIndex

configure_logging("INFO")
log = get_logger("mcp_server")

# ── Instantiate modules ───────────────────────────────────────────────────────
_arxiv_client = ArxivClient()
_pdf_parser = PDFParser()
_context_builder = ContextBuilder()
_db = DatabaseClient(ARXIV_DB_PATH)

atexit.register(lambda: _db.close())

# ── MCP Server ────────────────────────────────────────────────────────────────
server = Server("arxiv-mcp")


# ── Tool Definitions ──────────────────────────────────────────────────────────


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_arxiv",
            description=(
                "Search arXiv for research papers using a natural language query or arXiv ID. "
                "Returns titles, authors, abstracts, categories and PDF URLs. "
                "Automatically detects arXiv IDs in the query (e.g., '2603.17216') and routes to precise lookup."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query or arXiv ID (e.g., 'attention is all you need' or '1706.03762')",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 10, max: 50)",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="get_paper_by_id",
            description=(
                "Retrieve complete metadata for a specific arXiv paper by its ID. "
                "Returns title, authors, abstract, categories, publication dates, and PDF URL."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "arxiv_id": {
                        "type": "string",
                        "description": "arXiv paper ID, e.g. '2603.17216' or '1706.03762'",
                    },
                },
                "required": ["arxiv_id"],
            },
        ),
        types.Tool(
            name="download_pdf",
            description=(
                "Download the PDF of an arXiv paper to local storage. "
                "Returns the local file path and file size. "
                "Caches downloads — subsequent calls for the same ID are instant."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "arxiv_id": {
                        "type": "string",
                        "description": "arXiv paper ID",
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Re-download even if already cached (default: false)",
                        "default": False,
                    },
                },
                "required": ["arxiv_id"],
            },
        ),
        types.Tool(
            name="extract_text",
            description=(
                "Extract and structure the full text from a downloaded arXiv PDF. "
                "Downloads the PDF first if not already cached. "
                "Returns full text, page count, and token-aware chunks ready for LLM use."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "arxiv_id": {
                        "type": "string",
                        "description": "arXiv paper ID",
                    },
                },
                "required": ["arxiv_id"],
            },
        ),
        types.Tool(
            name="get_paper_context",
            description=(
                "Get a complete LLM-ready context bundle for an arXiv paper. "
                "Combines metadata + full PDF text into structured chunks with a system prompt. "
                "Ideal for paper summarization, question answering, and analysis. "
                "Downloads and parses the PDF automatically if needed."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "arxiv_id": {
                        "type": "string",
                        "description": "arXiv paper ID",
                    },
                    "max_chunks": {
                        "type": "integer",
                        "description": "Limit number of text chunks to include (default: all chunks)",
                        "minimum": 1,
                    },
                },
                "required": ["arxiv_id"],
            },
        ),
        types.Tool(
            name="arxiv_citation_graph",
            description="Fetch citation graph for an arXiv paper via Semantic Scholar.",
            inputSchema={
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string"},
                    "max_references": {"type": "integer", "default": 50},
                    "max_citations": {"type": "integer", "default": 50},
                    "influential_only": {"type": "boolean", "default": False},
                },
                "required": ["arxiv_id"],
            },
        ),
        types.Tool(
            name="arxiv_extract_contributions",
            description="Extract structured contributions from an arXiv paper.",
            inputSchema={
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string"},
                    "force_refresh": {"type": "boolean", "default": False},
                },
                "required": ["arxiv_id"],
            },
        ),
        types.Tool(
            name="arxiv_compare_papers",
            description="Compare multiple papers by structured contributions and synthesize insights.",
            inputSchema={
                "type": "object",
                "properties": {
                    "arxiv_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["arxiv_ids"],
            },
        ),
        types.Tool(
            name="arxiv_find_related",
            description="Find semantically related papers from local index by query_text or query_arxiv_id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query_arxiv_id": {"type": "string"},
                    "query_text": {"type": "string"},
                    "top_k": {"type": "integer", "default": 10},
                },
                "anyOf": [
                    {"required": ["query_arxiv_id"]},
                    {"required": ["query_text"]},
                ],
            },
        ),
        types.Tool(
            name="arxiv_extract_code_links",
            description="Find all code repositories, datasets, models, and project pages linked to a paper.",
            inputSchema={
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string"},
                    "force_refresh": {"type": "boolean", "default": False},
                },
                "required": ["arxiv_id"],
            },
        ),
        types.Tool(
            name="arxiv_reproducibility_score",
            description="Score paper reproducibility based on code, data, and experiments.",
            inputSchema={
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string"},
                    "force_refresh": {"type": "boolean", "default": False},
                },
                "required": ["arxiv_id"],
            },
        ),
        types.Tool(
            name="arxiv_diff_implementations",
            description="Compare paper method against GitHub repo implementation and report divergences.",
            inputSchema={
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string"},
                    "github_url": {"type": "string"},
                    "force_refresh": {"type": "boolean", "default": False},
                },
                "required": ["arxiv_id", "github_url"],
            },
        ),
        types.Tool(
            name="arxiv_reading_list",
            description="CRUD reading list operations (add, list, stats, update, remove).",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["add", "list", "stats", "get", "update", "remove"]},
                    "arxiv_id": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "string"},
                    "read_status": {"type": "string", "enum": ["unread", "reading", "read"]},
                    "filter_status": {"type": "string", "enum": ["unread", "reading", "read"]},
                    "limit": {"type": "integer"},
                },
                "required": ["action"],
            },
        ),
        types.Tool(
            name="arxiv_watch_topic",
            description="Manage watched topics and check for new papers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["add", "remove", "list", "check", "check_all"]},
                    "query": {"type": "string"},
                    "label": {"type": "string"},
                    "topic_id": {"type": "integer"},
                },
                "required": ["action"],
            },
        ),
        types.Tool(
            name="arxiv_explain_for_audience",
            description="Generate audience-targeted explanation for a paper (LLM + fallback).",
            inputSchema={
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string"},
                    "audience": {"type": "string", "enum": ["layperson", "undergrad", "practitioner", "researcher", "executive"]},
                    "focus": {"type": "string", "enum": ["full", "abstract_only", "contributions_only"], "default": "full"},
                    "force_refresh": {"type": "boolean", "default": False},
                },
                "required": ["arxiv_id", "audience"],
            },
        ),
    ]


# ── Tool Handlers ─────────────────────────────────────────────────────────────


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    log.info("Tool called", tool=name, args=arguments)

    try:
        if name == "search_arxiv":
            return await _handle_search_arxiv(arguments)
        elif name == "get_paper_by_id":
            return await _handle_get_paper_by_id(arguments)
        elif name == "download_pdf":
            return await _handle_download_pdf(arguments)
        elif name == "extract_text":
            return await _handle_extract_text(arguments)
        elif name == "get_paper_context":
            return await _handle_get_paper_context(arguments)
        elif name == "arxiv_citation_graph":
            return await _handle_citation_graph(arguments)
        elif name == "arxiv_extract_contributions":
            return await _handle_extract_contributions(arguments)
        elif name == "arxiv_compare_papers":
            return await _handle_compare_papers(arguments)
        elif name == "arxiv_find_related":
            return await _handle_find_related(arguments)
        elif name == "arxiv_extract_code_links":
            return await _handle_extract_code_links(arguments)
        elif name == "arxiv_reproducibility_score":
            return await _handle_reproducibility_score(arguments)
        elif name == "arxiv_diff_implementations":
            return await _handle_diff_implementations(arguments)
        elif name == "arxiv_reading_list":
            return await _handle_reading_list(arguments)
        elif name == "arxiv_watch_topic":
            return await _handle_watch_topic(arguments)
        elif name == "arxiv_explain_for_audience":
            return await _handle_explain_for_audience(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

    except Exception as exc:
        log.error("Tool execution failed", tool=name, error=str(exc), exc_info=True)
        error_payload = json.dumps(
            {"error": str(exc), "tool": name, "type": type(exc).__name__},
            indent=2,
        )
        return [types.TextContent(type="text", text=error_payload)]


async def _handle_reading_list(args: dict[str, Any]) -> list[types.TextContent]:
    action = args.get("action")
    if not action:
        return [types.TextContent(type="text", text=json.dumps({"error": "action required"}))]

    manager = ReadingListManager(_db, _arxiv_client)
    params = {k: v for k, v in args.items() if k != "action"}

    result = await manager.dispatch(action, **params)
    return [
        types.TextContent(
            type="text",
            text=json.dumps(result.model_dump(), indent=2, default=str),
        )
    ]


async def _handle_watch_topic(args: dict[str, Any]) -> list[types.TextContent]:
    action = args.get("action")
    if not action:
        return [types.TextContent(type="text", text=json.dumps({"error": "action required"}))]

    watcher = TopicWatcher(_db, _arxiv_client)
    params = {k: v for k, v in args.items() if k != "action"}

    result = await watcher.dispatch(action, **params)
    return [
        types.TextContent(
            type="text",
            text=json.dumps(result.model_dump(), indent=2, default=str),
        )
    ]


async def _handle_explain_for_audience(args: dict[str, Any]) -> list[types.TextContent]:
    arxiv_id = args.get("arxiv_id", "").strip()
    audience = args.get("audience", "").strip()
    focus = args.get("focus", "full").strip()
    force_refresh = bool(args.get("force_refresh", False))

    if not arxiv_id or not audience:
        return [types.TextContent(type="text", text=json.dumps({"error": "arxiv_id and audience required"}))]

    explainer = Explainer(_db, ContributionExtractor(), _arxiv_client)

    explanation = await explainer.explain(
        arxiv_id=arxiv_id,
        audience=audience,
        focus=focus,
        force_refresh=force_refresh,
    )

    return [
        types.TextContent(
            type="text",
            text=json.dumps(explanation.model_dump(), indent=2, default=str),
        )
    ]


async def _handle_citation_graph(args: dict[str, Any]) -> list[types.TextContent]:
    arxiv_id = args.get("arxiv_id", "").strip()
    if not arxiv_id:
        return [types.TextContent(type="text", text=json.dumps({"error": "arxiv_id required"}))]

    async with SemanticScholarClient() as client:
        graph = await client.get_citation_graph(
            arxiv_id,
            max_references=int(args.get("max_references", 50)),
            max_citations=int(args.get("max_citations", 50)),
            influential_only=bool(args.get("influential_only", False)),
        )

    return [types.TextContent(type="text", text=json.dumps(graph.model_dump(), indent=2, default=str))]


async def _handle_extract_contributions(args: dict[str, Any]) -> list[types.TextContent]:
    arxiv_id = args.get("arxiv_id", "").strip()
    if not arxiv_id:
        return [types.TextContent(type="text", text=json.dumps({"error": "arxiv_id required"}))]

    extractor = ContributionExtractor()
    contributions = await extractor.extract(arxiv_id, force_refresh=bool(args.get("force_refresh", False)))

    return [types.TextContent(type="text", text=json.dumps(contributions.model_dump(), indent=2))]


async def _handle_extract_code_links(args: dict[str, Any]) -> list[types.TextContent]:
    arxiv_id = args.get("arxiv_id", "").strip()
    if not arxiv_id:
        return [types.TextContent(type="text", text=json.dumps({"error": "arxiv_id required"}))]

    extractor = LinkExtractor()
    code_links = await extractor.extract(arxiv_id, force_refresh=bool(args.get("force_refresh", False)))

    return [types.TextContent(type="text", text=json.dumps(code_links.model_dump(), indent=2, default=str))]


async def _handle_reproducibility_score(args: dict[str, Any]) -> list[types.TextContent]:
    arxiv_id = args.get("arxiv_id", "").strip()
    if not arxiv_id:
        return [types.TextContent(type="text", text=json.dumps({"error": "arxiv_id required"}))]

    scorer = ReproducibilityScorer()
    report = scorer.score(arxiv_id, force_refresh=bool(args.get("force_refresh", False)))

    return [types.TextContent(type="text", text=json.dumps(report.model_dump(), indent=2, default=str))]


async def _handle_diff_implementations(args: dict[str, Any]) -> list[types.TextContent]:
    arxiv_id = args.get("arxiv_id", "").strip()
    github_url = args.get("github_url", "").strip()

    if not arxiv_id:
        return [types.TextContent(type="text", text=json.dumps({"error": "arxiv_id required"}))]
    if not github_url:
        return [types.TextContent(type="text", text=json.dumps({"error": "github_url required"}))]

    differ = ImplementationDiffer()
    diff = differ.diff(arxiv_id, github_url, force_refresh=bool(args.get("force_refresh", False)))

    return [types.TextContent(type="text", text=json.dumps(diff.model_dump(), indent=2, default=str))]


async def _handle_compare_papers(args: dict[str, Any]) -> list[types.TextContent]:
    arxiv_ids = args.get("arxiv_ids") or []
    if not isinstance(arxiv_ids, list) or not arxiv_ids:
        return [types.TextContent(type="text", text=json.dumps({"error": "arxiv_ids required"}))]

    extractor = ContributionExtractor()
    comparator = PaperComparator(extractor)

    try:
        report = await comparator.compare(arxiv_ids)
    except Exception as exc:
        return [types.TextContent(type="text", text=json.dumps({"error": str(exc)}))]

    return [types.TextContent(type="text", text=json.dumps(report.model_dump(), indent=2, default=str))]


async def _handle_find_related(args: dict[str, Any]) -> list[types.TextContent]:
    query_arxiv_id = args.get("query_arxiv_id")
    query_text = args.get("query_text")
    top_k = int(args.get("top_k", 10))

    if not query_arxiv_id and not query_text:
        return [types.TextContent(type="text", text=json.dumps({"error": "query_arxiv_id or query_text required"}))]

    index = SemanticIndex()

    try:
        if query_arxiv_id:
            results = index.query_by_paper(query_arxiv_id, top_k=top_k)
        else:
            results = index.query_by_text(query_text, top_k=top_k)
    except Exception as exc:
        return [types.TextContent(type="text", text=json.dumps({"error": str(exc)}))]

    return [types.TextContent(type="text", text=json.dumps(results.model_dump(), indent=2, default=str))]


async def _handle_search_arxiv(args: dict) -> list[types.TextContent]:
    query = args["query"]
    max_results = min(int(args.get("max_results", 10)), 50)

    results = await _arxiv_client.search(query=query, max_results=max_results)

    if not results:
        payload = {
            "message": "No papers found for the given query.",
            "query": query,
            "results": [],
        }
    else:
        payload = {
            "query": query,
            "total_found": len(results),
            "results": [r.model_dump() for r in results],
        }

    return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]


async def _handle_get_paper_by_id(args: dict) -> list[types.TextContent]:
    arxiv_id = args["arxiv_id"].strip()
    metadata = await _arxiv_client.get_by_id(arxiv_id)

    if metadata is None:
        payload = {"error": f"Paper not found: {arxiv_id}", "arxiv_id": arxiv_id}
    else:
        payload = metadata.model_dump()

    # Non-blocking semantic index side effect.
    # add_paper is synchronous, so run in thread to avoid event-loop mismatch.
    try:
        import asyncio

        asyncio.create_task(
            asyncio.to_thread(
                SemanticIndex().add_paper,
                arxiv_id,
                metadata.title,
                metadata.abstract,
                None,
            )
        )
    except Exception as exc:
        log.warning("Semantic index side effect failed", arxiv_id=arxiv_id, error=str(exc))

    return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]


async def _handle_download_pdf(args: dict) -> list[types.TextContent]:
    arxiv_id = args["arxiv_id"].strip()
    force = bool(args.get("force", False))

    async with PDFFetcher() as fetcher:
        result = await fetcher.download(arxiv_id, force=force)

    return [
        types.TextContent(type="text", text=json.dumps(result.model_dump(), indent=2))
    ]


async def _handle_extract_text(args: dict) -> list[types.TextContent]:
    arxiv_id = args["arxiv_id"].strip()

    # Ensure PDF is downloaded first
    async with PDFFetcher() as fetcher:
        dl_result = await fetcher.download(arxiv_id)

    if not dl_result.success:
        payload = {"error": dl_result.error, "arxiv_id": arxiv_id}
        return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]

    extracted = _pdf_parser.parse(dl_result.local_path, arxiv_id)

    # Non-blocking semantic index side effect.
    # add_paper is synchronous, so run in thread to avoid event-loop mismatch.
    try:
        import asyncio

        asyncio.create_task(
            asyncio.to_thread(
                SemanticIndex().add_paper,
                arxiv_id,
                extracted.title,
                extracted.full_text,
                None,
            )
        )
    except Exception as exc:
        log.warning("Semantic index side effect failed", arxiv_id=arxiv_id, error=str(exc))

    # Remove temporary download if configured to keep PDFs off disk
    if not KEEP_PDFS:
        try:
            Path(dl_result.local_path).unlink()
            log.info("Temp PDF removed", arxiv_id=arxiv_id, path=dl_result.local_path)
        except Exception as exc:
            log.warning(
                "Failed to remove temp PDF",
                arxiv_id=arxiv_id,
                error=str(exc),
                path=dl_result.local_path,
            )

    # Serialize — omit full_text in response to keep it manageable
    payload = {
        "arxiv_id": extracted.arxiv_id,
        "title": extracted.title,
        "total_pages": extracted.total_pages,
        "extraction_method": extracted.extraction_method,
        "full_text_length_chars": len(extracted.full_text),
        "chunk_count": len(extracted.chunks),
        "total_tokens": sum(c.token_count for c in extracted.chunks),
        "first_500_chars": extracted.full_text[:500],
        "chunks_preview": [
            {
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
                "section_hint": c.section_hint,
                "preview": c.text[:200],
            }
            for c in extracted.chunks[:3]
        ],
    }

    return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]


async def _handle_get_paper_context(args: dict) -> list[types.TextContent]:
    arxiv_id = args["arxiv_id"].strip()
    max_chunks = args.get("max_chunks")

    # 1. Fetch metadata
    metadata = await _arxiv_client.get_by_id(arxiv_id)
    if metadata is None:
        payload = {"error": f"Paper not found on arXiv: {arxiv_id}"}
        return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]

    # 2. Download PDF
    async with PDFFetcher() as fetcher:
        dl_result = await fetcher.download(arxiv_id)

    if not dl_result.success:
        payload = {
            "error": f"PDF download failed: {dl_result.error}",
            "arxiv_id": arxiv_id,
        }
        return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]

    # 3. Parse PDF
    extracted = _pdf_parser.parse(dl_result.local_path, arxiv_id)

    # Remove temporary download if configured to keep PDFs off disk
    if not KEEP_PDFS:
        try:
            Path(dl_result.local_path).unlink()
            log.info("Temp PDF removed", arxiv_id=arxiv_id, path=dl_result.local_path)
        except Exception as exc:
            log.warning(
                "Failed to remove temp PDF",
                arxiv_id=arxiv_id,
                error=str(exc),
                path=dl_result.local_path,
            )

    # 4. Build context
    context = _context_builder.build(
        metadata=metadata,
        extracted=extracted,
        max_chunks=max_chunks,
    )

    # 5. Serialize for LLM consumption
    payload = {
        "arxiv_id": arxiv_id,
        "metadata": metadata.model_dump(),
        "llm_system_prompt": context.llm_system_prompt,
        "summary_prompt": context.summary_prompt,
        "total_tokens": context.total_tokens,
        "chunk_count": context.chunk_count,
        "chunks": [c.model_dump() for c in context.chunks],
    }

    return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]


# ── Entry Point ───────────────────────────────────────────────────────────────


async def main() -> None:
    log.info("Starting arxiv-mcp server", version="1.0.0")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
