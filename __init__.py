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
from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from src.arxiv_client import ArxivClient
from src.pdf_fetcher import PDFFetcher
from src.pdf_parser import PDFParser
from src.context_builder import ContextBuilder
from src.logger import get_logger, configure_logging

configure_logging("INFO")
log = get_logger("mcp_server")

# ── Instantiate modules ───────────────────────────────────────────────────────
_arxiv_client = ArxivClient()
_pdf_parser = PDFParser()
_context_builder = ContextBuilder()

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
        else:
            raise ValueError(f"Unknown tool: {name}")

    except Exception as exc:
        log.error("Tool execution failed", tool=name, error=str(exc), exc_info=True)
        error_payload = json.dumps(
            {"error": str(exc), "tool": name, "type": type(exc).__name__},
            indent=2,
        )
        return [types.TextContent(type="text", text=error_payload)]


async def _handle_search_arxiv(args: dict) -> list[types.TextContent]:
    query = args["query"]
    max_results = min(int(args.get("max_results", 10)), 50)

    results = await _arxiv_client.search(query=query, max_results=max_results)

    if not results:
        payload = {"message": "No papers found for the given query.", "query": query, "results": []}
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

    return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]


async def _handle_download_pdf(args: dict) -> list[types.TextContent]:
    arxiv_id = args["arxiv_id"].strip()
    force = bool(args.get("force", False))

    async with PDFFetcher() as fetcher:
        result = await fetcher.download(arxiv_id, force=force)

    return [types.TextContent(type="text", text=json.dumps(result.model_dump(), indent=2))]


async def _handle_extract_text(args: dict) -> list[types.TextContent]:
    arxiv_id = args["arxiv_id"].strip()

    # Ensure PDF is downloaded first
    async with PDFFetcher() as fetcher:
        dl_result = await fetcher.download(arxiv_id)

    if not dl_result.success:
        payload = {"error": dl_result.error, "arxiv_id": arxiv_id}
        return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]

    extracted = _pdf_parser.parse(dl_result.local_path, arxiv_id)

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
        payload = {"error": f"PDF download failed: {dl_result.error}", "arxiv_id": arxiv_id}
        return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]

    # 3. Parse PDF
    extracted = _pdf_parser.parse(dl_result.local_path, arxiv_id)

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
