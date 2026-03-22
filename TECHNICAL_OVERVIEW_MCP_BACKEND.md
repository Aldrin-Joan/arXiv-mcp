# arXiv MCP Backend Technical Deep Dive

This document explains the full backend architecture of `arxiv-mcp` (Model Context Protocol server) with high technical fidelity. It is written for engineering leadership and provides end-to-end flow, data transformation, caching, resiliency, and deployment behavior.

---

## 1. Core Objective

`arxiv-mcp` is a specialized MCP tool server that:

- Receives tool invocations (`search_arxiv`, `get_paper_by_id`, `download_pdf`, `extract_text`, `get_paper_context`) over the MCP protocol.
- Queries arXiv metadata via the official `arxiv` Python library.
- Downloads PDF content from `https://arxiv.org/pdf/{id}.pdf` with retry and cache.
- Parses PDF text using PyMuPDF and token-aware chunking for LLM context.
- Builds structured LLM-ready prompt bundles (`PaperContext`).

This is designed to be consumed by MVC-style agent pipelines (MCP enabled clients, VS Code, CLI, or custom tool orchestration).

---

## 2. Package Layout

Root code lives in `src/`:

- `src/arxiv_client/`: metadata search and ID validation logic.
- `src/pdf_fetcher/`: downloads PDFs with caching and retry.
- `src/pdf_parser/`: PDF-to-text extraction + chunking.
- `src/context_builder/`: builds LLM prompts and context structures.
- `src/mcp_server/`: MCP server bindings and tool endpoint wiring.
- `src/models.py`: Pydantic models and global constants.
- `src/cache.py`: simple metadata cache.
- `src/logger.py`: structured logging config.

### Layer2 Intelligence Extensions
- `src/intelligence/citation_graph.py`: fetches and caches citation graphs from Semantic Scholar API.
- `src/intelligence/contribution_extractor.py`: extracts contributions using heuristic rules and Ollama LLM fallback.
- `src/intelligence/semantic_index.py`: builds / queries semantic vector index using sentence-transformers and optional ChromaDB.
- `src/intelligence/paper_comparator.py`: compares two paper contexts on contributions, citation metrics, and similarity.
- `src/mcp_server/__init__.py`: exposes new MCP tools `arxiv_citation_graph`, `arxiv_extract_contributions`, `arxiv_semantic_index`, `arxiv_compare_papers`, plus new Layer 3 tools `arxiv_extract_code_links`, `arxiv_reproducibility_score`, `arxiv_diff_implementations`.

### Layer 3 Devtools
- `src/devtools/link_extractor.py`: multi-source code/dataset/model link extraction from PapersWithCode, arXiv abstract page, and PDF content; link normalization, classification, and caching.
- `src/devtools/reproducibility_scorer.py`: deterministic reproducibility scoring based on link signals, hyperparameters, ablation, seeds, compute environment and license.
- `src/devtools/implementation_differ.py`: GitHub repository analysis and method-vs-code divergence report full pipeline.
- `src/devtools/prompts/diff.txt`: LLM prompt template used for implementation fidelity analysis.

Supporting files in root:

- `requirements.txt` (dependencies: arxiv, httpx, PyMuPDF, tenacity, mcp, etc.)
- `pyproject.toml` (package metadata)
- `tests/` (pytest suite).

---

## 3. Runtime Behavior and Data Flow

### 3.1 MCP Protocol Entrypoint

- `python -m src.mcp_server` starts at `src/mcp_server/__main__.py` (calls `main()`).
- `main()` creates STDIO-based MCP server via `mcp.server.stdio.std::stdio_server`.
- Each incoming request is routed by `server.call_tool()` to `_handle_*` handler.

### 3.2 Tool Definitions (`@server.list_tools`)

Defines 5 exposed tools plus JSON schema for arguments. That enables client discovery and auto-complete.

### 3.3 Tool Execution Flow

#### `search_arxiv`:
- Uses `ArxivClient.search(query, max_results)`.
- ID detection: if query contains arXiv ID (`detect_arxiv_id`) then delegates to `get_by_id` for exact metadata.
- Produces `SearchResult` list.

#### `get_paper_by_id`:
- Calls `ArxivClient.get_by_id(arxiv_id)`.
- Caches metadata in `src/cache`.

#### `download_pdf`:
- Uses `PDFFetcher.download(arxiv_id, force=False)`.
- Checks `_is_cached(arxiv_id)` by local file existence and size.
- On miss, calls `_download_with_retry` with exponential backoff and strong validation: HTTP 404 special-case, content-type PDF/octe stream, size cap.

#### `extract_text`:
- Ensures PDF present via `PDFFetcher.download`.
- Calls `PDFParser.parse(pdf_path, arxiv_id)`.
- Optionally deletes PDF from disk if `KEEP_PDFS` is False (config item in `src/models`).
- Returns limited extraction summary with chunk previews and no full text payload, to keep response manageable.

#### `get_paper_context`:
- Combines all steps in pipeline:
  1. `get_by_id` metadata
  2. `download_pdf`
  3. `PDFParser.parse`
  4. `ContextBuilder.build`
- Returns metadata + system prompt + summary prompt + all chunk objects.

---

## 4. Key Components

### 4.1 ArxivClient (`src/arxiv_client/__init__.py`)

- Rate limiting global variable `_last_call_time` + async lock to avoid rapid arXiv hits.
- Uses `le: arxiv.Client(page_size=50, delay_seconds=RATE_LIMIT_DELAY)` with internal retried requests.
- Text normalization/ID regex ensures robust ID handling (e.g., `arxiv:1234.56789v3`).
- Caches paper metadata in `src/cache` to reduce API traffic.

### 4.2 PDF Fetcher (`src/pdf_fetcher/__init__.py`)

- Asynchronous `httpx.AsyncClient` with timeout config from `src.models`.
- Local caching path derived from `get_download_dir()` (rename slash characters to underscores: `cs/1201.2345` -> `cs_1201.2345.pdf`).
- Multi-attempt retry policy with 2^attempt sleep, up to `MAX_RETRIES`.
- Size check via `PDF_MAX_SIZE_MB`.

### 4.3 PDF Parser (`src/pdf_parser/__init__.py`)

- Uses `fitz.open(...)` to read PDF, with page count ceiling `PDF_MAX_PAGES`.
- Text clean-up: normalize whitespace, remove page numbers/hyphenation, section heading heuristics.
- Pipeline: full text -> `ExtractedPaper`.

### 4.4 Chunking

- Primary: `tiktoken.cl100k_base` tokenizer.
- CHUNK_SIZE_TOKENS and CHUNK_OVERLAP_TOKENS from env/config.
- Creates overlapped chunks with heuristic `section_hint` from headers.
- Fallback: char-based chunking when `tiktoken` missing.

### 4.5 Context Builder (`src/context_builder/__init__.py`)

- Builds `PaperContext` with:
  - metadata (from arxiv)
  - chunks (from parser)
  - `llm_system_prompt` (populates template with paper data)
  - `summary_prompt` (one canonical prompt for LLM summarization)
- Supports windowing/chunk substring helpers for multistep QA.

### 4.6 Models (`src/models.py`)

- Defines Pydantic models: `PaperMetadata`, `SearchResult`, `DownloadResult`, `ExtractedPaper`, `TextChunk`, `PaperContext`.
- Defines default constants and environment variable reads.
- Global config: `DOWNLOAD_DIR`, `PDF_MAX_PAGES`, `CHUNK_SIZE_TOKENS`, etc.
- Maintains `KEEP_PDFS` boolean for temporary cleanup.

### 4.7 Cache (`src/cache.py`)

- Simple JSON file-based or in-memory cache to store metadata objects keyed by arXiv ID.
- TTL and invalidation usually prepared here (e.g., 7 days) so repeated calls are local.

### 4.8 Logging (`src/logger.py`)

- Structured logging with `python-json-logger` style via `logging` module.
- All major steps log info/warn/error with context fields.

---

## 5. Deployment & Configuration

### 5.1 Environment Variables

Key variables (in docs and `src/models`):

- `ARXIV_DOWNLOAD_DIR` (default `./downloads`)
- `ARXIV_KEEP_PDFS` (default `false`) toggles deletion after parsing
- `CHUNK_SIZE_TOKENS` (default 800)
- `CHUNK_OVERLAP_TOKENS` (default 100)
- `ARXIV_RATE_LIMIT_DELAY` (default 3.0 sec)
- `MAX_RETRIES` (default 3)
- `HTTP_TIMEOUT` (default 60 seconds)
- `PDF_MAX_SIZE_MB`, `PDF_MAX_PAGES`, others.

### 5.2 Process Start

- `python -m src.mcp_server` directly launches.
- Also available as console script `arxiv-mcp` (entry points in `setup.py/pyproject`.

### 5.3 MCP Client Integration

- Works with any MCP client (VS Code: `.vscode/settings.json` entry under `servers`).
- Data exchanged as `types.TextContent` carrying JSON payload.

---

## 6. Resiliency and Observability

- Retry + exponential backoff for arXiv metadata and PDF downloads.
- Rate limiting protects API/abides by arXiv policy.
- Error propagation in MCP: exceptions converted to consistent JSON error objects.
- Chunking fallback ensures core feature works even when optional token libs are unavailable.

---

## 7. Example End-to-End Request Sequence

1. Client calls `search_arxiv` with query `quantum computing`.
2. Server executes arXiv API query via `ArxivClient.search`, returns metadata.
3. Client calls `get_paper_context` for selected ID.
4. Workflow:
   - `get_paper_by_id` metadata (cache or arXiv API)
   - `download_pdf` to local disk
   - `parse` to cleaned text + chunks
   - `build` to `PaperContext` with prompt templates
5. Output JSON contains `llm_system_prompt`, `summary_prompt`, and chunk array.

---

## 8. Code Safety Notes

- HTTP I/O is async and non-blocking (`httpx.AsyncClient`, `asyncio`).
- File I/O paths are sanitized through `arxiv_id` normalization and replacement of invalid path chars.
- PDF parser enforces maximum page limit to avoid deeply huge files.

---

## 9. Recommended Improvements (non-breaking roadmap ideas)

- Persist metadata cache in Redis / SQLite for high-throughput multi-instance deployments.
- Add parallel chunk retrieval in server for user request latency bound.
- Add optional coherent page boundaries in chunk metadata.
- Add an explicit tool `refresh_cache` / `clear_cache`.
- Add pruning job for old downloaded PDFs when `KEEP_PDFS` is true.

---

## 10. Quick Developer Onboarding Snippets

Run locally:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.mcp_server
```

Test a call (MCP JSON format, example client code is in tests):

```python
from mcp.client import Client
client = Client.from_stdio()  # or raw socket
payload = client.call_tool('search_arxiv', {'query': '2603.17216', 'max_results': 1})
print(payload)
```

---

## 11. References in code

- MCP server orchestration: `src/mcp_server/__init__.py`
- arXiv API wrapper: `src/arxiv_client/__init__.py`
- PDF download: `src/pdf_fetcher/__init__.py`
- PDF parsing + chunking: `src/pdf_parser/__init__.py`
- Context builder: `src/context_builder/__init__.py`
- Models + defaults: `src/models.py`
