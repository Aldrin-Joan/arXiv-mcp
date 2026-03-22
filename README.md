# arXiv MCP (Model Context Protocol)

[![PyPI version](https://img.shields.io/pypi/v/arxiv-mcp-aj)](https://pypi.org/project/arxiv-mcp-aj/) [![Python versions](https://img.shields.io/pypi/pyversions/arxiv-mcp-aj)](https://pypi.org/project/arxiv-mcp-aj/) [![License](https://img.shields.io/pypi/l/arxiv-mcp-aj)](https://pypi.org/project/arxiv-mcp-aj/)

## Overview

`arxiv-mcp` is a production-grade Model Context Protocol (MCP) server focused on arXiv research retrieval.
It provides:

- arXiv Atom API search by ID/query
- PDF download, validation, and cache
- PDF text extraction (title, abstract, body, references)
- Token-aware context chunking for LLM pipelines
- CLI, API, and autonomous agent integration support

---

## Table of Contents

1. [Quickstart](#quickstart)
2. [Installation](#installation)
3. [Usage](#usage)
4. [MCP Server](#mcp-server)
5. [Project structure](#project-structure)
6. [Configuration](#configuration)
7. [Testing](#testing)
8. [Troubleshooting](#troubleshooting)
9. [Contributing](#contributing)
10. [License](#license)

---

## Quickstart

### 1. Clone repository

```bash
git clone https://github.com/blazickjp/arxiv-mcp-server.git
cd arxiv-mcp
```

### 2. Set up Python environment (recommended)

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run smoke test

```bash
python test_smoke.py
```

---

## Installation

From source:

```bash
pip install -e .
```

From PyPI:

```bash
pip install arxiv-mcp-aj
```

---

## Usage

### CLI

```bash
arxiv-mcp --help
```

Run server locally:

```bash
python -m src.mcp_server
```

### Python API

```python
from src.arxiv_client import ArxivClient
from src.pdf_fetcher import PdfFetcher
from src.pdf_parser import PdfParser
from src.context_builder import ContextBuilder

client = ArxivClient()
results = client.search('quantum computing', max_results=3)

pdf_path = PdfFetcher().fetch_paper(results[0].id)
parsed = PdfParser().parse(pdf_path)

context = ContextBuilder().build(parsed)
print(context.summary)
```

---

## Architecture Layers

| Layer | Features |
|---|---|
| **Layer 1 — retrieval** *(both tools have this)* | Search · PDF fetch + cache · Text extraction + chunking |
| **Layer 2 — intelligence** *(your opportunity)* | Citation graph · Concept extraction · Cross-paper synthesis |
| **Layer 3 — dev tooling** *(highly unique)* | Code + dataset links · Implementation diff · Reproducibility audit |
| **Layer 4 — research workflows** *(unique)* | Reading lists · Topic tracking + alerts · Agent-ready Q&A |

## MCP Server

`src/mcp_server/__main__.py` starts an MCP tool server exposing:

- `arxiv_search` (query or ID expand)
- `arxiv_fetch_pdf` (download + cache)
- `arxiv_parse_pdf` (extract text and metadata)
- `arxiv_build_context` (chunk to LLM-friendly context)
- `arxiv_citation_graph` (author/paper citation network)
- `arxiv_extract_contributions` (structured contribution extractor)
- `arxiv_semantic_index` (semantic similarity index builder/query)
- `arxiv_compare_papers` (paper comparison report)
- `arxiv_extract_code_links` (discover official GitHub/HuggingFace/Kaggle links from a paper)
- `arxiv_reproducibility_score` (reproducibility heuristic score with evidence details)
- `arxiv_diff_implementations` (compare paper method claims against a GitHub implementation)
- `arxiv_reading_list` (persistent reading list CRUD and filters)
- `arxiv_watch_topic` (watch query topics and detect new papers)
- `arxiv_explain_for_audience` (audience-specific explanation synthesis)

Use any MCP-capable client (VS Code MCP extension, custom agent SDK) to connect.

### VS Code MCP server setup

In VS Code, add an MCP server entry to your workspace settings (e.g., `.vscode/settings.json`):

```json
{
  "servers": {
    "arxiv-mcp": {
      "command": "D:/Softwares/Anaconda3/python.exe",
      "args": ["-m", "src.mcp_server"],
      "cwd": "${workspaceFolder}",
      "env": {
        "PYTHONPATH": "${workspaceFolder}",
        "ARXIV_DOWNLOAD_DIR": "${workspaceFolder}/downloads",
        "ARXIV_KEEP_PDFS": "true",
        "CHUNK_SIZE_TOKENS": "800",
        "CHUNK_OVERLAP_TOKENS": "100",
        "ARXIV_RATE_LIMIT_DELAY": "3.0",
        "MAX_RETRIES": "3",
        "HTTP_TIMEOUT": "60"
      }
    }
  }
}
```

- `ARXIV_DOWNLOAD_DIR`: local storage for downloaded PDFs.
- `ARXIV_KEEP_PDFS`: keep cached PDFs after parse.
- `CHUNK_SIZE_TOKENS` / `CHUNK_OVERLAP_TOKENS`: controls text-chunking in context builder.
- `ARXIV_RATE_LIMIT_DELAY`: delay between arXiv API calls.
- `MAX_RETRIES`, `HTTP_TIMEOUT`: network robustness.

You can apply this configuration also in other compatible MCP clients using their server configuration schema.

---

## Project structure

- `src/` - package source
  - `arxiv_client/` - arXiv Atom API logic
  - `pdf_fetcher/` - download/cache PDF
  - `pdf_parser/` - extract/clean PDF text
  - `context_builder/` - tokenization + chunking
  - `mcp_server/` - MCP protocol/adapters
- `tests/` - pytest suite
- `requirements.txt` - dependencies
- `pyproject.toml` - package metadata

---

## Configuration

Environment variables:

- `ARXIV_CACHE_DIR` (default: `./downloads`)
- `ARXIV_CACHE_TTL` (default: `604800` seconds / 7 days)
- `ARXIV_DB_PATH` (default: `${ARXIV_DOWNLOAD_DIR}/arxiv_mcp.db`) path to the SQLite workflow database
- `ARXIV_RATE_LIMIT` (default: `1` request/sec)
- `S2_API_KEY` (optional; Semantic Scholar API key for higher rate limits)
- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OLLAMA_MODEL` (default: `mistral`)
- `SEMANTIC_INDEX_DIR` (default: `${ARXIV_DOWNLOAD_DIR}/semantic_index`)
- `CITATION_CACHE_TTL` (default: `86400` seconds / 24 hours)
- `CONTRIBUTION_CACHE_TTL` (default: `604800` seconds / 7 days)
- `EMBEDDING_MODEL` (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `GITHUB_TOKEN` (optional; for GitHub API auth, improves 60 -> 5000 req/hour)
- `LINK_CACHE_TTL` (default: `172800` seconds / 48 hours)
- `REPRO_CACHE_TTL` (default: `604800` seconds / 7 days)
- `DIFF_CACHE_TTL` (default: `86400` seconds / 24 hours)
- `GITHUB_MAX_FILES` (default: `20`)
- `GITHUB_MAX_FILE_SIZE_KB` (default: `50`)

Set in shell or via `.env` before running.

---

## Testing

Run full tests:

```bash
pytest -q
```

Smoke test:

```bash
python test_smoke.py
```

---

## Troubleshooting

- `arxiv-mcp` command not found: ensure virtualenv is active and package installed
- PDF download failure: check network access to `https://arxiv.org/pdf/`
- Rate-limit errors: lower request frequency or adjust `ARXIV_RATE_LIMIT`

---

## Contributing

1. Fork repo
2. Create feature branch
3. Add tests and update README
4. Open PR

Follow style checks (Black, formatting and lint).

---

## License

Apache-2.0
