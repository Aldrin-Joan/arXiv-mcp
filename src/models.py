"""
Shared Pydantic models and configuration for arxiv-mcp.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────


def get_download_dir() -> Path:
    """Return the current download directory, reading from env each call."""
    default_download_dir = os.getenv("ARXIV_DOWNLOAD_DIR")
    if default_download_dir:
        download_dir = Path(default_download_dir).expanduser().resolve()
    else:
        download_dir = Path(tempfile.gettempdir()) / "arxiv_mcp_downloads"

    download_dir.mkdir(parents=True, exist_ok=True)
    return download_dir


DOWNLOAD_DIR = get_download_dir()

ARXIV_DB_PATH = os.getenv("ARXIV_DB_PATH", str(DOWNLOAD_DIR / "arxiv_mcp.db"))

MAX_RESULTS_DEFAULT = int(os.getenv("ARXIV_MAX_RESULTS", "10"))
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "800"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "100"))
RATE_LIMIT_DELAY = float(
    os.getenv("ARXIV_RATE_LIMIT_DELAY", "3.0")
)  # seconds between API calls
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "60"))
KEEP_PDFS = os.getenv("ARXIV_KEEP_PDFS", "true").lower() in ("1", "true", "yes")
S2_API_KEY = os.getenv("S2_API_KEY", "").strip()

# Hardened config for script performance and safety
MAX_CONCURRENT_DOWNLOADS = int(os.getenv("ARXIV_MAX_CONCURRENT_DOWNLOADS", "4"))
MAX_CONCURRENT_PARSERS = int(os.getenv("ARXIV_MAX_CONCURRENT_PARSERS", "2"))
ARXIV_CACHE_DB = os.getenv(
    "ARXIV_CACHE_DB", ""
)  # optional sqlite path for caching metadata
PDF_MAX_SIZE_MB = int(os.getenv("ARXIV_PDF_MAX_SIZE_MB", "100"))
PDF_MAX_PAGES = int(os.getenv("ARXIV_PDF_MAX_PAGES", "500"))


# ── Pydantic Models ───────────────────────────────────────────────────────────


class Author(BaseModel):
    name: str


class PaperMetadata(BaseModel):
    """Metadata returned from arXiv API for a single paper."""

    arxiv_id: str
    title: str
    authors: list[Author]
    abstract: str
    categories: list[str]
    primary_category: str
    published: str  # ISO date string
    updated: str  # ISO date string
    pdf_url: str
    entry_url: str
    comment: Optional[str] = None
    journal_ref: Optional[str] = None


class TextChunk(BaseModel):
    """A single token-aware chunk of extracted paper text."""

    chunk_index: int
    text: str
    token_count: int
    section_hint: Optional[str] = None  # e.g., "Introduction", "Abstract"


class ExtractedPaper(BaseModel):
    """Full parsed content of a downloaded PDF."""

    arxiv_id: str
    title: str
    total_pages: int
    full_text: str
    chunks: list[TextChunk]
    extraction_method: str = "pymupdf"


class PaperContext(BaseModel):
    """LLM-ready context bundle for a paper."""

    metadata: PaperMetadata
    chunks: list[TextChunk]
    total_tokens: int
    chunk_count: int
    llm_system_prompt: str
    summary_prompt: str


class SearchResult(BaseModel):
    """Lightweight result for listing search hits."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract_snippet: str
    published: str
    categories: list[str]
    pdf_url: str


class DownloadResult(BaseModel):
    """Result of a PDF download operation."""

    arxiv_id: str
    local_path: str
    file_size_bytes: int
    success: bool
    error: Optional[str] = None


class CitationNode(BaseModel):
    """Citation node model for Semantic Scholar graph results."""

    arxiv_id: Optional[str] = None
    s2_id: str
    title: str
    year: Optional[int] = None
    citation_count: int
    is_influential: bool


class CitationGraph(BaseModel):
    """Citation graph for a root paper."""

    root_arxiv_id: str
    root_title: str
    references: list[CitationNode]
    cited_by: list[CitationNode]
    reference_count: int
    citation_count: int
    fetched_at: str


class PaperContributions(BaseModel):
    """Extracted structured contributions for a paper."""

    arxiv_id: str
    core_claim: str
    proposed_method: str
    key_results: list[str]
    baselines_compared: list[str]
    limitations: list[str]
    datasets_used: list[str]
    task_domain: str
    novelty_type: str
    extraction_method: str
    extracted_at: str


class ReadingListEntry(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    year: Optional[int]
    abstract: str
    tags: list[str]
    notes: str
    read_status: Literal["unread", "reading", "read"]
    added_at: datetime
    updated_at: datetime


class ReadingListResult(BaseModel):
    action: str
    entries: Optional[list[ReadingListEntry]] = None
    entry: Optional[ReadingListEntry] = None
    total_count: Optional[int] = None
    stats: Optional[dict] = None
    message: str


class WatchedTopic(BaseModel):
    id: int
    query: str
    label: str
    last_checked: Optional[datetime]
    check_count: int
    created_at: datetime
    new_papers_this_check: Optional[int] = None


class TopicCheckResult(BaseModel):
    topic: WatchedTopic
    new_papers: list[PaperMetadata]
    baseline_established: bool


class WatcherResult(BaseModel):
    action: str
    topics: Optional[list[WatchedTopic]] = None
    check_results: Optional[list[TopicCheckResult]] = None
    message: str


class ExplanationResult(BaseModel):
    arxiv_id: str
    title: str
    audience: str
    what_it_is: str
    problem_solved: str
    how_it_works: str
    why_it_matters: str
    key_result: str
    reading_time_minutes: int
    generation_method: Literal["llm", "passthrough"]
    generated_at: datetime


class CodeLink(BaseModel):
    """A single extracted code or dataset link with classification metadata."""

    url: str
    link_type: Literal[
        "github_repo",
        "huggingface_model",
        "huggingface_dataset",
        "kaggle_dataset",
        "project_page",
        "demo",
        "other",
    ]
    source: Literal[
        "paperswithcode",
        "arxiv_page",
        "pdf_text",
        "pdf_hyperlink",
    ]
    confidence: float


class CodeLinks(BaseModel):
    """Collection of classified source links for an arXiv paper."""

    arxiv_id: str
    github_repos: list[CodeLink]
    huggingface_links: list[CodeLink]
    dataset_links: list[CodeLink]
    project_pages: list[CodeLink]
    other_links: list[CodeLink]
    has_official_code: bool
    fetched_at: datetime


class ReproducibilitySignal(BaseModel):
    """Single heuristic signal for reproducibility scoring."""

    name: str
    points_awarded: float
    points_possible: float
    detected: bool
    evidence: str


class ReproducibilityReport(BaseModel):
    """The reproducibility score report for an arXiv paper."""

    arxiv_id: str
    score: float
    band: str
    signals: list[ReproducibilitySignal]
    code_links: CodeLinks
    scored_at: datetime


class Divergence(BaseModel):
    """Divergence between paper claims and implementation."""

    paper_says: str
    code_does: str
    location: str
    severity: Literal["minor", "moderate", "critical"]


class ImplementationDiff(BaseModel):
    """Comparison report between paper and code."""

    arxiv_id: str
    github_url: str
    paper_title: str
    divergences: list[Divergence]
    faithful_implementations: list[str]
    missing_implementations: list[str]
    overall_fidelity: Literal["high", "medium", "low"]
    summary: str
    code_files_analyzed: list[str]
    total_code_tokens: int
    diffed_at: datetime


class PaperDimension(BaseModel):
    """Dimension detail for paper comparison."""

    dimension: str
    values: dict[str, str]


class ComparisonReport(BaseModel):
    """Comparison report across multiple papers."""

    paper_ids: list[str]
    paper_titles: dict[str, str]
    shared_task_domain: Optional[str] = None
    dimensions: list[PaperDimension]
    conflicting_claims: list[str]
    strongest_results: str
    recommendation: str
    compared_at: str


class SimilarPaper(BaseModel):
    """Semantic similarity candidate paper."""

    arxiv_id: str
    title: str
    similarity_score: float
    year: Optional[int] = None
    abstract_preview: str


class SimilarityResults(BaseModel):
    """Semantic similarity query result."""

    query_arxiv_id: Optional[str] = None
    query_text: Optional[str] = None
    results: list[SimilarPaper]
    index_size: int


# Layer 2 configuration variables
S2_API_KEY: str = os.getenv("S2_API_KEY", "")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")
SEMANTIC_INDEX_DIR: Path = (
    Path(os.getenv("SEMANTIC_INDEX_DIR", str(DOWNLOAD_DIR / "semantic_index")))
    .expanduser()
    .resolve()
)
CITATION_CACHE_TTL: int = int(
    os.getenv("CITATION_CACHE_TTL", str(86400))
)  # 24h in seconds
CONTRIBUTION_CACHE_TTL: int = int(
    os.getenv("CONTRIBUTION_CACHE_TTL", str(604800))
)  # 7d
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# Layer 3 environment variables
GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
LINK_CACHE_TTL: int = int(os.getenv("LINK_CACHE_TTL", str(48 * 3600)))  # seconds
REPRO_CACHE_TTL: int = int(os.getenv("REPRO_CACHE_TTL", str(7 * 24 * 3600)))  # seconds
DIFF_CACHE_TTL: int = int(os.getenv("DIFF_CACHE_TTL", str(24 * 3600)))  # seconds
GITHUB_MAX_FILES: int = int(os.getenv("GITHUB_MAX_FILES", "25"))
GITHUB_MAX_FILE_SIZE_KB: int = int(os.getenv("GITHUB_MAX_FILE_SIZE_KB", "50"))
