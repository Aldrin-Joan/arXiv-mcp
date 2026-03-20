"""
pdf_parser — extract structured text from arXiv PDFs using PyMuPDF (fitz).

Responsibilities:
  - Open and validate PDFs
  - Extract full text with page awareness
  - Detect section headers heuristically
  - Return ExtractedPaper with full_text and page breakdown
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from src.models import ExtractedPaper, TextChunk, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS
from src.logger import get_logger

log = get_logger("pdf_parser")

# Common academic section header patterns
_SECTION_PATTERNS = [
    re.compile(r"^\d+\.?\s+[A-Z][A-Za-z ]{3,60}$"),          # "1. Introduction"
    re.compile(r"^[A-Z][A-Z\s]{4,40}$"),                       # "ABSTRACT", "CONCLUSION"
    re.compile(r"^(?:Abstract|Introduction|Conclusion|References|Appendix|Methodology|Results|Discussion|Related Work|Background|Experiments?|Evaluation)\b", re.IGNORECASE),
]


def _looks_like_section_header(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 80:
        return False
    return any(p.match(line) for p in _SECTION_PATTERNS)


def _clean_text(text: str) -> str:
    """Basic cleanup of PDF-extracted text."""
    # Remove excessive whitespace while preserving paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse more than 2 consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lines that are just page numbers or hyphens
    text = re.sub(r"^\s*[-–—]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    # Fix hyphenated line breaks common in PDFs
    text = re.sub(r"-\n(?=[a-z])", "", text)
    return text.strip()


class PDFParser:
    """
    Extracts and structures text from a PDF file.
    """

    def parse(self, pdf_path: str, arxiv_id: str) -> ExtractedPaper:
        """
        Parse a PDF and return an ExtractedPaper with full text and chunks.

        Args:
            pdf_path: Absolute path to the local PDF file.
            arxiv_id: arXiv ID (used as identifier in the result).

        Returns:
            ExtractedPaper with full_text, chunks, and metadata.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If the file is not a valid PDF.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        log.info("Parsing PDF", path=pdf_path, arxiv_id=arxiv_id)

        try:
            doc = fitz.open(str(path))
        except Exception as exc:
            raise ValueError(f"Cannot open PDF '{pdf_path}': {exc}") from exc

        if doc.page_count == 0:
            raise ValueError(f"PDF has no pages: {pdf_path}")

        total_pages = doc.page_count
        page_texts: list[str] = []

        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text("text")  # type: ignore[arg-type]
            page_texts.append(text)

        doc.close()

        full_raw = "\n\n".join(page_texts)
        full_text = _clean_text(full_raw)

        # Extract title from first ~200 chars heuristically
        first_block = full_text[:500]
        title = _extract_title_heuristic(first_block)

        log.info(
            "PDF parsed",
            arxiv_id=arxiv_id,
            pages=total_pages,
            chars=len(full_text),
            title_hint=title[:60] if title else "unknown",
        )

        chunks = _chunk_text(full_text)

        return ExtractedPaper(
            arxiv_id=arxiv_id,
            title=title or arxiv_id,
            total_pages=total_pages,
            full_text=full_text,
            chunks=chunks,
            extraction_method="pymupdf",
        )


def _extract_title_heuristic(text: str) -> str:
    """
    Attempt to extract the paper title from the first block of text.
    Titles tend to be the longest non-author lines near the top.
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    # Filter out very short lines (page numbers, single words)
    candidates = [ln for ln in lines[:15] if 10 < len(ln) < 200]
    if candidates:
        # Prefer longer lines that don't look like author lists (no commas)
        for c in candidates:
            if "," not in c and not re.match(r"^\d", c):
                return c
        return candidates[0]
    return ""


def _chunk_text(text: str) -> list[TextChunk]:
    """
    Split full_text into token-aware overlapping chunks.

    Uses tiktoken for accurate token counting (cl100k_base = GPT-4 tokenizer).
    Falls back to character-based estimate if tiktoken fails.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")

        def count_tokens(t: str) -> int:
            return len(enc.encode(t))

        def tokens_to_text_slice(tokens: list[int], start: int, end: int) -> str:
            return enc.decode(tokens[start:end])

        all_tokens = enc.encode(text)
        total = len(all_tokens)
        chunks: list[TextChunk] = []
        idx = 0
        chunk_index = 0

        while idx < total:
            end = min(idx + CHUNK_SIZE_TOKENS, total)
            chunk_tokens = all_tokens[idx:end]
            chunk_text = enc.decode(chunk_tokens)

            # Detect section hint from first line
            first_line = chunk_text.strip().split("\n")[0]
            hint = first_line[:80] if _looks_like_section_header(first_line) else None

            chunks.append(
                TextChunk(
                    chunk_index=chunk_index,
                    text=chunk_text,
                    token_count=len(chunk_tokens),
                    section_hint=hint,
                )
            )

            chunk_index += 1
            step = CHUNK_SIZE_TOKENS - CHUNK_OVERLAP_TOKENS
            idx += step

        log.debug("Text chunked", chunks=len(chunks), total_tokens=total)
        return chunks

    except Exception as exc:
        log.warning("tiktoken chunking failed, falling back to char-based", error=str(exc))
        return _chunk_text_chars(text)


def _chunk_text_chars(text: str) -> list[TextChunk]:
    """Character-based chunking fallback (~4 chars per token estimate)."""
    char_size = CHUNK_SIZE_TOKENS * 4
    char_overlap = CHUNK_OVERLAP_TOKENS * 4
    chunks: list[TextChunk] = []
    idx = 0
    chunk_index = 0

    while idx < len(text):
        end = min(idx + char_size, len(text))
        chunk_text = text[idx:end]
        chunks.append(
            TextChunk(
                chunk_index=chunk_index,
                text=chunk_text,
                token_count=len(chunk_text) // 4,
                section_hint=None,
            )
        )
        chunk_index += 1
        idx += char_size - char_overlap

    return chunks
