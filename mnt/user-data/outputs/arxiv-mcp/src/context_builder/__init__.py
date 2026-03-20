"""
context_builder — assemble LLM-ready context from paper metadata + extracted text.

Produces PaperContext objects containing:
  - Structured metadata
  - Token-aware chunks
  - Ready-to-use system prompt
  - Suggested summarization prompt
"""
from __future__ import annotations

from src.models import ExtractedPaper, PaperMetadata, PaperContext, TextChunk
from src.logger import get_logger

log = get_logger("context_builder")

_SYSTEM_PROMPT_TEMPLATE = """\
You are a research assistant analyzing the following arXiv paper.

## Paper Metadata
- **Title**: {title}
- **Authors**: {authors}
- **Published**: {published}
- **arXiv ID**: {arxiv_id}
- **Categories**: {categories}
- **PDF**: {pdf_url}

## Abstract
{abstract}

## Instructions
You have access to the full text of this paper split into {chunk_count} chunks \
({total_tokens} total tokens). Use this content to answer questions accurately. \
When citing specific claims, reference them as coming from this paper.
"""

_SUMMARY_PROMPT_TEMPLATE = """\
Please provide a comprehensive summary of the arXiv paper "{title}" ({arxiv_id}).

Your summary should cover:
1. **Problem Statement** — What problem does this paper address?
2. **Key Contributions** — What are the main contributions or findings?
3. **Methodology** — How did the authors approach the problem?
4. **Results** — What were the key results and metrics?
5. **Limitations** — What limitations or future work do the authors identify?
6. **Relevance** — Who would benefit most from reading this paper?

Base your summary on the provided paper content.
"""


class ContextBuilder:
    """
    Combines metadata and extracted text into LLM-consumable context bundles.
    """

    def build(
        self,
        metadata: PaperMetadata,
        extracted: ExtractedPaper,
        max_chunks: int | None = None,
    ) -> PaperContext:
        """
        Build a PaperContext from metadata and extracted PDF content.

        Args:
            metadata: Paper metadata from arXiv API.
            extracted: Extracted text and chunks from PDF parser.
            max_chunks: Optional limit on number of chunks to include.

        Returns:
            PaperContext ready for LLM consumption.
        """
        chunks = extracted.chunks
        if max_chunks is not None:
            chunks = chunks[:max_chunks]

        total_tokens = sum(c.token_count for c in chunks)
        chunk_count = len(chunks)

        authors_str = ", ".join(a.name for a in metadata.authors[:5])
        if len(metadata.authors) > 5:
            authors_str += f" et al. (+{len(metadata.authors) - 5} more)"

        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            title=metadata.title,
            authors=authors_str,
            published=metadata.published[:10] if metadata.published else "unknown",
            arxiv_id=metadata.arxiv_id,
            categories=", ".join(metadata.categories),
            pdf_url=metadata.pdf_url,
            abstract=metadata.abstract,
            chunk_count=chunk_count,
            total_tokens=total_tokens,
        )

        summary_prompt = _SUMMARY_PROMPT_TEMPLATE.format(
            title=metadata.title,
            arxiv_id=metadata.arxiv_id,
        )

        context = PaperContext(
            metadata=metadata,
            chunks=chunks,
            total_tokens=total_tokens,
            chunk_count=chunk_count,
            llm_system_prompt=system_prompt,
            summary_prompt=summary_prompt,
        )

        log.info(
            "Context built",
            arxiv_id=metadata.arxiv_id,
            chunks=chunk_count,
            total_tokens=total_tokens,
        )
        return context

    def get_chunk_window(
        self,
        context: PaperContext,
        start: int = 0,
        count: int = 5,
    ) -> list[TextChunk]:
        """
        Return a sliding window of chunks for multi-turn LLM conversations.

        Args:
            context: Previously built PaperContext.
            start: Starting chunk index (0-based).
            count: Number of chunks to return.

        Returns:
            List of TextChunk objects.
        """
        end = min(start + count, len(context.chunks))
        return context.chunks[start:end]

    def chunks_to_text(self, chunks: list[TextChunk]) -> str:
        """Concatenate chunk texts with section hints as headers."""
        parts: list[str] = []
        for chunk in chunks:
            if chunk.section_hint:
                parts.append(f"\n## {chunk.section_hint}\n")
            parts.append(chunk.text)
        return "\n\n".join(parts)
