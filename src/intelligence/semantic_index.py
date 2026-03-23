from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import asyncio
import concurrent.futures

from sentence_transformers import SentenceTransformer

from src.models import (
    EMBEDDING_MODEL,
    SEMANTIC_INDEX_DIR,
    SimilarPaper,
    SimilarityResults,
)
from src.pdf_fetcher import PDFFetcher
from src.pdf_parser import PDFParser
from src.logger import get_logger

log = get_logger("semantic_index")


class SemanticIndex:
    """Local semantic index using ChromaDB and sentence-transformers."""

    def __init__(self) -> None:
        self._store_dir = Path(SEMANTIC_INDEX_DIR)
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._collection = None
        self._model: Optional[SentenceTransformer] = None

        self._chromadb = None
        self._chroma_settings = None

    def _initialize_chroma(self) -> None:
        if self._collection is not None:
            return

        if self._chromadb is None:
            try:
                import chromadb
                from chromadb.config import Settings
            except Exception as exc:
                raise RuntimeError(
                    "chromadb is required for SemanticIndex; install with chromadb==0.5.0"
                ) from exc
            self._chromadb = chromadb
            self._chroma_settings = Settings

        if self._client is None:
            self._client = self._chromadb.PersistentClient(
                path=str(self._store_dir), settings=self._chroma_settings()
            )

        self._collection = self._client.get_or_create_collection(
            name="arxiv_abstracts",
            metadata={"distance_metric": "cosine"},
            embedding_function=None,
        )

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(EMBEDDING_MODEL)
            log.info("Loaded embedding model", model=EMBEDDING_MODEL)
        return self._model

    def _run_sync(self, coro_or_func):
        if asyncio.iscoroutine(coro_or_func):
            coro = coro_or_func
        elif callable(coro_or_func):
            result = coro_or_func()
            if asyncio.iscoroutine(result):
                coro = result
            else:
                return result
        else:
            raise TypeError(
                "_run_sync expects a coroutine or callable returning a coroutine"
            )

        try:
            asyncio.get_running_loop()

            def _runner():
                return asyncio.run(coro)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(_runner).result()
        except RuntimeError:
            return asyncio.run(coro)

    def _get_collection(self):
        self._initialize_chroma()
        if self._collection is None:
            raise RuntimeError("Failed to initialize Chroma collection")
        return self._collection

    def add_paper(
        self,
        arxiv_id: str,
        title: str,
        abstract: str,
        year: Optional[int] = None,
    ) -> None:
        if not arxiv_id or not title or abstract is None:
            raise ValueError("arxiv_id, title, and abstract are required")

        collection = self._get_collection()
        model = self._load_model()
        text = f"{title}. {abstract}"
        embedding = model.encode(text)
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()

        metadata = {
            "title": title,
            "abstract_preview": abstract[:300],
        }
        if year is not None:
            metadata["year"] = year

        collection.upsert(
            ids=[arxiv_id],
            metadatas=[metadata],
            documents=[text],
            embeddings=[embedding],
        )

    def index_size(self) -> int:
        collection = self._get_collection()
        return collection.count()

    def _to_similarity_results(
        self,
        query_arxiv_id: Optional[str],
        query_text: Optional[str],
        records: List[Dict[str, Any]],
        distances: List[float],
    ) -> SimilarityResults:
        results: List[SimilarPaper] = []
        for rec, dist in zip(records, distances):
            results.append(
                SimilarPaper(
                    arxiv_id=rec["id"],
                    title=rec["metadata"].get("title", ""),
                    similarity_score=1.0 - dist,
                    year=rec["metadata"].get("year"),
                    abstract_preview=rec["metadata"].get("abstract_preview", ""),
                )
            )

        return SimilarityResults(
            query_arxiv_id=query_arxiv_id,
            query_text=query_text,
            results=results,
            index_size=self.index_size(),
        )

    def query_by_paper(self, arxiv_id: str, top_k: int = 10) -> SimilarityResults:
        collection = self._get_collection()

        try:
            item = collection.get(ids=[arxiv_id], include=["metadatas", "embeddings"])
            if not item["ids"]:
                raise ValueError("Paper not found in index")
            emb = item["embeddings"][0]
        except Exception:
            log.info("Paper missing in index, cold-starting", arxiv_id=arxiv_id)

            async def _build_and_query():
                async with PDFFetcher() as fetcher:
                    dl = await fetcher.download(arxiv_id)
                if not dl.success:
                    raise RuntimeError(f"Failed to download paper {arxiv_id}")
                parsed = PDFParser().parse(dl.local_path, arxiv_id)
                self.add_paper(arxiv_id, parsed.title, parsed.full_text, None)
                item2 = collection.get(
                    ids=[arxiv_id], include=["metadatas", "embeddings"]
                )
                emb2 = item2["embeddings"][0]
                return emb2

            emb = self._run_sync(_build_and_query)

        query_results = collection.query(
            query_embeddings=[emb],
            n_results=top_k + 1,
            include=["metadatas", "distances", "documents"],
        )

        out_records = []
        out_distances = []
        for idx, rec_id in enumerate(query_results["ids"][0]):
            if rec_id == arxiv_id:
                continue
            out_records.append(
                {
                    "id": rec_id,
                    "metadata": query_results["metadatas"][0][idx],
                }
            )
            out_distances.append(query_results["distances"][0][idx])
            if len(out_records) >= top_k:
                break

        return self._to_similarity_results(arxiv_id, None, out_records, out_distances)

    def query_by_text(self, query: str, top_k: int = 10) -> SimilarityResults:
        if not query.strip():
            raise ValueError("query text must not be empty")

        collection = self._get_collection()
        model = self._load_model()
        embedding = model.encode(query)
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()

        query_results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["metadatas", "distances", "documents"],
        )

        records = []
        distances = []
        for idx, rec_id in enumerate(query_results["ids"][0]):
            records.append(
                {
                    "id": rec_id,
                    "metadata": query_results["metadatas"][0][idx],
                }
            )
            distances.append(query_results["distances"][0][idx])

        return self._to_similarity_results(None, query, records, distances)
