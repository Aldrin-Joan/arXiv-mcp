"""Topic watcher manager for Layer 4 workflows."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.models import TopicCheckResult, WatcherResult, WatchedTopic
from src.workflows.db import DatabaseClient


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TopicWatcher:
    """Manage watched research topics and new paper polling."""

    def __init__(self, db: DatabaseClient, arxiv_client: Any) -> None:
        self.db = db
        self.arxiv_client = arxiv_client

    async def add(self, query: str, label: str) -> WatcherResult:
        query = (query or "").strip()
        label = (label or "").strip()
        if not query or not label:
            raise ValueError("query and label are required")

        results = await self.arxiv_client.search(query, max_results=1)
        if not results:
            return WatcherResult(
                action="add", message="Query returned no results — topic not saved"
            )

        now = _now_iso()
        self.db.execute(
            "INSERT INTO watched_topics (query, label, created_at) VALUES (?, ?, ?)",
            (query, label, now),
        )

        row = self.db.fetchone(
            "SELECT * FROM watched_topics WHERE id = last_insert_rowid()", ()
        )
        topic = self._row_to_topic(row)
        return WatcherResult(action="add", topics=[topic], message="Topic saved")

    async def remove(self, topic_id: int) -> WatcherResult:
        row = self.db.fetchone("SELECT * FROM watched_topics WHERE id = ?", (topic_id,))
        if not row:
            return WatcherResult(action="remove", message="Topic not found")

        self.db.execute("DELETE FROM watched_topics WHERE id = ?", (topic_id,))
        return WatcherResult(action="remove", message="Topic removed")

    async def list(self) -> WatcherResult:
        rows = self.db.fetchall(
            "SELECT * FROM watched_topics ORDER BY created_at DESC", ()
        )
        topics = [self._row_to_topic(row) for row in rows]
        return WatcherResult(action="list", topics=topics, message="OK")

    def _row_to_topic(self, row: Dict[str, Any]) -> WatchedTopic:
        if row is None:
            raise ValueError("row is required")

        return WatchedTopic(
            id=int(row["id"]),
            query=row["query"],
            label=row["label"],
            last_checked=(
                datetime.fromisoformat(row["last_checked"])
                if row.get("last_checked")
                else None
            ),
            check_count=int(row.get("check_count", 0)),
            created_at=datetime.fromisoformat(row["created_at"]),
            new_papers_this_check=None,
        )

    async def _check_one(self, topic: WatchedTopic) -> TopicCheckResult:
        papers = await self.arxiv_client.search(topic.query, max_results=50)
        existing_rows = self.db.fetchall(
            "SELECT arxiv_id FROM watched_topic_seen WHERE topic_id = ?", (topic.id,)
        )
        existing_ids = {row["arxiv_id"] for row in existing_rows}

        # Convert SearchResult to PaperMetadata-like dict for model compatibility
        from src.models import PaperMetadata, Author

        def to_metadata(paper):
            if isinstance(paper, PaperMetadata):
                return paper
            # fallback from SearchResult
            return PaperMetadata(
                arxiv_id=paper.arxiv_id,
                title=getattr(paper, "title", ""),
                authors=[Author(name=a) for a in getattr(paper, "authors", [])],
                abstract=getattr(paper, "abstract_snippet", ""),
                categories=getattr(paper, "categories", []),
                primary_category=(getattr(paper, "categories", [""])[0] if getattr(paper, "categories", []) else ""),
                published=getattr(paper, "published", ""),
                updated=getattr(paper, "published", ""),
                pdf_url=getattr(paper, "pdf_url", ""),
                entry_url=f"https://arxiv.org/abs/{paper.arxiv_id}",
            )

        papers = [to_metadata(p) for p in papers]

        if topic.last_checked is None:
            for p in papers:
                self.db.execute(
                    "INSERT OR IGNORE INTO watched_topic_seen (topic_id, arxiv_id, seen_at) VALUES (?, ?, ?)",
                    (topic.id, p.arxiv_id, _now_iso()),
                )
            self.db.execute(
                "UPDATE watched_topics SET last_checked = ?, check_count = check_count + 1 WHERE id = ?",
                (_now_iso(), topic.id),
            )
            checked_topic = self.db.fetchone(
                "SELECT * FROM watched_topics WHERE id = ?", (topic.id,)
            )
            new_topic = self._row_to_topic(checked_topic)
            return TopicCheckResult(
                topic=new_topic, new_papers=[], baseline_established=True
            )

        new_papers = [p for p in papers if p.arxiv_id not in existing_ids]
        for p in new_papers:
            self.db.execute(
                "INSERT OR IGNORE INTO watched_topic_seen (topic_id, arxiv_id, seen_at) VALUES (?, ?, ?)",
                (topic.id, p.arxiv_id, _now_iso()),
            )

        self.db.execute(
            "UPDATE watched_topics SET last_checked = ?, check_count = check_count + 1 WHERE id = ?",
            (_now_iso(), topic.id),
        )

        checked_topic = self.db.fetchone(
            "SELECT * FROM watched_topics WHERE id = ?", (topic.id,)
        )
        new_topic = self._row_to_topic(checked_topic)
        new_topic.new_papers_this_check = len(new_papers)

        return TopicCheckResult(
            topic=new_topic, new_papers=new_papers, baseline_established=False
        )

    async def check(self, topic_id: Optional[int] = None) -> WatcherResult:
        if topic_id is not None:
            row = self.db.fetchone(
                "SELECT * FROM watched_topics WHERE id = ?", (topic_id,)
            )
            if not row:
                return WatcherResult(
                    action="check", check_results=[], message="Topic not found"
                )
            topic = self._row_to_topic(row)
            check_result = await self._check_one(topic)
            return WatcherResult(
                action="check",
                check_results=[check_result],
                message="Checked one topic",
            )

        rows = self.db.fetchall(
            "SELECT * FROM watched_topics ORDER BY created_at ASC", ()
        )
        check_results: List[TopicCheckResult] = []
        for row in rows:
            topic = self._row_to_topic(row)
            check_results.append(await self._check_one(topic))
        return WatcherResult(
            action="check_all",
            check_results=check_results,
            message="Checked all topics",
        )

    async def dispatch(self, action: str, **kwargs: Any) -> WatcherResult:
        if action == "add":
            return await self.add(kwargs.get("query", ""), kwargs.get("label", ""))
        if action == "remove":
            topic_id = kwargs.get("topic_id")
            if topic_id is None:
                raise ValueError("topic_id is required")
            return await self.remove(int(topic_id))
        if action == "list":
            return await self.list()
        if action == "check":
            topic_id = kwargs.get("topic_id")
            if topic_id is None:
                return await self.check(None)
            return await self.check(int(topic_id))
        if action == "check_all":
            return await self.check(None)
        raise ValueError(
            "Unknown action; valid actions are add, remove, list, check, check_all"
        )
