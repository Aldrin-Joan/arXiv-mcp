"""Reading list CRUD manager for Layer 4 workflows."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.models import (  # noqa: F401
    PaperMetadata,
    ReadingListEntry,
    ReadingListResult,
    WatchedTopic,
)
from src.workflows.db import DatabaseClient


VALID_READ_STATUSES = {"unread", "reading", "read"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReadingListManager:
    """Manage reading list items in sqlite backed workflow state."""

    def __init__(self, db: DatabaseClient, arxiv_client: Any) -> None:
        """Create a manager instance.

        Args:
            db: database client instance.
            arxiv_client: object with async get_by_id(arxiv_id) -> PaperMetadata.
        """
        self.db = db
        self.arxiv_client = arxiv_client

    async def _fetch_metadata(self, arxiv_id: str) -> PaperMetadata:
        """Fetch metadata for arXiv ID via the arXiv client."""
        meta = await self.arxiv_client.get_by_id(arxiv_id)
        if meta is None:
            raise ValueError(f"Paper not found: {arxiv_id}")
        return meta

    @staticmethod
    def _deserialize_tags(value: Any) -> List[str]:
        if value is None or value == "":
            return []
        if isinstance(value, list):
            return [str(x) for x in value]
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
        return []

    @staticmethod
    def _serialize_tags(tags: List[str]) -> str:
        return json.dumps(
            list(dict.fromkeys(str(t).strip() for t in tags if str(t).strip()))
        )

    @staticmethod
    def _row_to_entry(row: Optional[Dict[str, Any]]) -> ReadingListEntry:
        if row is None:
            raise ValueError("row is required")

        return ReadingListEntry(
            arxiv_id=row["arxiv_id"],
            title=row["title"],
            authors=json.loads(row["authors"]),
            year=row.get("year"),
            abstract=row.get("abstract", "") or "",
            tags=ReadingListManager._deserialize_tags(row.get("tags")),
            notes=row.get("notes", "") or "",
            read_status=row.get("read_status", "unread"),
            added_at=datetime.fromisoformat(row["added_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    async def add(
        self,
        arxiv_id: str,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        read_status: str = "unread",
    ) -> ReadingListResult:
        if not arxiv_id or not arxiv_id.strip():
            raise ValueError("arxiv_id is required")

        if read_status not in VALID_READ_STATUSES:
            raise ValueError(f"Invalid read_status: {read_status}")

        tags = tags or []
        notes = notes or ""

        existing = self.db.fetchone(
            "SELECT * FROM reading_list WHERE arxiv_id = ?", (arxiv_id,)
        )

        now = _now_iso()

        if existing:
            current_tags = set(self._deserialize_tags(existing.get("tags")))
            merged_tags = current_tags.union(
                {t for t in tags if t and isinstance(t, str)}
            )
            merged_tags_list = sorted(merged_tags)

            existing_notes = existing.get("notes", "") or ""
            combined_notes = existing_notes
            if notes.strip():
                if combined_notes:
                    combined_notes += "\n---\n"
                combined_notes += notes.strip()

            self.db.execute(
                """
                UPDATE reading_list
                SET tags = ?, notes = ?, read_status = ?, updated_at = ?
                WHERE arxiv_id = ?
                """,
                (
                    self._serialize_tags(merged_tags_list),
                    combined_notes,
                    read_status,
                    now,
                    arxiv_id,
                ),
            )

            updated = self.db.fetchone(
                "SELECT * FROM reading_list WHERE arxiv_id = ?", (arxiv_id,)
            )
            entry = self._row_to_entry(updated)
            return ReadingListResult(
                action="add",
                entry=entry,
                message="Entry updated (already existed)",
            )

        metadata = await self._fetch_metadata(arxiv_id)

        self.db.execute(
            """
            INSERT INTO reading_list
                (arxiv_id, title, authors, year, abstract, tags, notes, read_status, added_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metadata.arxiv_id,
                metadata.title,
                json.dumps([a.name for a in metadata.authors]),
                int(metadata.published[:4]) if metadata.published else None,
                metadata.abstract,
                self._serialize_tags(tags),
                notes.strip(),
                read_status,
                now,
                now,
            ),
        )

        new_entry = self.db.fetchone(
            "SELECT * FROM reading_list WHERE arxiv_id = ?", (arxiv_id,)
        )

        return ReadingListResult(
            action="add",
            entry=self._row_to_entry(new_entry),
            message="Entry added",
        )

    async def remove(self, arxiv_id: str) -> ReadingListResult:
        if not arxiv_id or not arxiv_id.strip():
            raise ValueError("arxiv_id is required")

        existing = self.db.fetchone(
            "SELECT * FROM reading_list WHERE arxiv_id = ?", (arxiv_id,)
        )
        if not existing:
            return ReadingListResult(
                action="remove", message="Paper not in reading list"
            )

        self.db.execute("DELETE FROM reading_list WHERE arxiv_id = ?", (arxiv_id,))
        return ReadingListResult(action="remove", message="Removed")

    async def update(
        self,
        arxiv_id: str,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        read_status: Optional[str] = None,
    ) -> ReadingListResult:
        if not arxiv_id or not arxiv_id.strip():
            raise ValueError("arxiv_id is required")

        existing = self.db.fetchone(
            "SELECT * FROM reading_list WHERE arxiv_id = ?", (arxiv_id,)
        )
        if not existing:
            return ReadingListResult(
                action="update", message="Paper not found in reading list"
            )

        set_clauses = []
        parameters: List[Any] = []

        if tags is not None:
            merged = set(self._deserialize_tags(existing.get("tags")))
            merged.update({t for t in tags if t and isinstance(t, str)})
            set_clauses.append("tags = ?")
            parameters.append(self._serialize_tags(sorted(merged)))

        if notes is not None:
            set_clauses.append("notes = ?")
            parameters.append(notes.strip())

        if read_status is not None:
            if read_status not in VALID_READ_STATUSES:
                raise ValueError(f"Invalid read_status: {read_status}")
            set_clauses.append("read_status = ?")
            parameters.append(read_status)

        if not set_clauses:
            return ReadingListResult(action="update", message="No updates provided")

        set_clauses.append("updated_at = ?")
        parameters.append(_now_iso())
        parameters.append(arxiv_id)

        self.db.execute(
            f"UPDATE reading_list SET {', '.join(set_clauses)} WHERE arxiv_id = ?",
            tuple(parameters),
        )

        updated = self.db.fetchone(
            "SELECT * FROM reading_list WHERE arxiv_id = ?", (arxiv_id,)
        )
        return ReadingListResult(
            action="update", entry=self._row_to_entry(updated), message="Updated"
        )

    async def get(self, arxiv_id: str) -> ReadingListResult:
        if not arxiv_id or not arxiv_id.strip():
            raise ValueError("arxiv_id is required")

        row = self.db.fetchone(
            "SELECT * FROM reading_list WHERE arxiv_id = ?", (arxiv_id,)
        )
        if not row:
            return ReadingListResult(action="get", entry=None, message="Not found")

        return ReadingListResult(
            action="get", entry=self._row_to_entry(row), message="Found"
        )

    async def list(
        self,
        filter_tags: Optional[List[str]] = None,
        filter_status: Optional[str] = None,
        filter_query: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        sort_by: str = "added_at",
        sort_order: str = "desc",
        limit: int = 20,
        offset: int = 0,
    ) -> ReadingListResult:
        if filter_status and filter_status not in VALID_READ_STATUSES:
            raise ValueError(f"Invalid read_status: {filter_status}")

        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")

        if offset < 0:
            raise ValueError("offset must be >= 0")

        valid_sort_by = {"added_at", "year", "title"}
        if sort_by not in valid_sort_by:
            raise ValueError(f"Invalid sort_by: {sort_by}")

        sort_order = sort_order.lower()
        if sort_order not in {"asc", "desc"}:
            raise ValueError(f"Invalid sort_order: {sort_order}")

        where_clauses = []
        parameters: List[Any] = []

        if filter_status:
            where_clauses.append("read_status = ?")
            parameters.append(filter_status)

        if year_min is not None:
            where_clauses.append("year >= ?")
            parameters.append(year_min)

        if year_max is not None:
            where_clauses.append("year <= ?")
            parameters.append(year_max)

        if filter_query:
            q = f"%{filter_query.strip().lower()}%"
            where_clauses.append("(LOWER(title) LIKE ? OR LOWER(abstract) LIKE ?)")
            parameters.extend([q, q])

        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

        count_row = self.db.fetchone(
            f"SELECT COUNT(*) as total FROM reading_list WHERE {where_clause}",
            tuple(parameters),
        )
        total_count = int(count_row["total"]) if count_row else 0

        rows = self.db.fetchall(
            f"SELECT * FROM reading_list WHERE {where_clause} ORDER BY {sort_by} {sort_order} LIMIT ? OFFSET ?",
            tuple(parameters) + (limit, offset),
        )

        entries = [self._row_to_entry(row) for row in rows]

        if filter_tags:
            filter_tags_lower = {t.strip().lower() for t in filter_tags if t.strip()}
            entries = [
                entry
                for entry in entries
                if filter_tags_lower.intersection({x.lower() for x in entry.tags})
            ]

        return ReadingListResult(
            action="list",
            entries=entries,
            total_count=total_count,
            message="Listed",
        )

    async def stats(self) -> ReadingListResult:
        base = self.db.fetchone("SELECT COUNT(*) as total FROM reading_list", ())
        total = int(base["total"]) if base else 0

        counts = {}
        for rs in VALID_READ_STATUSES:
            row = self.db.fetchone(
                "SELECT COUNT(*) as cnt FROM reading_list WHERE read_status = ?", (rs,)
            )
            counts[rs] = int(row["cnt"]) if row else 0

        rows = self.db.fetchall("SELECT tags FROM reading_list", ())
        tag_counter = Counter()
        for row in rows:
            for tag in self._deserialize_tags(row.get("tags")):
                tag_clean = tag.strip().lower()
                if tag_clean:
                    tag_counter[tag_clean] += 1

        top_tags = sorted(tag_counter.items(), key=lambda v: (-v[1], v[0]))[:10]

        return ReadingListResult(
            action="stats",
            stats={
                "unread": counts.get("unread", 0),
                "reading": counts.get("reading", 0),
                "read": counts.get("read", 0),
                "total": total,
                "top_tags": top_tags,
            },
            message="Stats",
        )

    async def dispatch(self, action: str, **kwargs: Any) -> ReadingListResult:
        if action not in {"add", "remove", "update", "get", "list", "stats"}:
            raise ValueError(
                "Unknown action; valid actions are add, remove, update, get, list, stats"
            )

        return await getattr(self, action)(**kwargs)
