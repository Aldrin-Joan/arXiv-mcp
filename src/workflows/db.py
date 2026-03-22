"""Persistence layer for Layer 4 research workflows (SQLite)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


class DatabaseClient:
    """Lightweight SQLite client for workflow storage.

    The client is safe for concurrency across worker threads via
    `check_same_thread=False` and uses context-managed transactions.
    """

    def __init__(self, db_path: str) -> None:
        """Initialize client and create schema.

        Args:
            db_path: path to sqlite3 database file.

        Raises:
            sqlite3.DatabaseError: on invalid DB path or initialization error.
        """
        if db_path == ":memory:" or db_path.startswith("file::memory:"):
            self.conn = sqlite3.connect(db_path, timeout=10, check_same_thread=False)
        else:
            db_file = Path(db_path).expanduser().resolve()
            db_file.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(
                str(db_file), timeout=10, check_same_thread=False
            )

        self.conn.row_factory = sqlite3.Row
        self._migrate()

    def _migrate(self) -> None:
        """Create required tables, idempotent."""
        create_sql = """
        BEGIN;
        CREATE TABLE IF NOT EXISTS reading_list (
            arxiv_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            authors TEXT NOT NULL,
            year INTEGER,
            abstract TEXT,
            tags TEXT NOT NULL DEFAULT '[]',
            notes TEXT NOT NULL DEFAULT '',
            read_status TEXT NOT NULL DEFAULT 'unread',
            added_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS watched_topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            label TEXT NOT NULL,
            last_checked TEXT,
            check_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS watched_topic_seen (
            topic_id INTEGER NOT NULL REFERENCES watched_topics(id) ON DELETE CASCADE,
            arxiv_id TEXT NOT NULL,
            seen_at TEXT NOT NULL,
            PRIMARY KEY (topic_id, arxiv_id)
        );

        CREATE TABLE IF NOT EXISTS explanations (
            arxiv_id TEXT NOT NULL,
            audience TEXT NOT NULL,
            content TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            PRIMARY KEY (arxiv_id, audience)
        );
        COMMIT;
        """
        with self.conn:
            self.conn.executescript(create_sql)

    def execute(
        self, sql: str, params: Optional[tuple[Any, ...]] = None
    ) -> sqlite3.Cursor:
        """Execute a SQL statement in a transactional context.

        Args:
            sql: SQL query with placeholders.
            params: tuple of parameters for the query.

        Returns:
            sqlite3.Cursor for further fetch operations.
        """
        params = params if params is not None else ()
        with self.conn:
            cursor = self.conn.execute(sql, params)
        return cursor

    def fetchall(
        self, sql: str, params: Optional[tuple[Any, ...]] = None
    ) -> List[Dict[str, Any]]:
        """Fetch all rows from query and convert to dictionary list."""
        cursor = self.execute(sql, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def fetchone(
        self, sql: str, params: Optional[tuple[Any, ...]] = None
    ) -> Optional[Dict[str, Any]]:
        """Fetch one row from query and convert to dictionary."""
        cursor = self.execute(sql, params)
        row = cursor.fetchone()
        return dict(row) if row is not None else None

    def close(self) -> None:
        """Close the SQLite connection."""
        try:
            self.conn.close()
        except sqlite3.Error:
            pass
