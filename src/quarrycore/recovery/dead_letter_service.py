"""
Dead Letter Queue service for handling failed document processing.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol
from uuid import UUID

import aiosqlite

from quarrycore.config.config import SQLiteConfig
from quarrycore.protocols import ProcessingResult


class DeadLetterService(Protocol):
    """Protocol for dead letter queue operations."""

    async def failure_stats(self) -> Dict[str, Any]:
        """Get failure statistics."""
        ...

    async def add_failure(self, document_id: UUID, stage: str, error: str, retryable: bool = True) -> None:
        """Add a failed document to the dead letter queue."""
        ...

    async def get_failed_documents(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get failed documents."""
        ...


class SQLiteDeadLetterService(DeadLetterService):
    """SQLite-based dead letter queue service."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the database and create tables."""
        if self._initialized:
            return

        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS failed_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    is_retryable BOOLEAN NOT NULL DEFAULT 1,
                    failure_count INTEGER NOT NULL DEFAULT 1,
                    first_failure_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_failure_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT DEFAULT '{}'
                )
            """
            )

            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_failed_documents_stage ON failed_documents(stage)
            """
            )

            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_failed_documents_retryable ON failed_documents(is_retryable)
            """
            )

            await conn.commit()

        self._initialized = True

    async def failure_stats(self) -> Dict[str, Any]:
        """Get comprehensive failure statistics."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row

            # Get total failure count
            cursor = await conn.execute("SELECT COUNT(*) as total FROM failed_documents")
            total_row = await cursor.fetchone()
            total_failures = total_row["total"] if total_row else 0

            # Get failures by stage
            cursor = await conn.execute(
                """
                SELECT stage, COUNT(*) as count
                FROM failed_documents
                GROUP BY stage
            """
            )
            stage_rows = await cursor.fetchall()
            failures_by_stage = {row["stage"]: row["count"] for row in stage_rows}

            # Get retryable vs permanent failures
            cursor = await conn.execute(
                """
                SELECT is_retryable, COUNT(*) as count
                FROM failed_documents
                GROUP BY is_retryable
            """
            )
            retry_rows = await cursor.fetchall()
            retryable_failures = 0
            permanent_failures = 0

            for row in retry_rows:
                if row["is_retryable"]:
                    retryable_failures = row["count"]
                else:
                    permanent_failures = row["count"]

            # Get recent failures (last 24 hours)
            cursor = await conn.execute(
                """
                SELECT COUNT(*) as count
                FROM failed_documents
                WHERE last_failure_at > datetime('now', '-24 hours')
            """
            )
            recent_row = await cursor.fetchone()
            recent_failures = recent_row["count"] if recent_row else 0

            return {
                "total_failures": total_failures,
                "failures_by_stage": failures_by_stage,
                "retryable_failures": retryable_failures,
                "permanent_failures": permanent_failures,
                "recent_failures_24h": recent_failures,
                "last_updated": "now",
            }

    async def add_failure(self, document_id: UUID, stage: str, error: str, retryable: bool = True) -> None:
        """Add a failed document to the dead letter queue."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as conn:
            # Check if document already exists
            cursor = await conn.execute(
                """
                SELECT id, failure_count FROM failed_documents
                WHERE document_id = ? AND stage = ?
            """,
                (str(document_id), stage),
            )

            existing = await cursor.fetchone()

            if existing:
                # Update existing record
                await conn.execute(
                    """
                    UPDATE failed_documents
                    SET failure_count = failure_count + 1,
                        last_failure_at = CURRENT_TIMESTAMP,
                        error_message = ?,
                        is_retryable = ?
                    WHERE id = ?
                """,
                    (error, retryable, existing[0]),
                )
            else:
                # Insert new record
                await conn.execute(
                    """
                    INSERT INTO failed_documents
                    (document_id, stage, error_message, is_retryable, failure_count)
                    VALUES (?, ?, ?, ?, 1)
                """,
                    (str(document_id), stage, error, retryable),
                )

            await conn.commit()

    async def get_failed_documents(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get failed documents with optional limit."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row

            cursor = await conn.execute(
                """
                SELECT document_id, stage, error_message, is_retryable,
                       failure_count, first_failure_at, last_failure_at, metadata
                FROM failed_documents
                ORDER BY last_failure_at DESC
                LIMIT ?
            """,
                (limit,),
            )

            rows = await cursor.fetchall()

            return [
                {
                    "document_id": row["document_id"],
                    "stage": row["stage"],
                    "error_message": row["error_message"],
                    "is_retryable": bool(row["is_retryable"]),
                    "failure_count": row["failure_count"],
                    "first_failure_at": row["first_failure_at"],
                    "last_failure_at": row["last_failure_at"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                }
                for row in rows
            ]

    async def retry_failed_documents(self, stage: Optional[str] = None) -> int:
        """Mark failed documents as ready for retry."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as conn:
            if stage:
                cursor = await conn.execute(
                    """
                    UPDATE failed_documents
                    SET failure_count = 0, last_failure_at = CURRENT_TIMESTAMP
                    WHERE stage = ? AND is_retryable = 1
                """,
                    (stage,),
                )
            else:
                cursor = await conn.execute(
                    """
                    UPDATE failed_documents
                    SET failure_count = 0, last_failure_at = CURRENT_TIMESTAMP
                    WHERE is_retryable = 1
                """
                )

            await conn.commit()
            return cursor.rowcount

    async def clear_permanent_failures(self) -> int:
        """Clear permanent failures from the queue."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute(
                """
                DELETE FROM failed_documents WHERE is_retryable = 0
            """
            )
            await conn.commit()
            return cursor.rowcount


__all__ = ["DeadLetterService", "SQLiteDeadLetterService"]
