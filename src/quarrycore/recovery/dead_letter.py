"""
Dead letter queue for failed document processing.

Provides retry mechanisms and failure analysis for documents that fail processing.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import aiosqlite

from quarrycore.protocols import ErrorInfo, ErrorSeverity


@dataclass
class FailedDocument:
    """Represents a document that failed processing."""

    document_id: UUID
    url: str
    failure_stage: str
    error_info: ErrorInfo
    attempt_count: int = 0
    max_retries: int = 3
    first_failure_time: datetime = field(default_factory=datetime.utcnow)
    last_failure_time: datetime = field(default_factory=datetime.utcnow)
    next_retry_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_next_retry(self) -> datetime:
        """Calculate next retry time with exponential backoff."""
        base_delay = 60  # 1 minute
        max_delay = 3600 * 24  # 24 hours

        # Exponential backoff: delay = base * (2 ^ attempt)
        delay_seconds = min(base_delay * (2**self.attempt_count), max_delay)

        return datetime.utcnow() + timedelta(seconds=delay_seconds)

    def should_retry(self) -> bool:
        """Check if document should be retried."""
        if self.attempt_count >= self.max_retries:
            return False

        if self.next_retry_time and datetime.utcnow() < self.next_retry_time:
            return False

        # Don't retry critical errors
        if self.error_info.severity == ErrorSeverity.CRITICAL:
            return False

        return self.error_info.is_retryable

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "document_id": str(self.document_id),
            "url": self.url,
            "failure_stage": self.failure_stage,
            "error_info": asdict(self.error_info),
            "attempt_count": self.attempt_count,
            "max_retries": self.max_retries,
            "first_failure_time": self.first_failure_time.isoformat(),
            "last_failure_time": self.last_failure_time.isoformat(),
            "next_retry_time": (self.next_retry_time.isoformat() if self.next_retry_time else None),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FailedDocument:
        """Create from dictionary."""
        error_info = ErrorInfo(**data["error_info"])

        return cls(
            document_id=UUID(data["document_id"]),
            url=data["url"],
            failure_stage=data["failure_stage"],
            error_info=error_info,
            attempt_count=data["attempt_count"],
            max_retries=data["max_retries"],
            first_failure_time=datetime.fromisoformat(data["first_failure_time"]),
            last_failure_time=datetime.fromisoformat(data["last_failure_time"]),
            next_retry_time=(datetime.fromisoformat(data["next_retry_time"]) if data["next_retry_time"] else None),
            metadata=data.get("metadata", {}),
        )


class DeadLetterQueue:
    """
    Dead letter queue for managing failed documents.

    Features:
    - Async queue for failed documents
    - Exponential backoff retry mechanism
    - Maximum retry attempts with circuit breaker
    - Failed document analysis and reporting
    - Manual retry capability
    - Persistent storage for recovery
    """

    def __init__(
        self,
        db_path: Path = Path("./data/dead_letter.db"),
        max_queue_size: int = 10000,
        cleanup_after_days: int = 30,
    ):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.max_queue_size = max_queue_size
        self.cleanup_after_days = cleanup_after_days

        self._db: Optional[aiosqlite.Connection] = None
        self._processing_queue: asyncio.Queue[FailedDocument] = asyncio.Queue(maxsize=100)
        self._retry_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize the dead letter queue database."""
        self._db = await aiosqlite.connect(self.db_path)

        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS failed_documents (
                document_id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                failure_stage TEXT NOT NULL,
                error_type TEXT NOT NULL,
                error_message TEXT NOT NULL,
                error_severity TEXT NOT NULL,
                attempt_count INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 3,
                first_failure_time TEXT NOT NULL,
                last_failure_time TEXT NOT NULL,
                next_retry_time TEXT,
                is_retryable INTEGER DEFAULT 1,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_next_retry
            ON failed_documents(next_retry_time)
            WHERE is_retryable = 1
        """
        )

        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_failure_stage
            ON failed_documents(failure_stage)
        """
        )

        await self._db.commit()

        # Start retry task
        self._retry_task = asyncio.create_task(self._retry_loop())

    async def add_failed_document(
        self,
        url: str,
        failure_stage: str,
        error_info: ErrorInfo,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UUID:
        """Add a failed document to the queue."""
        document_id = uuid4()

        failed_doc = FailedDocument(
            document_id=document_id,
            url=url,
            failure_stage=failure_stage,
            error_info=error_info,
            metadata=metadata or {},
        )

        # Calculate next retry time
        failed_doc.next_retry_time = failed_doc.calculate_next_retry()

        # Store in database
        await self._store_failed_document(failed_doc)

        # Add to processing queue if should retry soon
        if failed_doc.should_retry() and failed_doc.next_retry_time:
            time_until_retry = (failed_doc.next_retry_time - datetime.utcnow()).total_seconds()
            if time_until_retry <= 300:  # Within 5 minutes
                await self._processing_queue.put(failed_doc)

        return document_id

    async def retry_document(self, document_id: UUID) -> bool:
        """Manually retry a specific document."""
        failed_doc = await self._get_failed_document(document_id)
        if not failed_doc:
            return False

        # Reset retry time to now
        failed_doc.next_retry_time = datetime.utcnow()
        await self._update_failed_document(failed_doc)

        # Add to processing queue
        await self._processing_queue.put(failed_doc)

        return True

    async def get_failed_documents(
        self,
        stage: Optional[str] = None,
        limit: int = 100,
    ) -> List[FailedDocument]:
        """Get failed documents, optionally filtered by stage."""
        if self._db is None:
            return []

        query = """
            SELECT * FROM failed_documents
            WHERE 1=1
        """
        params = []

        if stage:
            query += " AND failure_stage = ?"
            params.append(stage)

        query += " ORDER BY last_failure_time DESC LIMIT ?"
        params.append(limit)

        async with self._db.execute(query, params) as cursor:
            rows = await cursor.fetchall()

        documents = []
        for row in rows:
            doc_data = self._row_to_dict(row)
            documents.append(FailedDocument.from_dict(doc_data))

        return documents

    async def get_failure_statistics(self) -> Dict[str, Any]:
        """Get statistics about failed documents."""
        if self._db is None:
            return {}

        # Total failures by stage
        async with self._db.execute(
            """
            SELECT failure_stage, COUNT(*) as count
            FROM failed_documents
            GROUP BY failure_stage
        """
        ) as cursor:
            rows = await cursor.fetchall()
            stage_counts = {row[0]: row[1] for row in rows}

        # Total failures by error type
        async with self._db.execute(
            """
            SELECT error_type, COUNT(*) as count
            FROM failed_documents
            GROUP BY error_type
        """
        ) as cursor:
            rows = await cursor.fetchall()
            error_counts = {row[0]: row[1] for row in rows}

        # Retry statistics
        async with self._db.execute(
            """
            SELECT
                COUNT(*) as total_failures,
                SUM(CASE WHEN is_retryable = 1 THEN 1 ELSE 0 END) as retryable,
                SUM(CASE WHEN attempt_count >= max_retries THEN 1 ELSE 0 END) as max_retries_reached,
                AVG(attempt_count) as avg_attempts
            FROM failed_documents
        """
        ) as cursor:
            retry_stats = await cursor.fetchone()

        if retry_stats:
            return {
                "total_failures": retry_stats[0] or 0,
                "retryable_failures": retry_stats[1] or 0,
                "max_retries_reached": retry_stats[2] or 0,
                "average_attempts": retry_stats[3] or 0,
                "failures_by_stage": stage_counts,
                "failures_by_error": error_counts,
            }
        else:
            return {
                "total_failures": 0,
                "retryable_failures": 0,
                "max_retries_reached": 0,
                "average_attempts": 0,
                "failures_by_stage": stage_counts,
                "failures_by_error": error_counts,
            }

    async def cleanup_old_failures(self) -> int:
        """Remove old failures beyond retention period."""
        if self._db is None:
            return 0

        cutoff_date = datetime.utcnow() - timedelta(days=self.cleanup_after_days)

        async with self._db.execute(
            """
            DELETE FROM failed_documents
            WHERE last_failure_time < ?
        """,
            (cutoff_date.isoformat(),),
        ) as cursor:
            deleted = cursor.rowcount

        await self._db.commit()
        return deleted

    async def close(self) -> None:
        """Close the dead letter queue."""
        if self._retry_task:
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass

        if self._db:
            await self._db.close()

    async def _retry_loop(self) -> None:
        """Background task to retry failed documents."""
        while True:
            try:
                # Check for documents ready to retry
                await self._check_retry_queue()

                # Sleep for a bit
                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue
                print(f"Error in retry loop: {e}")
                await asyncio.sleep(60)

    async def _check_retry_queue(self) -> None:
        """Check for documents ready to retry."""
        if self._db is None:
            return

        now = datetime.utcnow()

        async with self._db.execute(
            """
            SELECT * FROM failed_documents
            WHERE is_retryable = 1
            AND next_retry_time <= ?
            AND attempt_count < max_retries
            LIMIT 10
        """,
            (now.isoformat(),),
        ) as cursor:
            rows = await cursor.fetchall()

        for row in rows:
            doc_data = self._row_to_dict(row)
            failed_doc = FailedDocument.from_dict(doc_data)

            # Add to processing queue
            try:
                self._processing_queue.put_nowait(failed_doc)
            except asyncio.QueueFull:
                # Queue is full, stop adding for now
                break

    async def _store_failed_document(self, doc: FailedDocument) -> None:
        """Store a failed document in the database."""
        if self._db is None:
            return

        await self._db.execute(
            """
            INSERT OR REPLACE INTO failed_documents (
                document_id, url, failure_stage, error_type, error_message,
                error_severity, attempt_count, max_retries, first_failure_time,
                last_failure_time, next_retry_time, is_retryable, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                str(doc.document_id),
                doc.url,
                doc.failure_stage,
                doc.error_info.error_type,
                doc.error_info.error_message,
                doc.error_info.severity.value,
                doc.attempt_count,
                doc.max_retries,
                doc.first_failure_time.isoformat(),
                doc.last_failure_time.isoformat(),
                doc.next_retry_time.isoformat() if doc.next_retry_time else None,
                1 if doc.error_info.is_retryable else 0,
                json.dumps(doc.metadata),
            ),
        )

        await self._db.commit()

    async def _update_failed_document(self, doc: FailedDocument) -> None:
        """Update a failed document in the database."""
        await self._store_failed_document(doc)

    async def _get_failed_document(self, document_id: UUID) -> Optional[FailedDocument]:
        """Get a specific failed document."""
        if self._db is None:
            return None

        async with self._db.execute(
            """
            SELECT * FROM failed_documents
            WHERE document_id = ?
        """,
            (str(document_id),),
        ) as cursor:
            row = await cursor.fetchone()

        if not row:
            return None

        doc_data = self._row_to_dict(row)
        return FailedDocument.from_dict(doc_data)

    def _row_to_dict(self, row: Any) -> Dict[str, Any]:
        """Convert database row to dictionary."""
        return {
            "document_id": row[0],
            "url": row[1],
            "failure_stage": row[2],
            "error_info": {
                "error_type": row[3],
                "error_message": row[4],
                "severity": row[5],
                "is_retryable": bool(row[11]),
            },
            "attempt_count": row[6],
            "max_retries": row[7],
            "first_failure_time": row[8],
            "last_failure_time": row[9],
            "next_retry_time": row[10],
            "metadata": json.loads(row[12]) if row[12] else {},
        }
