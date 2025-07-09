"""
Comprehensive unit tests for DeadLetterQueue to achieve ≥90% coverage.

Tests focus on the actual DeadLetterQueue implementation:
- Real constructor parameters and methods
- Database operations and upsert behavior
- Error handling and edge cases
- Statistics and monitoring
- Retry mechanisms and cleanup
"""

import asyncio
import json
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import aiosqlite
import pytest
from quarrycore.protocols import ErrorInfo, ErrorSeverity
from quarrycore.recovery.dead_letter import DeadLetterQueue, FailedDocument


class TestDeadLetterQueueInitialization:
    """Test DeadLetterQueue initialization and setup."""

    @pytest.mark.asyncio
    async def test_initialization_with_defaults(self):
        """Test initialization with default settings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"

            dlq = DeadLetterQueue(db_path=db_path)
            await dlq.initialize()

            assert dlq._db is not None
            assert dlq.db_path == db_path
            assert dlq.max_queue_size == 10000
            assert dlq.cleanup_after_days == 30

            await dlq.close()

    @pytest.mark.asyncio
    async def test_initialization_with_custom_settings(self):
        """Test initialization with custom settings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "custom.db"

            dlq = DeadLetterQueue(db_path=db_path, max_queue_size=5000, cleanup_after_days=14)

            assert dlq.max_queue_size == 5000
            assert dlq.cleanup_after_days == 14

            await dlq.initialize()
            await dlq.close()

    @pytest.mark.asyncio
    async def test_database_schema_creation(self):
        """Test that database schema is created correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "schema_test.db"

            dlq = DeadLetterQueue(db_path=db_path)
            await dlq.initialize()

            # Verify tables exist
            assert dlq._db is not None
            cursor = await dlq._db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='failed_documents'"
            )
            result = await cursor.fetchone()
            assert result is not None

            # Check indexes exist
            cursor = await dlq._db.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_next_retry'"
            )
            result = await cursor.fetchone()
            assert result is not None

            cursor = await dlq._db.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_failure_stage'"
            )
            result = await cursor.fetchone()
            assert result is not None

            await dlq.close()

    @pytest.mark.asyncio
    async def test_database_file_creation(self):
        """Test that database file is created with proper path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test nested directory creation
            nested_path = Path(temp_dir) / "nested" / "dir" / "test.db"

            dlq = DeadLetterQueue(db_path=nested_path)
            await dlq.initialize()

            assert nested_path.exists()
            assert nested_path.is_file()

            await dlq.close()


class TestDeadLetterQueueOperations:
    """Test core DeadLetterQueue operations."""

    @pytest.mark.asyncio
    async def test_add_failed_document_basic(self):
        """Test adding a failed document with basic information."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "basic.db"

            dlq = DeadLetterQueue(db_path=db_path)
            await dlq.initialize()

            error_info = ErrorInfo(
                error_type="NetworkError", error_message="Connection timeout", severity=ErrorSeverity.MEDIUM
            )

            document_id = await dlq.add_failed_document(
                url="https://example.com/test", failure_stage="crawl", error_info=error_info, metadata={"test": "data"}
            )

            assert document_id is not None

            # Verify record exists in database
            assert dlq._db is not None
            cursor = await dlq._db.execute("SELECT * FROM failed_documents WHERE document_id = ?", (str(document_id),))
            record = await cursor.fetchone()
            assert record is not None
            assert record[1] == "https://example.com/test"  # url
            assert record[2] == "crawl"  # failure_stage

            await dlq.close()

    @pytest.mark.asyncio
    async def test_add_failed_document_upsert_behavior(self):
        """Test AC-04: Upsert behavior with UNIQUE constraint."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "upsert.db"

            dlq = DeadLetterQueue(db_path=db_path)
            await dlq.initialize()

            error_info = ErrorInfo(
                error_type="NetworkError", error_message="Connection timeout", severity=ErrorSeverity.MEDIUM
            )

            url = "https://example.com/duplicate"
            stage = "crawl"

            # Add first failure
            doc_id_1 = await dlq.add_failed_document(url=url, failure_stage=stage, error_info=error_info)

            # Add same URL/stage again - should increment failure_count
            doc_id_2 = await dlq.add_failed_document(url=url, failure_stage=stage, error_info=error_info)

            # Should get different document IDs but same record updated
            assert doc_id_1 != doc_id_2  # New UUIDs but same record

            # Verify failure count incremented
            assert dlq._db is not None
            cursor = await dlq._db.execute(
                "SELECT failure_count FROM failed_documents WHERE url = ? AND failure_stage = ?", (url, stage)
            )
            record = await cursor.fetchone()
            assert record is not None
            assert record[0] == 2  # failure_count should be 2

            await dlq.close()

    @pytest.mark.asyncio
    async def test_get_failed_documents_basic(self):
        """Test retrieving failed documents."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "retrieve.db"

            dlq = DeadLetterQueue(db_path=db_path)
            await dlq.initialize()

            # Add some failed documents
            error_info = ErrorInfo(error_type="NetworkError", error_message="Timeout", severity=ErrorSeverity.MEDIUM)

            for i in range(3):
                await dlq.add_failed_document(
                    url=f"https://example.com/page{i}", failure_stage="crawl", error_info=error_info
                )

            # Retrieve all documents
            documents = await dlq.get_failed_documents()
            assert len(documents) == 3

            # Check document structure
            doc = documents[0]
            assert isinstance(doc, FailedDocument)
            assert doc.url.startswith("https://example.com/page")
            assert doc.failure_stage == "crawl"

            await dlq.close()

    @pytest.mark.asyncio
    async def test_get_failed_documents_with_limit(self):
        """Test retrieving failed documents with limit."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "limit.db"

            dlq = DeadLetterQueue(db_path=db_path)
            await dlq.initialize()

            # Add documents
            error_info = ErrorInfo(error_type="NetworkError", error_message="Timeout", severity=ErrorSeverity.MEDIUM)

            for i in range(10):
                await dlq.add_failed_document(
                    url=f"https://example.com/page{i}", failure_stage="crawl", error_info=error_info
                )

            # Retrieve with limit
            documents = await dlq.get_failed_documents(limit=3)
            assert len(documents) == 3

            await dlq.close()

    @pytest.mark.asyncio
    async def test_get_failed_documents_by_stage(self):
        """Test retrieving failed documents filtered by stage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "stage_filter.db"

            dlq = DeadLetterQueue(db_path=db_path)
            await dlq.initialize()

            error_info = ErrorInfo(error_type="ProcessingError", error_message="Failed", severity=ErrorSeverity.MEDIUM)

            # Add documents to different stages
            stages = ["crawl", "extract", "quality"]
            for stage in stages:
                for i in range(2):
                    await dlq.add_failed_document(
                        url=f"https://example.com/{stage}{i}", failure_stage=stage, error_info=error_info
                    )

            # Retrieve only crawl stage
            documents = await dlq.get_failed_documents(stage="crawl")
            assert len(documents) == 2
            assert all(doc.failure_stage == "crawl" for doc in documents)

            await dlq.close()

    @pytest.mark.asyncio
    async def test_get_failure_statistics_comprehensive(self):
        """Test getting comprehensive statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "stats.db"

            dlq = DeadLetterQueue(db_path=db_path)
            await dlq.initialize()

            # Add documents with different error types and stages
            error_types = [
                ("NetworkError", ErrorSeverity.MEDIUM),
                ("ParseError", ErrorSeverity.LOW),
                ("DatabaseError", ErrorSeverity.HIGH),
            ]
            stages = ["crawl", "extract", "quality"]

            for stage in stages:
                for error_type, severity in error_types:
                    error_info = ErrorInfo(
                        error_type=error_type, error_message=f"{error_type} in {stage}", severity=severity
                    )
                    await dlq.add_failed_document(
                        url=f"https://example.com/{stage}_{error_type}", failure_stage=stage, error_info=error_info
                    )

            stats = await dlq.get_failure_statistics()

            assert "total_failures" in stats
            assert "failures_by_stage" in stats
            assert "failures_by_error" in stats
            assert "retryable_failures" in stats
            assert "max_retries_reached" in stats
            assert "average_attempts" in stats

            assert stats["total_failures"] == 9  # 3 stages * 3 error types
            assert stats["failures_by_stage"]["crawl"] == 3
            assert stats["failures_by_stage"]["extract"] == 3
            assert stats["failures_by_stage"]["quality"] == 3
            assert stats["failures_by_error"]["NetworkError"] == 3
            assert stats["failures_by_error"]["ParseError"] == 3
            assert stats["failures_by_error"]["DatabaseError"] == 3

            await dlq.close()

    @pytest.mark.asyncio
    async def test_cleanup_old_failures(self):
        """Test cleaning up old records."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "cleanup.db"

            dlq = DeadLetterQueue(db_path=db_path, cleanup_after_days=1)
            await dlq.initialize()

            # Add old document by directly inserting with past timestamp
            old_dt = datetime.utcnow() - timedelta(days=2)

            assert dlq._db is not None
            await dlq._db.execute(
                """
                INSERT INTO failed_documents (
                    document_id, url, failure_stage, error_type, error_message,
                    error_severity, attempt_count, max_retries, first_failure_time,
                    last_failure_time, is_retryable, failure_count, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    str(uuid4()),
                    "https://example.com/old",
                    "crawl",
                    "NetworkError",
                    "Timeout",
                    "MEDIUM",
                    1,
                    3,
                    old_dt.isoformat(),
                    old_dt.isoformat(),
                    1,
                    1,
                    "{}",
                ),
            )
            await dlq._db.commit()

            # Add recent document
            error_info = ErrorInfo(
                error_type="NetworkError", error_message="Recent timeout", severity=ErrorSeverity.MEDIUM
            )
            await dlq.add_failed_document(
                url="https://example.com/recent", failure_stage="crawl", error_info=error_info
            )

            # Before cleanup
            cursor = await dlq._db.execute("SELECT COUNT(*) FROM failed_documents")
            count_before_result = await cursor.fetchone()
            assert count_before_result is not None
            count_before = count_before_result[0]
            assert count_before == 2

            # Run cleanup
            cleaned = await dlq.cleanup_old_failures()
            assert cleaned == 1

            # After cleanup
            cursor = await dlq._db.execute("SELECT COUNT(*) FROM failed_documents")
            count_after_result = await cursor.fetchone()
            assert count_after_result is not None
            count_after = count_after_result[0]
            assert count_after == 1

            # Verify only recent document remains
            cursor = await dlq._db.execute("SELECT url FROM failed_documents")
            remaining_url_result = await cursor.fetchone()
            assert remaining_url_result is not None
            remaining_url = remaining_url_result[0]
            assert remaining_url == "https://example.com/recent"

            await dlq.close()

    @pytest.mark.asyncio
    async def test_retry_document_manual(self):
        """Test manually retrying a specific document."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "retry.db"

            dlq = DeadLetterQueue(db_path=db_path)
            await dlq.initialize()

            error_info = ErrorInfo(error_type="NetworkError", error_message="Timeout", severity=ErrorSeverity.MEDIUM)

            # Add failed document
            doc_id = await dlq.add_failed_document(
                url="https://example.com/retry", failure_stage="crawl", error_info=error_info
            )

            # Retry the document
            success = await dlq.retry_document(doc_id)
            assert success

            # Try to retry non-existent document
            fake_id = uuid4()
            success = await dlq.retry_document(fake_id)
            assert not success

            await dlq.close()

    @pytest.mark.asyncio
    async def test_operations_without_initialization(self):
        """Test operations called without proper initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "uninitialized.db"

            dlq = DeadLetterQueue(db_path=db_path)
            # Don't call initialize()

            # Operations should handle None database gracefully
            documents = await dlq.get_failed_documents()
            assert documents == []

            stats = await dlq.get_failure_statistics()
            assert stats == {}

            cleaned = await dlq.cleanup_old_failures()
            assert cleaned == 0


class TestDeadLetterQueueErrorHandling:
    """Test DeadLetterQueue error handling scenarios."""

    @pytest.mark.asyncio
    async def test_database_connection_error(self):
        """Test handling database connection errors."""
        # Test with invalid database path that can't be created
        invalid_path = Path("/root/cannot_create_here.db")

        dlq = DeadLetterQueue(db_path=invalid_path)

        # Should handle error gracefully during initialization
        with pytest.raises((OSError, PermissionError, sqlite3.OperationalError)):  # Filesystem and database errors
            await dlq.initialize()

    @pytest.mark.asyncio
    async def test_database_corruption_handling(self):
        """Test handling database corruption scenarios."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "corrupt.db"

            # Create corrupted database file
            with open(db_path, "wb") as f:
                f.write(b"This is not a valid SQLite database")

            dlq = DeadLetterQueue(db_path=db_path)

            # Should handle corruption gracefully
            with pytest.raises(aiosqlite.DatabaseError):  # Database corruption error
                await dlq.initialize()

    @pytest.mark.asyncio
    async def test_close_without_initialization(self):
        """Test closing without initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "close_test.db"

            dlq = DeadLetterQueue(db_path=db_path)

            # Should not fail when closing without initialization
            await dlq.close()

    @pytest.mark.asyncio
    async def test_close_cancels_retry_task(self):
        """Test that close properly cancels the retry task."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "retry_task.db"

            dlq = DeadLetterQueue(db_path=db_path)
            await dlq.initialize()

            # Verify retry task was started
            assert dlq._retry_task is not None
            assert not dlq._retry_task.done()

            # Close should cancel the task
            await dlq.close()

            # Verify task was cancelled
            assert dlq._retry_task.cancelled()


class TestDeadLetterQueueEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_very_long_urls(self):
        """Test handling very long URLs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "long_urls.db"

            dlq = DeadLetterQueue(db_path=db_path)
            await dlq.initialize()

            # Create very long URL
            long_url = "https://example.com/" + "x" * 2000

            error_info = ErrorInfo(error_type="NetworkError", error_message="Timeout", severity=ErrorSeverity.MEDIUM)

            document_id = await dlq.add_failed_document(url=long_url, failure_stage="crawl", error_info=error_info)

            assert document_id is not None

            # Verify record was stored
            assert dlq._db is not None
            cursor = await dlq._db.execute(
                "SELECT url FROM failed_documents WHERE document_id = ?", (str(document_id),)
            )
            result = await cursor.fetchone()
            assert result is not None
            stored_url = result[0]
            assert stored_url == long_url

            await dlq.close()

    @pytest.mark.asyncio
    async def test_unicode_and_special_characters(self):
        """Test handling Unicode and special characters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "unicode.db"

            dlq = DeadLetterQueue(db_path=db_path)
            await dlq.initialize()

            # URL with Unicode characters
            unicode_url = "https://example.com/测试/🚀/ñoño"
            unicode_message = "Erreur de réseau: 网络超时 🔥"

            error_info = ErrorInfo(
                error_type="NetworkError", error_message=unicode_message, severity=ErrorSeverity.MEDIUM
            )

            document_id = await dlq.add_failed_document(url=unicode_url, failure_stage="crawl", error_info=error_info)

            # Verify Unicode data was stored correctly
            assert dlq._db is not None
            cursor = await dlq._db.execute(
                "SELECT url, error_message FROM failed_documents WHERE document_id = ?", (str(document_id),)
            )
            result = await cursor.fetchone()
            assert result is not None
            assert result[0] == unicode_url
            assert result[1] == unicode_message

            await dlq.close()

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """Test concurrent access to dead letter queue."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "concurrent.db"

            dlq = DeadLetterQueue(db_path=db_path)
            await dlq.initialize()

            error_info = ErrorInfo(error_type="NetworkError", error_message="Timeout", severity=ErrorSeverity.MEDIUM)

            # Simulate concurrent writes
            async def add_document(i):
                return await dlq.add_failed_document(
                    url=f"https://example.com/concurrent{i}", failure_stage="crawl", error_info=error_info
                )

            # Run multiple concurrent operations
            tasks = [add_document(i) for i in range(10)]
            results = await asyncio.gather(*tasks)

            # All should succeed
            assert len(results) == 10
            assert all(result is not None for result in results)

            # Verify all records in database
            assert dlq._db is not None
            cursor = await dlq._db.execute("SELECT COUNT(*) FROM failed_documents")
            result = await cursor.fetchone()
            assert result is not None
            count = result[0]
            assert count == 10

            await dlq.close()

    @pytest.mark.asyncio
    async def test_retry_queue_processing(self):
        """Test the retry queue processing mechanism."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "retry_queue.db"

            dlq = DeadLetterQueue(db_path=db_path)
            await dlq.initialize()

            # Add document with immediate retry time
            error_info = ErrorInfo(
                error_type="NetworkError", error_message="Timeout", severity=ErrorSeverity.MEDIUM, is_retryable=True
            )

            # Create a FailedDocument with past retry time
            doc = FailedDocument(
                document_id=uuid4(), url="https://example.com/retry_ready", failure_stage="crawl", error_info=error_info
            )
            doc.next_retry_time = datetime.utcnow() - timedelta(minutes=1)  # Past time

            # Store directly to control retry time
            await dlq._store_failed_document(doc)

            # Check retry queue manually
            await dlq._check_retry_queue()

            # The document should have been added to processing queue
            # We can't easily test this without accessing private queue, but method should not fail

            await dlq.close()


class TestFailedDocumentModel:
    """Test FailedDocument model functionality."""

    def test_failed_document_creation(self):
        """Test creating FailedDocument instances."""
        error_info = ErrorInfo(
            error_type="NetworkError", error_message="Connection timeout", severity=ErrorSeverity.MEDIUM
        )

        doc = FailedDocument(
            document_id=uuid4(), url="https://example.com/test", failure_stage="crawl", error_info=error_info
        )

        assert doc.url == "https://example.com/test"
        assert doc.failure_stage == "crawl"
        assert doc.attempt_count == 0
        assert doc.max_retries == 3

    def test_should_retry_logic(self):
        """Test should_retry logic with different scenarios."""
        error_info = ErrorInfo(
            error_type="NetworkError", error_message="Timeout", severity=ErrorSeverity.MEDIUM, is_retryable=True
        )

        doc = FailedDocument(
            document_id=uuid4(), url="https://example.com/test", failure_stage="crawl", error_info=error_info
        )

        # Should retry when under max attempts
        assert doc.should_retry()

        # Should not retry when max attempts exceeded
        doc.attempt_count = 5
        assert not doc.should_retry()

        # Should not retry critical errors
        doc.attempt_count = 1
        doc.error_info.severity = ErrorSeverity.CRITICAL
        assert not doc.should_retry()

        # Should not retry non-retryable errors
        doc.error_info.severity = ErrorSeverity.MEDIUM
        doc.error_info.is_retryable = False
        assert not doc.should_retry()

    def test_calculate_next_retry_exponential_backoff(self):
        """Test exponential backoff calculation."""
        error_info = ErrorInfo(error_type="NetworkError", error_message="Timeout", severity=ErrorSeverity.MEDIUM)

        doc = FailedDocument(
            document_id=uuid4(), url="https://example.com/test", failure_stage="crawl", error_info=error_info
        )

        # Test different attempt counts
        base_time = datetime.utcnow()

        # First attempt: 60 seconds
        doc.attempt_count = 0
        next_retry = doc.calculate_next_retry()
        delay = (next_retry - base_time).total_seconds()
        assert 55 <= delay <= 65  # Allow some time variance

        # Second attempt: 120 seconds
        doc.attempt_count = 1
        next_retry = doc.calculate_next_retry()
        delay = (next_retry - base_time).total_seconds()
        assert 115 <= delay <= 125

        # Test max delay cap
        doc.attempt_count = 20  # Very high attempt count
        next_retry = doc.calculate_next_retry()
        delay = (next_retry - base_time).total_seconds()
        assert delay <= 86400 + 5  # Should not exceed 24 hours + small buffer

    def test_serialization_to_dict(self):
        """Test converting FailedDocument to dictionary."""
        error_info = ErrorInfo(error_type="NetworkError", error_message="Timeout", severity=ErrorSeverity.MEDIUM)

        doc_id = uuid4()
        doc = FailedDocument(
            document_id=doc_id,
            url="https://example.com/test",
            failure_stage="crawl",
            error_info=error_info,
            metadata={"test": "value"},
        )

        doc_dict = doc.to_dict()

        assert doc_dict["document_id"] == str(doc_id)
        assert doc_dict["url"] == "https://example.com/test"
        assert doc_dict["failure_stage"] == "crawl"
        assert doc_dict["metadata"]["test"] == "value"
        assert isinstance(doc_dict["error_info"], dict)

    def test_deserialization_from_dict(self):
        """Test creating FailedDocument from dictionary."""
        doc_id = uuid4()
        doc_dict = {
            "document_id": str(doc_id),
            "url": "https://example.com/test",
            "failure_stage": "crawl",
            "error_info": {
                "error_type": "NetworkError",
                "error_message": "Timeout",
                "severity": "MEDIUM",
                "is_retryable": True,
            },
            "attempt_count": 2,
            "max_retries": 3,
            "first_failure_time": "2023-01-01T00:00:00",
            "last_failure_time": "2023-01-01T00:05:00",
            "next_retry_time": "2023-01-01T00:10:00",
            "metadata": {"test": "value"},
        }

        doc = FailedDocument.from_dict(doc_dict)

        assert doc.document_id == doc_id
        assert doc.url == "https://example.com/test"
        assert doc.failure_stage == "crawl"
        assert doc.attempt_count == 2
        assert doc.metadata["test"] == "value"
        assert doc.error_info.error_type == "NetworkError"

    def test_next_retry_time_future_check(self):
        """Test should_retry with future next_retry_time."""
        error_info = ErrorInfo(
            error_type="NetworkError", error_message="Timeout", severity=ErrorSeverity.MEDIUM, is_retryable=True
        )

        doc = FailedDocument(
            document_id=uuid4(), url="https://example.com/test", failure_stage="crawl", error_info=error_info
        )

        # Set retry time to future
        doc.next_retry_time = datetime.utcnow() + timedelta(hours=1)

        # Should not retry yet
        assert not doc.should_retry()
