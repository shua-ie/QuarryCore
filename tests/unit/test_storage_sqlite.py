"""
Tests for SQLite storage manager to boost coverage.

These tests focus on meaningful functionality like store/query operations
to increase global line coverage toward the 24% target.
"""

import tempfile
from pathlib import Path

import pytest
from quarrycore.config.config import SQLiteConfig
from quarrycore.storage.sqlite_manager import SQLiteManager


@pytest.mark.unit
class TestSQLiteManager:
    """Test SQLite storage manager functionality."""

    @pytest.fixture
    async def db_manager(self):
        """Create a test SQLite manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SQLiteConfig(db_path=Path(temp_dir) / "test.db")
            manager = SQLiteManager(config)
            await manager.initialize()
            yield manager
            await manager.close()

    @pytest.mark.asyncio
    async def test_store_batch_and_search(self, db_manager):
        """Test storing batch data and searching."""
        # Create test batch data
        batch_data = [
            {
                "content_id": "test-1",
                "url": "https://example.com/test1",
                "content_hash": "hash1",
                "parquet_path": "path/to/file1.parquet",
                "title": "Test Document 1",
                "description": "This is a test document",
                "domain": "technical",
                "author": "Test Author",
                "published_date": None,
                "quality_score": 0.8,
                "is_duplicate": False,
                "toxicity_score": 0.1,
                "coherence_score": 0.9,
                "grammar_score": 0.85,
                "full_metadata": '{"test": "metadata"}',
            },
            {
                "content_id": "test-2",
                "url": "https://example.com/test2",
                "content_hash": "hash2",
                "parquet_path": "path/to/file2.parquet",
                "title": "Another Test",
                "description": "Another test document",
                "domain": "technical",
                "author": "Test Author 2",
                "published_date": None,
                "quality_score": 0.7,
                "is_duplicate": False,
                "toxicity_score": 0.2,
                "coherence_score": 0.8,
                "grammar_score": 0.75,
                "full_metadata": '{"test": "metadata2"}',
            },
        ]

        # Store batch
        await db_manager.store_batch(batch_data)

        # Search for content
        results = await db_manager.search("test", limit=10)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_get_recent_content(self, db_manager):
        """Test retrieving recent content."""
        # Store some test data first
        test_data = [
            {
                "content_id": "recent-1",
                "url": "https://example.com/recent",
                "content_hash": "recenthash",
                "parquet_path": "path/recent.parquet",
                "title": "Recent Document",
                "description": "A recent document",
                "domain": "general",
                "quality_score": 0.9,
                "is_duplicate": False,
                "full_metadata": "{}",
            }
        ]

        await db_manager.store_batch(test_data)

        # Get recent content
        recent = await db_manager.get_recent_content(limit=5)
        assert len(recent) >= 1
        assert recent[0]["content_id"] == "recent-1"

    @pytest.mark.asyncio
    async def test_get_content_metadata(self, db_manager):
        """Test retrieving content metadata by ID."""
        # Store test data
        test_data = [
            {
                "content_id": "meta-test",
                "url": "https://example.com/meta",
                "content_hash": "metahash",
                "parquet_path": "path/meta.parquet",
                "title": "Metadata Test",
                "description": "Testing metadata retrieval",
                "domain": "test",
                "quality_score": 0.6,
                "is_duplicate": False,
                "full_metadata": '{"key": "value", "number": 42}',
            }
        ]

        await db_manager.store_batch(test_data)

        # Get metadata by ID (would be 1 for first insert)
        metadata = await db_manager.get_content_metadata(1)
        assert metadata is not None
        assert metadata["key"] == "value"
        assert metadata["number"] == 42

    @pytest.mark.asyncio
    async def test_get_all_metadata(self, db_manager):
        """Test retrieving all metadata."""
        # Store test data
        test_data = [
            {
                "content_id": "all-1",
                "url": "https://example.com/all1",
                "content_hash": "allhash1",
                "parquet_path": "path/all1.parquet",
                "title": "All Test 1",
                "description": "First document",
                "domain": "test",
                "quality_score": 0.5,
                "is_duplicate": False,
                "full_metadata": '{"type": "first"}',
            },
            {
                "content_id": "all-2",
                "url": "https://example.com/all2",
                "content_hash": "allhash2",
                "parquet_path": "path/all2.parquet",
                "title": "All Test 2",
                "description": "Second document",
                "domain": "test",
                "quality_score": 0.4,
                "is_duplicate": False,
                "full_metadata": '{"type": "second"}',
            },
        ]

        await db_manager.store_batch(test_data)

        # Get all metadata
        all_metadata = await db_manager.get_all_metadata()
        assert len(all_metadata) >= 2

    @pytest.mark.asyncio
    async def test_empty_batch_handling(self, db_manager):
        """Test handling of empty batch data."""
        # Should handle empty batch gracefully
        await db_manager.store_batch([])

        # Should still be able to search
        results = await db_manager.search("nonexistent", limit=5)
        assert len(results) == 0
