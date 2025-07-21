"""
SQLite-based Exact Hash Deduplication Database.

Implements the exact hash layer for hybrid deduplication:
- WAL (Write-Ahead Logging) mode for high concurrency
- Unique index on hash column for instant duplicate detection
- Thread-safe operations with connection pooling
- Atomic insert-or-ignore operations
- Efficient batch operations
- Schema: hash_dedup(hash TEXT PRIMARY KEY, url TEXT, timestamp REAL)
"""

import asyncio
import hashlib
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Set, Tuple

import aiosqlite
import structlog

logger = structlog.get_logger(__name__)


class HashDatabase:
    """
    SQLite-based exact hash deduplication storage.

    Features:
    - WAL mode for high-concurrency reads/writes
    - Unique constraint on hash for instant duplicate detection
    - Async operations with aiosqlite
    - Connection pooling and proper cleanup
    - Batch operations for performance
    """

    def __init__(self, db_path: Path, wal_mode: bool = True):
        """
        Initialize hash database.

        Args:
            db_path: Path to SQLite database file
            wal_mode: Enable Write-Ahead Logging mode (recommended)
        """
        self.db_path = Path(db_path)
        self.wal_mode = wal_mode

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connection pool
        self._connection_pool: List[aiosqlite.Connection] = []
        self._pool_size = 5
        self._pool_lock = asyncio.Lock()

        # Statistics
        self._total_checks = 0
        self._duplicate_hits = 0
        self._unique_inserts = 0

        # Initialize database synchronously
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database schema synchronously."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Enable WAL mode for better concurrency
                if self.wal_mode:
                    conn.execute("PRAGMA journal_mode=WAL")
                    logger.info(f"Enabled WAL mode for {self.db_path}")

                # Set other performance pragmas
                conn.execute("PRAGMA synchronous=NORMAL")  # Faster than FULL, still safe with WAL
                conn.execute("PRAGMA cache_size=10000")  # 40MB cache
                conn.execute("PRAGMA temp_store=MEMORY")  # Store temp tables in memory

                # Create the hash deduplication table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS hash_dedup (
                        hash TEXT PRIMARY KEY,
                        url TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                # Create indexes for performance
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_hash_dedup_timestamp
                    ON hash_dedup(timestamp)
                """
                )

                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_hash_dedup_url
                    ON hash_dedup(url)
                """
                )

                conn.commit()
                logger.info(f"Initialized hash database at {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize hash database: {e}")
            raise

    @asynccontextmanager
    async def _get_connection(self):
        """Get a connection from the pool or create a new one."""
        async with self._pool_lock:
            if self._connection_pool:
                conn = self._connection_pool.pop()
            else:
                conn = await aiosqlite.connect(self.db_path)
                # Set pragmas for new connections
                if self.wal_mode:
                    await conn.execute("PRAGMA journal_mode=WAL")
                await conn.execute("PRAGMA synchronous=NORMAL")

        try:
            yield conn
        finally:
            async with self._pool_lock:
                if len(self._connection_pool) < self._pool_size:
                    self._connection_pool.append(conn)
                else:
                    await conn.close()

    async def is_duplicate(self, content: str, url: str = "unknown") -> Tuple[bool, str]:
        """
        Check if content is a duplicate and optionally store it.

        Args:
            content: Canonicalized content to check
            url: Source URL for tracking

        Returns:
            Tuple of (is_duplicate, content_hash)
        """
        # Calculate SHA-256 hash
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        self._total_checks += 1

        try:
            async with self._get_connection() as conn:
                # Try to insert the hash (will fail if duplicate due to unique constraint)
                try:
                    await conn.execute(
                        "INSERT INTO hash_dedup (hash, url, timestamp) VALUES (?, ?, ?)",
                        (content_hash, url, time.time()),
                    )
                    await conn.commit()

                    # Successfully inserted = not a duplicate
                    self._unique_inserts += 1
                    logger.debug(f"New unique content: {content_hash[:16]}... from {url}")
                    return False, content_hash

                except aiosqlite.IntegrityError:
                    # Unique constraint violation = duplicate found
                    self._duplicate_hits += 1
                    logger.debug(f"Duplicate detected: {content_hash[:16]}... from {url}")
                    return True, content_hash

        except Exception as e:
            logger.error(f"Hash database error: {e}")
            # On error, assume not duplicate to avoid blocking pipeline
            return False, content_hash

    async def batch_check(self, contents: List[Tuple[str, str]]) -> List[Tuple[bool, str]]:
        """
        Batch check multiple contents for duplicates.

        Args:
            contents: List of (content, url) tuples

        Returns:
            List of (is_duplicate, content_hash) tuples
        """
        results = []

        # Calculate all hashes first
        hash_data = []
        for content, url in contents:
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            hash_data.append((content_hash, url, time.time()))

        try:
            async with self._get_connection() as conn:
                # Check which hashes already exist
                hash_list = [h[0] for h in hash_data]
                placeholders = ",".join(["?" for _ in hash_list])

                cursor = await conn.execute(f"SELECT hash FROM hash_dedup WHERE hash IN ({placeholders})", hash_list)
                existing_hashes = {row[0] for row in await cursor.fetchall()}

                # Prepare batch insert for new hashes
                new_records = []
                for _i, (content_hash, url, timestamp) in enumerate(hash_data):
                    if content_hash in existing_hashes:
                        results.append((True, content_hash))  # Duplicate
                        self._duplicate_hits += 1
                    else:
                        results.append((False, content_hash))  # Unique
                        new_records.append((content_hash, url, timestamp))
                        self._unique_inserts += 1

                    self._total_checks += 1

                # Insert new records
                if new_records:
                    await conn.executemany(
                        "INSERT OR IGNORE INTO hash_dedup (hash, url, timestamp) VALUES (?, ?, ?)", new_records
                    )
                    await conn.commit()

                logger.debug(
                    f"Batch processed {len(contents)} items: {len(new_records)} new, {len(existing_hashes)} duplicates"
                )

        except Exception as e:
            logger.error(f"Batch hash check error: {e}")
            # On error, return all as unique to avoid blocking
            results = [(False, h[0]) for h in hash_data]

        return results

    async def get_hash_info(self, content_hash: str) -> Optional[dict]:
        """
        Get information about a hash.

        Args:
            content_hash: SHA-256 hash to look up

        Returns:
            Dict with hash info or None if not found
        """
        try:
            async with self._get_connection() as conn:
                cursor = await conn.execute(
                    "SELECT hash, url, timestamp, created_at FROM hash_dedup WHERE hash = ?", (content_hash,)
                )
                row = await cursor.fetchone()

                if row:
                    return {"hash": row[0], "url": row[1], "timestamp": row[2], "created_at": row[3]}
                return None

        except Exception as e:
            logger.error(f"Hash lookup error: {e}")
            return None

    async def cleanup_old_hashes(self, days: int = 90) -> int:
        """
        Clean up hash records older than specified days.

        Args:
            days: Age threshold in days

        Returns:
            Number of records deleted
        """
        cutoff_time = time.time() - (days * 24 * 3600)

        try:
            async with self._get_connection() as conn:
                cursor = await conn.execute("DELETE FROM hash_dedup WHERE timestamp < ?", (cutoff_time,))
                await conn.commit()

                deleted_count = cursor.rowcount
                logger.info(f"Cleaned up {deleted_count} hash records older than {days} days")
                return deleted_count

        except Exception as e:
            logger.error(f"Hash cleanup error: {e}")
            return 0

    async def get_stats(self) -> dict:
        """Get database statistics."""
        stats = {
            "total_checks": self._total_checks,
            "duplicate_hits": self._duplicate_hits,
            "unique_inserts": self._unique_inserts,
            "duplicate_rate": self._duplicate_hits / max(1, self._total_checks),
            "db_path": str(self.db_path),
            "wal_mode": self.wal_mode,
        }

        try:
            async with self._get_connection() as conn:
                # Get total record count
                cursor = await conn.execute("SELECT COUNT(*) FROM hash_dedup")
                total_records = (await cursor.fetchone())[0]
                stats["total_records"] = total_records

                # Get database size
                cursor = await conn.execute("PRAGMA page_count")
                page_count = (await cursor.fetchone())[0]
                cursor = await conn.execute("PRAGMA page_size")
                page_size = (await cursor.fetchone())[0]
                stats["db_size_bytes"] = page_count * page_size

        except Exception as e:
            logger.error(f"Stats query error: {e}")

        return stats

    async def close(self) -> None:
        """Close all database connections."""
        async with self._pool_lock:
            for conn in self._connection_pool:
                await conn.close()
            self._connection_pool.clear()

        logger.info("Hash database connections closed")


def calculate_content_hash(content: str) -> str:
    """
    Calculate SHA-256 hash of content.

    Args:
        content: Content to hash

    Returns:
        Hexadecimal SHA-256 hash
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
