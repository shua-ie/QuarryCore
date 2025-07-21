"""
Manages the SQLite database for hot metadata storage.
"""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import aiosqlite
import structlog
from sqlalchemy import create_engine
from sqlalchemy.schema import MetaData

from quarrycore.config.config import SQLiteConfig

from .schema import metadata as db_metadata

logger = structlog.get_logger(__name__)

# The current version of the database schema.
# This should be incremented whenever the schema in schema.py changes.
CURRENT_SCHEMA_VERSION = 1


class SQLiteManager:
    """Handles all interactions with the SQLite database."""

    def __init__(self, config: SQLiteConfig):
        self.config = config
        self.db_path = Path(config.db_path)
        self._pool: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue(maxsize=config.pool_size)
        self._engine = create_engine(f"sqlite:///{self.db_path}")

    async def initialize(self) -> None:
        """Initializes the database, connection pool, and runs migrations."""
        for _ in range(self.config.pool_size):
            conn = await self._create_connection()
            await self._pool.put(conn)

        async with self.get_connection() as conn:
            await self._run_migrations(conn)

    async def _create_connection(self) -> aiosqlite.Connection:
        """Creates and configures a new database connection."""
        conn = await aiosqlite.connect(self.db_path)
        if self.config.wal_mode:
            await conn.execute("PRAGMA journal_mode=WAL;")
        await conn.execute("PRAGMA foreign_keys=ON;")
        await conn.execute("PRAGMA busy_timeout = 5000;")
        conn.row_factory = aiosqlite.Row
        return conn

    @asynccontextmanager
    async def get_connection(self) -> AsyncIterator[aiosqlite.Connection]:
        """Gets a connection from the pool."""
        conn = await self._pool.get()
        try:
            yield conn
        finally:
            await self._pool.put(conn)

    async def _run_migrations(self, conn: aiosqlite.Connection) -> None:
        """Checks schema version and applies migrations if necessary."""
        cursor = await conn.execute("PRAGMA user_version;")
        version_row = await cursor.fetchone()
        current_version = version_row[0] if version_row is not None else 0

        if current_version < CURRENT_SCHEMA_VERSION:
            logger.info(
                f"Database schema is out of date (v{current_version}). Migrating to v{CURRENT_SCHEMA_VERSION}..."
            )
            # For now, we just create all tables. A real migration system would be more complex.
            db_metadata.create_all(self._engine)

            # Create the FTS table
            await self._setup_fts(conn)

            await conn.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION};")
            await conn.commit()
            logger.info("Database migration complete.")

    async def _setup_fts(self, conn: aiosqlite.Connection) -> None:
        """Sets up the FTS5 table and triggers."""
        await conn.execute("DROP TABLE IF EXISTS content_fts;")
        await conn.execute(
            f"""
            CREATE VIRTUAL TABLE content_fts USING {self.config.fts_version}(
                title,
                description,
                content='processed_content',
                content_rowid='id'
            );
        """
        )
        # Triggers to keep FTS table in sync
        await conn.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS processed_content_ai AFTER INSERT ON processed_content BEGIN
              INSERT INTO content_fts(rowid, title, description) VALUES (new.id, new.title, new.description);
            END;
            CREATE TRIGGER IF NOT EXISTS processed_content_ad AFTER DELETE ON processed_content BEGIN
              INSERT INTO content_fts(content_fts, rowid, title, description) VALUES ('delete', old.id, old.title, old.description);
            END;
            CREATE TRIGGER IF NOT EXISTS processed_content_au AFTER UPDATE ON processed_content BEGIN
              INSERT INTO content_fts(content_fts, rowid, title, description) VALUES ('delete', old.id, old.title, old.description);
              INSERT INTO content_fts(rowid, title, description) VALUES (new.id, new.title, new.description);
            END;
        """
        )
        await conn.commit()

    async def store_batch(self, batch: List[dict[str, Any]]) -> None:
        """Stores a batch of metadata records in a single transaction."""
        if not batch:
            return

        # This is a simplified insert, assumes dict keys match column names
        # A more robust solution would use SQLAlchemy Core expressions
        keys = batch[0].keys()
        placeholders = ", ".join("?" for _ in keys)
        sql = f"INSERT INTO processed_content ({', '.join(keys)}) VALUES ({placeholders})"

        values = [tuple(item[k] for k in keys) for item in batch]

        async with self.get_connection() as conn:
            async with conn.cursor() as cursor:
                await cursor.executemany(sql, values)
            await conn.commit()

    async def search(self, query: str, limit: int = 20) -> List[aiosqlite.Row]:
        """Performs a full-text search."""
        sql = "SELECT rowid, title, description FROM content_fts WHERE content_fts MATCH ? ORDER BY rank LIMIT ?"
        async with self.get_connection() as conn:
            cursor = await conn.execute(sql, (query, limit))
            rows = await cursor.fetchall()
            return list(rows)

    async def backup(self, backup_path: Path) -> None:
        """Performs an online backup of the database."""
        logger.info(f"Starting database backup to {backup_path}...")
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        async with self.get_connection() as conn:
            await conn.backup(sqlite3.connect(str(backup_path)))

        logger.info("Database backup complete.")

    async def close(self) -> None:
        """Closes all connections in the pool."""
        while not self._pool.empty():
            conn = await self._pool.get()
            await conn.close()

    async def vacuum(self) -> None:
        """Rebuilds the database file, repacking it into a minimal amount of disk space."""
        logger.info("Starting database VACUUM operation...")

        # VACUUM requires exclusive access - create a dedicated connection
        # that bypasses the connection pool to avoid conflicts
        vacuum_conn = await self._create_connection()
        try:
            # Ensure no other operations are in progress
            await vacuum_conn.execute("BEGIN IMMEDIATE;")
            await vacuum_conn.rollback()  # Release the lock

            # Now perform VACUUM with exclusive access
            await vacuum_conn.execute("VACUUM;")
            logger.info("Database VACUUM complete.")
        finally:
            await vacuum_conn.close()

    async def get_content_metadata(self, content_id: int) -> Optional[Dict[str, Any]]:
        """Get content metadata by ID."""
        async with self.get_connection() as conn:
            cursor = await conn.execute("SELECT full_metadata FROM processed_content WHERE id = ?", (content_id,))
            row = await cursor.fetchone()

            if row is not None:
                # Convert Row to dictionary safely
                metadata_json = row[0] if row[0] is not None else "{}"
                try:
                    import json

                    metadata_dict: Dict[str, Any] = json.loads(metadata_json)
                    return metadata_dict
                except (json.JSONDecodeError, TypeError):
                    return {}
            return None

    async def get_recent_content(self, limit: int = 100) -> list[Dict[str, Any]]:
        """Get recent content from the database."""
        async with self.get_connection() as conn:
            cursor = await conn.execute("SELECT * FROM processed_content ORDER BY processed_at DESC LIMIT ?", (limit,))
            rows = await cursor.fetchall()

            # Convert rows to list of dictionaries
            return [dict(row) for row in rows]

    async def get_all_metadata(self) -> list[Dict[str, Any]]:
        """Get all content metadata."""
        async with self.get_connection() as conn:
            cursor = await conn.execute("SELECT id, full_metadata FROM processed_content")
            rows = await cursor.fetchall()

            # Convert to list of dictionaries
            result = []
            for row in rows:
                try:
                    import json

                    metadata = json.loads(row[1]) if row[1] else {}
                    metadata["id"] = row[0]
                    result.append(metadata)
                except (json.JSONDecodeError, TypeError):
                    result.append({"id": row[0], "error": "invalid_metadata"})

            return result
